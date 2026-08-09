"""Task orchestration and single-sample inference workflows for StructMapping."""

import json
import os
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from termcolor import cprint
from tqdm import tqdm

import confs.path as PathConfig
import model.structmapping.utils_model.utils_model_io as ModelIOUtils
import model.structmapping.utils_model.utils_uncond_gen as UncondGenUtils
import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import utils.utils_render as RenderUtils


@dataclass(frozen=True)
class _DatasetRewritingPreset:
    view_name: str
    mode_geo: str
    normal_mode: Optional[str]
    sample_name: str = 'sample{index}'


_DATASET_REWRITING_PRESETS = {
    'generation': _DatasetRewritingPreset(
        view_name='view_0500',
        mode_geo='complete',
        normal_mode=None,
        sample_name='sample_{index:03d}',
    ),
}


@dataclass(frozen=True)
class _CompletionSample:
    sample_id: str
    category: str
    partial_pointcloud: Path
    gt_mesh: Optional[Path]


@dataclass(frozen=True)
class _ReconstructionSample:
    sample_id: str
    category: str
    mesh: Path


def _manifest_identifier(value, field_name, line_number, manifest_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f'{manifest_name} manifest line {line_number}: {field_name} must be '
            'a non-empty string'
        )
    identifier = value.strip()
    if identifier in {'.', '..'} or '/' in identifier or '\\' in identifier:
        raise ValueError(
            f'{manifest_name} manifest line {line_number}: {field_name} must not '
            'contain path separators'
        )
    return identifier


def _resolve_manifest_input(
    manifest_dir,
    value,
    field_name,
    line_number,
    manifest_name,
):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f'{manifest_name} manifest line {line_number}: {field_name} must be '
            'a non-empty path string'
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'{manifest_name} manifest line {line_number}: {field_name} not found: '
            f'{path}'
        )
    return path


def _load_manifest_rows(path_manifest, manifest_name):
    manifest_path = Path(path_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'{manifest_name} manifest not found: {manifest_path}'
        )

    rows = []
    with manifest_path.open(encoding='utf-8') as manifest_file:
        for line_number, raw_line in enumerate(manifest_file, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'{manifest_name} manifest line {line_number}: invalid '
                    f'JSON: {error.msg}'
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f'{manifest_name} manifest line {line_number}: expected a '
                    'JSON object'
                )
            rows.append((line_number, row))

    if not rows:
        raise ValueError(f'{manifest_name} manifest is empty: {manifest_path}')
    return manifest_path, tuple(rows)


def _manifest_sample_identity(
    row,
    line_number,
    manifest_name,
    supported_categories,
    output_keys,
):
    sample_id = _manifest_identifier(
        row.get('id'), 'id', line_number, manifest_name
    )
    category = _manifest_identifier(
        row.get('category'), 'category', line_number, manifest_name
    ).casefold()
    if category not in supported_categories:
        raise ValueError(
            f'{manifest_name} manifest line {line_number}: unsupported '
            f'category: {category}'
        )

    output_key = (category, sample_id.casefold())
    if output_key in output_keys:
        raise ValueError(
            f'{manifest_name} manifest line {line_number}: duplicate '
            f'category/id output key: {category}/{sample_id}'
        )
    output_keys.add(output_key)
    return sample_id, category


def _validate_completion_pointcloud(path, line_number):
    try:
        with h5py.File(path, 'r') as pointcloud_file:
            required_keys = {'pointcloud', 'normal', 'lis_cam_sample'}
            missing_keys = sorted(
                required_keys.difference(pointcloud_file.keys())
            )
            if missing_keys:
                raise ValueError(
                    f'Completion manifest line {line_number}: '
                    f'{path} is missing HDF5 datasets: '
                    f'{", ".join(missing_keys)}'
                )

            datasets = {
                key: pointcloud_file[key]
                for key in required_keys
            }
            pointcloud_shape = datasets['pointcloud'].shape
            normal_shape = datasets['normal'].shape
            camera_shape = datasets['lis_cam_sample'].shape
            if (
                len(pointcloud_shape) != 2
                or pointcloud_shape[0] == 0
                or pointcloud_shape[-1] != 3
                or normal_shape != pointcloud_shape
            ):
                raise ValueError(
                    f'Completion manifest line {line_number}: pointcloud and '
                    f'normal must have matching non-empty [N, 3] shapes, got '
                    f'{pointcloud_shape} and {normal_shape}'
                )
            if (
                len(camera_shape) != 2
                or camera_shape[0] == 0
                or camera_shape[-1] != 3
            ):
                raise ValueError(
                    f'Completion manifest line {line_number}: lis_cam_sample '
                    f'must have non-empty shape [V, 3], got {camera_shape}'
                )

            nonnumeric_keys = sorted(
                key
                for key, dataset in datasets.items()
                if not np.issubdtype(dataset.dtype, np.number)
            )
            if nonnumeric_keys:
                raise ValueError(
                    f'Completion manifest line {line_number}: HDF5 datasets '
                    f'must be numeric: {", ".join(nonnumeric_keys)}'
                )
            arrays = {
                key: dataset[...]
                for key, dataset in datasets.items()
            }
    except OSError as error:
        raise ValueError(
            f'Completion manifest line {line_number}: cannot read HDF5 file '
            f'{path}: {error}'
        ) from error

    nonfinite_keys = sorted(
        key for key, value in arrays.items() if not np.isfinite(value).all()
    )
    if nonfinite_keys:
        raise ValueError(
            f'Completion manifest line {line_number}: HDF5 datasets must '
            f'contain only finite values: {", ".join(nonfinite_keys)}'
        )

    pointcloud = arrays['pointcloud']
    point_limit = 1.15
    if np.any(pointcloud < -point_limit) or np.any(pointcloud > point_limit):
        raise ValueError(
            f'Completion manifest line {line_number}: pointcloud coordinates '
            f'must be normalized to [-{point_limit}, {point_limit}]'
        )


def _load_completion_manifest(path_manifest):
    manifest_name = 'Completion'
    manifest_path, manifest_rows = _load_manifest_rows(
        path_manifest, manifest_name
    )
    samples = []
    output_keys = set()
    supported_categories = set(
        PathConfig.Categorys_PartNet + PathConfig.Categorys_ShapeNet
    )
    for line_number, row in manifest_rows:
        sample_id, category = _manifest_sample_identity(
            row,
            line_number,
            manifest_name,
            supported_categories,
            output_keys,
        )
        partial_pointcloud = _resolve_manifest_input(
            manifest_path.parent,
            row.get('partial_pointcloud'),
            'partial_pointcloud',
            line_number,
            manifest_name,
        )
        if partial_pointcloud.suffix.lower() != '.h5':
            raise ValueError(
                f'{manifest_name} manifest line {line_number}: '
                'partial_pointcloud must be an .h5 file'
            )
        _validate_completion_pointcloud(partial_pointcloud, line_number)

        gt_mesh_value = row.get('gt_mesh')
        gt_mesh = None
        if gt_mesh_value is not None:
            gt_mesh = _resolve_manifest_input(
                manifest_path.parent,
                gt_mesh_value,
                'gt_mesh',
                line_number,
                manifest_name,
            )
            if gt_mesh.suffix.lower() != '.ply':
                raise ValueError(
                    f'{manifest_name} manifest line {line_number}: gt_mesh '
                    'must be a .ply file'
                )

        samples.append(
            _CompletionSample(
                sample_id=sample_id,
                category=category,
                partial_pointcloud=partial_pointcloud,
                gt_mesh=gt_mesh,
            )
        )

    return manifest_path, tuple(samples)


def _load_reconstruction_manifest(path_manifest, dataset_name=None):
    manifest_name = 'Reconstruction'
    manifest_path, manifest_rows = _load_manifest_rows(
        path_manifest, manifest_name
    )
    category_sets = {
        'partnet': set(PathConfig.Categorys_PartNet),
        'shapenet': set(PathConfig.Categorys_ShapeNet),
    }
    if dataset_name is None:
        supported_categories = set().union(*category_sets.values())
    else:
        try:
            supported_categories = category_sets[dataset_name]
        except KeyError as error:
            raise ValueError(
                f'Unsupported reconstruction dataset: {dataset_name}'
            ) from error

    samples = []
    output_keys = set()
    for line_number, row in manifest_rows:
        sample_id, category = _manifest_sample_identity(
            row,
            line_number,
            manifest_name,
            supported_categories,
            output_keys,
        )
        mesh = _resolve_manifest_input(
            manifest_path.parent,
            row.get('mesh'),
            'mesh',
            line_number,
            manifest_name,
        )
        if mesh.suffix.lower() != '.ply':
            raise ValueError(
                f'{manifest_name} manifest line {line_number}: mesh must be '
                'a .ply file'
            )
        samples.append(
            _ReconstructionSample(
                sample_id=sample_id,
                category=category,
                mesh=mesh,
            )
        )

    return manifest_path, tuple(samples)


_TASK_WORKFLOWS = {
    'generation_noise_sweep': (
        '_run_dataset_rewriting',
        (('preset_name', 'generation'),),
    ),
    'completion': ('_run_completion_manifest', ()),
    'reconstruction': ('_run_reconstruction_manifest', ()),
    'reconstruction_shapenet_batch': (
        '_run_reconstruction_manifest',
        (('dataset_name', 'shapenet'),),
    ),
    'reconstruction_partnet_batch': (
        '_run_reconstruction_manifest',
        (('dataset_name', 'partnet'),),
    ),
    'mesh_cycle': (
        '_run_mesh_rewriting',
        (('use_cycle_infer', True),),
    ),
    'mesh_reconstruct': (
        '_run_mesh_rewriting',
        (('use_cycle_infer', False),),
    ),
    'mesh_encode': ('_run_mesh_encoding', ()),
    'shape_code_generate': (
        '_run_shape_code_decoding',
        (('source_mode', 'distribution'),),
    ),
    'shape_code_interpolate': (
        '_run_shape_code_decoding',
        (('source_mode', 'interpolation'),),
    ),
}
_MISSING = object()


def _prepare_cycle_root_input(data_input, shape_output):
    root_bboxes = np.asarray(shape_output["lis_bbox_centerlwh"])
    root_codes = shape_output["lis_part_codes"].detach()
    if root_codes.ndim != 3:
        raise ValueError(f"Expected root codes shaped [T, 1, C], got {root_codes.shape}")
    if len(root_codes) != len(root_bboxes):
        raise ValueError(
            f"Cycle root code/bbox count mismatch: {len(root_codes)} != {len(root_bboxes)}"
        )

    cycle_input = {}
    for key, value in data_input.items():
        cycle_input[key] = value.detach().clone() if torch.is_tensor(value) else copy(value)

    cycle_input["bb_child"] = torch.as_tensor(
        root_bboxes[:, :6], dtype=root_codes.dtype, device=root_codes.device
    ).unsqueeze(1)
    cycle_input["code_child"] = root_codes.unsqueeze(1)
    cycle_input["code_idx_child"] = torch.zeros(
        (*root_codes.shape[:2], 1), dtype=torch.long, device=root_codes.device
    )
    cycle_input["idx_dir_token"] = torch.ones_like(cycle_input["idx_dir_token"])
    return cycle_input, root_bboxes.copy()


@contextmanager
def _temporary_mesh_inference_state(model):
    fields = (
        (model, 'mapping_direction_infer'),
        (model.args, 'mapping_direction_infer'),
        (model.args, 'struct_add_partcode_noise'),
        (model.args, 'mapping_use_patch_condition_last_step'),
    )
    previous = [
        (target, name, getattr(target, name, _MISSING))
        for target, name in fields
    ]
    try:
        yield
    finally:
        for target, name, value in previous:
            if value is _MISSING:
                if hasattr(target, name):
                    delattr(target, name)
            else:
                setattr(target, name, value)


def _load_shape_code_samples(args, source_mode):
    if source_mode == 'distribution':
        return UncondGenUtils.read_lis_part_code(
            path_shape_code=args.mapping_shape_code_path,
        )
    if source_mode != 'interpolation':
        raise ValueError(f'Unsupported shape-code source mode: {source_mode}')

    codes = UncondGenUtils.read_lis_part_code_only(
        path_shape_code=args.mapping_shape_code_path,
    )
    source_key = args.mapping_shape_code_source
    target_key = args.mapping_shape_code_target
    missing_keys = [key for key in (source_key, target_key) if key not in codes]
    if missing_keys:
        raise KeyError(
            'Shape-code interpolation keys not found: '
            + ', '.join(missing_keys)
        )
    if args.mapping_shape_code_steps < 2:
        raise ValueError('--mapping_shape_code_steps must be at least 2')

    source = np.asarray(codes[source_key])
    target = np.asarray(codes[target_key])
    while source.ndim > 1 and source.shape[0] == 1:
        source = source[0]
    while target.ndim > 1 and target.shape[0] == 1:
        target = target[0]
    if source.shape != target.shape:
        raise ValueError(
            f'Interpolation code shapes differ: {source.shape} != '
            f'{target.shape}'
        )

    weight_shape = (args.mapping_shape_code_steps,) + (1,) * source.ndim
    weights = np.linspace(
        0,
        1,
        args.mapping_shape_code_steps,
    ).reshape(weight_shape)
    return source[None, ...] * (1 - weights) + target[None, ...] * weights


class StructMappingInferenceMixin:
    def run_inference_task(self, task_name, dir_exp, args):
        """Run one canonical inference task through the model workflow."""
        try:
            method_name, options = _TASK_WORKFLOWS[task_name]
        except KeyError as error:
            raise ValueError(
                f'Unsupported StructMapping inference task: {task_name}'
            ) from error
        return getattr(self, method_name)(
            dir_exp=dir_exp,
            args=args,
            **dict(options),
        )

    def _run_dataset_rewriting(self, dir_exp, args, preset_name):
        """Run one filelist-backed dataset preset through the shared kernel."""
        preset = _DATASET_REWRITING_PRESETS[preset_name]
        dataset_name = 'partnet'
        data_root = PathConfig.DIR_ROOT_PARTNET

        category = args.category
        filelist_path = (
            f'{data_root}/filelists/{category}_{args.extract_mode}.lst'
        )
        model_names = IOUtils.read_txt_to_list(filelist_path)
        stop = args.model_id_end if args.model_id_end > 0 else None
        model_names = model_names[args.model_id_start:stop]
        if not model_names:
            raise ValueError(f'The selected model range is empty: {filelist_path}')
        print(
            f'[{dataset_name}/{category}] Number of models to process: '
            f'{len(model_names)}'
        )

        n_samples = (
            self.args.mapping_n_iter_rand
            if self.args.mapping_n_iter_rand is not None
            else 1
        )
        cprint(f'Number of random sampling: {n_samples}.', 'yellow')
        failures = []
        processed = []
        output_root = Path(dir_exp).parent
        for sample_index in tqdm(range(n_samples)):
            sample_dir = str(
                output_root
                / preset.sample_name.format(index=sample_index)
            )
            print(f'Exp dir: {sample_dir}')
            for model_name in tqdm(model_names):
                aug_name = 'aug_0000'
                print(
                    f'\n\n\n Processing: model {category}, {model_name}; '
                    f'view, {preset.view_name}'
                )
                try:
                    self._rewrite_one_mesh(
                        cat_curr=category,
                        model_name=model_name,
                        aug_name=aug_name,
                        view_name=preset.view_name,
                        mode_geo=preset.mode_geo,
                        dir_exp_test=sample_dir,
                        path_mesh_test=None,
                        inp_mode_normal=(
                            preset.normal_mode
                            if preset.normal_mode is not None
                            else args.mapping_inp_mode_normal
                        ),
                        path_input_mesh=None,
                        path_input_pointcloud=None,
                        dir_data_root=data_root,
                    )
                    processed.append((sample_index, model_name))
                except Exception as error:
                    message = (
                        f'sample {sample_index}, {model_name}: '
                        f'{type(error).__name__}: {error}'
                    )
                    print(f'Error in: {message}')
                    failures.append(message)

        if failures:
            raise RuntimeError(
                f'Failed to rewrite {len(failures)} of '
                f'{len(processed) + len(failures)} dataset runs:\n'
                + '\n'.join(failures)
            )
        return {'processed': processed, 'failed': failures}

    def _run_completion_manifest(self, dir_exp, args):
        """Complete every partial observation from an explicit JSONL manifest."""
        manifest_path, samples = _load_completion_manifest(
            args.mapping_input_manifest
        )
        stop = args.model_id_end if args.model_id_end > 0 else None
        samples = samples[args.model_id_start:stop]
        if not samples:
            raise ValueError('The selected completion manifest range is empty')

        n_samples = args.mapping_n_iter_rand
        if n_samples is None:
            n_samples = 1
        if n_samples < 1:
            raise ValueError('--mapping_n_iter_rand must be at least 1')

        failures = []
        processed = []
        output_root = Path(dir_exp)
        for sample_index in tqdm(range(n_samples)):
            sample_dir = str(
                output_root
                if n_samples == 1
                else output_root.parent / f'sample{sample_index:03d}'
            )
            print(f'Exp dir: {sample_dir}')
            for sample in tqdm(samples):
                print(
                    f'\n\n\n Processing completion: '
                    f'{sample.category}/{sample.sample_id}'
                )
                try:
                    self._rewrite_one_mesh(
                        cat_curr=sample.category,
                        model_name=sample.sample_id,
                        aug_name='aug_0000',
                        view_name='view_0010',
                        mode_geo='fps',
                        dir_exp_test=sample_dir,
                        path_mesh_test=None,
                        inp_mode_normal='estimated_normal',
                        path_input_mesh=(
                            str(sample.gt_mesh)
                            if sample.gt_mesh is not None
                            else None
                        ),
                        path_input_pointcloud=str(sample.partial_pointcloud),
                        dir_data_root=str(manifest_path.parent),
                        use_input_contract=True,
                    )
                    processed.append(
                        (sample_index, sample.category, sample.sample_id)
                    )
                except Exception as error:
                    message = (
                        f'sample {sample_index}, {sample.category}/'
                        f'{sample.sample_id}: {type(error).__name__}: {error}'
                    )
                    print(f'Error in: {message}')
                    failures.append(message)

        if failures:
            raise RuntimeError(
                f'Failed to complete {len(failures)} of '
                f'{len(processed) + len(failures)} manifest runs:\n'
                + '\n'.join(failures)
            )
        return {'processed': processed, 'failed': failures}


    def _run_reconstruction_manifest(
        self,
        dir_exp,
        args,
        dataset_name=None,
    ):
        """Reconstruct every mesh from an explicit JSONL manifest."""
        manifest_path, samples = _load_reconstruction_manifest(
            args.mapping_input_manifest,
            dataset_name=dataset_name,
        )
        stop = args.model_id_end if args.model_id_end > 0 else None
        samples = samples[args.model_id_start:stop]
        if not samples:
            raise ValueError(
                'The selected reconstruction manifest range is empty'
            )

        n_samples = args.mapping_n_iter_rand
        if n_samples is None:
            n_samples = 1
        if n_samples < 1:
            raise ValueError('--mapping_n_iter_rand must be at least 1')

        failures = []
        processed = []
        output_root = Path(dir_exp)
        for sample_index in tqdm(range(n_samples)):
            sample_dir = str(
                output_root
                if n_samples == 1
                else output_root.parent / f'sample{sample_index:03d}'
            )
            print(f'Exp dir: {sample_dir}')
            for sample in tqdm(samples):
                print(
                    f'Processing reconstruction: '
                    f'{sample.category}/{sample.sample_id}'
                )
                try:
                    with _temporary_mesh_inference_state(self):
                        self.mapping_direction_infer = 0
                        self.args.mapping_direction_infer = 0
                        self.args.struct_add_partcode_noise = None
                        self.args.mapping_use_patch_condition_last_step = False
                        self._rewrite_one_mesh(
                            cat_curr=sample.category,
                            model_name=sample.sample_id,
                            aug_name='aug_0000',
                            view_name='view_0000',
                            mode_geo='complete',
                            dir_exp_test=sample_dir,
                            path_mesh_test=str(sample.mesh),
                            inp_mode_normal='gt_normal',
                            path_input_mesh=str(sample.mesh),
                            path_input_pointcloud=None,
                            dir_data_root=str(manifest_path.parent),
                        )
                    processed.append((
                        sample_index,
                        sample.category,
                        sample.sample_id,
                    ))
                except Exception as error:
                    message = (
                        f'{sample.category}/{sample.sample_id}: '
                        f'{type(error).__name__}: {error}'
                    )
                    print(f'Error in: {message}')
                    failures.append(message)

        if failures:
            raise RuntimeError(
                f"Failed to reconstruct {len(failures)} of "
                f"{len(processed) + len(failures)} manifest runs:\n"
                + "\n".join(failures)
            )
        return {"processed": processed, "failed": failures}

    def _rewrite_one_mesh(
        self,
        *,
        cat_curr,
        model_name,
        aug_name,
        mode_geo,
        view_name,
        dir_exp_test,
        path_mesh_test,
        inp_mode_normal,
        path_input_mesh,
        path_input_pointcloud,
        dir_data_root,
        use_input_contract=False,
        use_cycle_infer=False,
        is_infer_part_code=False,
    ):



        if dir_exp_test is None:
            raise ValueError('dir_exp_test is required')

        path_partial = str(
            Path(dir_exp_test)
            / 'partial_inputs'
            / cat_curr
            / model_name
            / aug_name
            / f'{view_name}.ply'
        )
        path_output_pointcloud = str(
            Path(dir_exp_test)
            / 'partial_inputs'
            / cat_curr
            / model_name
            / aug_name
            / f'{view_name}.h5'
        )
        dict_conf = {
            'mode_geo': mode_geo,
            'inp_mode_normal': inp_mode_normal,
            'path_input_pointcloud': path_input_pointcloud,
            'path_output_pointcloud': path_output_pointcloud,
            'path_partial': path_partial,
        }
        path_mesh = f'{dir_data_root}/geometry/mesh_ply/{cat_curr}/{model_name}.ply'
        if cat_curr =='Airplane':
            path_mesh = f'{dir_data_root}/geometry/mesh/{cat_curr}/{model_name}/{aug_name}.ply'
        if cat_curr in PathConfig.Categorys_ShapeNet:
            path_mesh = f'{dir_data_root}/geometry/mesh/{cat_curr}/{model_name}/{aug_name}.ply'

        if path_mesh_test is not None:
            path_mesh2 = path_mesh_test
        elif path_input_mesh is not None:
            path_mesh2 = path_input_mesh
        elif use_input_contract:
            path_mesh2 = None
        else:
            path_mesh2 = path_mesh
        if use_input_contract or path_mesh_test is not None:
            path_pair = str(
                Path(dir_exp_test)
                / 'input_contract'
                / cat_curr
                / model_name
                / aug_name
                / view_name
                / 'hierarchy_pair_0000.h5'
            )
        else:
            path_pair = (
                f'{dir_data_root}/{PathConfig.Folder_Name_Hierarchy_Sparse_VAE}'
                f'/{cat_curr}/{model_name}/{aug_name}/{view_name}'
                f'/hierarchy_pair_0000.h5'
            )
        if path_mesh2 is not None and not os.path.exists(path_mesh2):
            raise FileNotFoundError(f'Input mesh not found: {path_mesh2}')
        if path_mesh2 is None and not (
            path_input_pointcloud and os.path.exists(path_input_pointcloud)
        ):
            raise FileNotFoundError(
                'An input mesh or existing input pointcloud is required'
            )

        dict_code_sparse_vae = self.infer_sparse_vae_code_single_mesh(
            path_mesh2,
            dict_conf=dict_conf,
        )
        if 'code_enc' in dict_code_sparse_vae:
            del dict_code_sparse_vae['code_enc']


        should_render_mesh = (
            path_mesh_test is None and not use_input_contract
        )
        dict_pair_part = {}; dict_info_shape_inp = None

        if self.mapping_direction_infer == 1:
            t3 = datetime.now()
            sparse_vae_batch = {}
            for key in dict_code_sparse_vae:
                if key=='code_idx':
                    dict_code_sparse_vae[key] = np.zeros((1,1))
                sparse_vae_batch[key] = torch.from_numpy(dict_code_sparse_vae[key][None, ...])
            dict_out_memory_part_vae = self.part_vae.infer_part_vae_memory_mesh(sparse_vae_batch)

            lis_code_infer = dict_out_memory_part_vae['code'].transpose(0,1)

            if is_infer_part_code:
                return lis_code_infer

            if self.args.struct_add_partcode_noise is None:
                raise ValueError(
                    "Direction-1 mesh rewriting requires part-code noise; "
                    "use mesh_encode to extract the unmodified shape code"
                )

            # get bbox
            box_curr = GeoUtils.get_aabb(GeoUtils.read_triangle_mesh(path_mesh2), return_6Dbox=True)
            box_lwh = GeoUtils.get_lwh_from_aabb(box_curr[None, ...])

            dict_pair_part = {
                'code_child': lis_code_infer[None, ...],
                'code_idx_child': torch.zeros((1,1,1)).long(),
                'bb_child': torch.from_numpy(box_lwh).float()[None, ...],
                'idx_dir_token': torch.ones((1,1)).long()
            }


            t4 = datetime.now()
            t_part_enc = (t4-t3).total_seconds()
            print(f'[Part Enc] Time: {t_part_enc:.04f}')

            print(f'Add noise to part code')
            max_thres_disturb = self.args.struct_add_partcode_noise

            noise_disturb = ModelIOUtils.perturb_code(lis_code_infer, max_thres_disturb)
            lis_code_infer = lis_code_infer + noise_disturb

            patch_output, lis_mesh = self.infer_partcode_2_patchcode(
                lis_code_infer,
                mode_decode_sdf='dense',
            )

            path_save_part = str(
                Path(dir_exp_test)
                / 'generated_noise_parts'
                / cat_curr
                / f'{model_name}_{aug_name}_{view_name}.ply'
            )
            IOUtils.ensure_pdir_existence(path_save_part)
            GeoUtils.write_triangle_mesh(path_save_part, lis_mesh[0])

            if should_render_mesh:
                lis_img_render_curr = RenderUtils.render_mesh(self.renderer, mesh_p3d=None, mesh_o3d=lis_mesh[0], path_mesh=None)

                path_mesh_gt = f'{dir_data_root}/{PathConfig.Folder_Mesh_Aug}/{cat_curr}/{model_name}/{aug_name}.ply'
                lis_img_render_gt = RenderUtils.render_mesh(self.renderer, mesh_p3d=None, mesh_o3d=None, path_mesh=path_mesh_gt)
                img_render_gt = lis_img_render_gt[0]

            else:
                lis_img_render_curr = [None]
                img_render_gt = None


            patch_bb_child = ModelIOUtils.convert_code_coords2bbox(patch_output['code_coords'].cpu().numpy()[0])[None, ...].to(lis_code_infer.device)


            patch_code_child = patch_output['code']
            patch_code_idx_child = torch.zeros(
                (1, patch_code_child.shape[1], 1),
                dtype=torch.long,
                device=patch_code_child.device,
            )

            dict_pair_part.update({
                'patch_bb_child': patch_bb_child,
                'patch_code_child': patch_code_child.unsqueeze(-2),
                'patch_code_idx_child': patch_code_idx_child,
            })


            dict_info_shape_inp = {
                'mesh_merge': lis_mesh[0],

                'lis_bbox_centerlwh': [box_lwh[0]],
                'lis_img_part': [lis_img_render_curr[0]],
                'lis_mesh': lis_mesh,

                'img_gt': img_render_gt,
                'img_part_merge': lis_img_render_curr[0],

                'lis_code_noise': noise_disturb.cpu().numpy()
            }





        dict_code_sparse_vae['path_pair'] = path_pair
        dict_pair = self.prepare_hierarchy_pair_mapping(dict_code_sparse_vae)
        dict_pair['path'] = [path_pair]
        dict_pair.update(dict_pair_part)
        self.dict_data_inp = self.to_cuda(dict_pair)


        _, dict_out_shape_last_hier = self.validate_hierarchy(dir_exp_test,
                                                                dict_data_inp_hier=self.dict_data_inp,
                                                                should_validate_hierarchy=True,
                                                                dict_info_shape_inp=dict_info_shape_inp,
                                                                should_render_mesh=should_render_mesh)



        if use_cycle_infer:
            self.mapping_direction_infer = 1
            if self.args.struct_add_partcode_noise is None:
                raise ValueError(
                    'mesh_cycle requires --struct_add_partcode_noise'
                )
            cprint(
                'Use cycle inference with noise: '
                f'{self.args.struct_add_partcode_noise}',
                'red',
            )
            self.args.mapping_use_patch_condition_last_step = True

            tag_noise = (
                f'noise{self.args.struct_add_partcode_noise:.02f}'
                .replace('.', '_')
            )
            dict_data_inp_hier, lis_bbox_lwh_np = _prepare_cycle_root_input(
                self.dict_data_inp,
                dict_out_shape_last_hier,
            )
            view_name_curr = f'{view_name}_{tag_noise}_sample00'
            path_pair_new = str(
                Path(dir_exp_test)
                / 'input_contract'
                / cat_curr
                / model_name
                / aug_name
                / view_name_curr
                / 'hierarchy_pair_00010.h5'
            )
            dict_data_inp_hier['path'] = [path_pair_new]
            path_root_perturb = str(
                Path(dir_exp_test)
                / 'mesh_merge'
                / cat_curr
                / model_name
                / aug_name
                / view_name_curr
                / 'perturb_sample.ply'
            )
            self._run_cycle_sample(
                dir_exp_test,
                deepcopy(dict_data_inp_hier),
                lis_bbox_lwh_np,
                path_root_perturb,
            )

    def _run_cycle_sample(
        self,
        dir_exp,
        dict_data_inp_hier,
        lis_bbox_lwh_np,
        path_root_perturb,
    ):
        # Add noise on root node
        lis_code_tn = dict_data_inp_hier['code_child']

        cprint(f'Add noise on root node: {self.args.struct_add_partcode_noise}', 'red')
        code_noise_perturb = ModelIOUtils.perturb_code(lis_code_tn, self.args.struct_add_partcode_noise)
        lis_code_tn += code_noise_perturb


        # update patch code
        dict_out_parts = self.decode_lis_partcode2mesh(lis_partcode=lis_code_tn.squeeze(dim=1),
                                                            lis_bbox_lwh=lis_bbox_lwh_np,
                                                            dict_info_memory_part_vae=None)

        # use patch condition of last step
        cprint(f'Using patch condtion of last step', 'yellow')
        dict_conf_infer_patch = {
            'mesh_inp': dict_out_parts['mesh_merge'],
            'mode_geo': 'complete',
            'inp_mode_normal': 'gt_normal'
        }
        dict_pair_patch_new = self.prepare_pair_patch_from_mesh(path_mesh=None, dict_conf=dict_conf_infer_patch)
        dict_data_inp_hier.update(dict_pair_patch_new)

        dict_info_shape_inp = dict_out_parts

        GeoUtils.write_triangle_mesh(path_root_perturb, dict_out_parts['mesh_merge'])


        self.validate_hierarchy(dir_exp, dict_data_inp_hier=dict_data_inp_hier,
                                    should_validate_hierarchy=True,
                                    dict_info_shape_inp=dict_info_shape_inp,
                                    should_render_mesh=True)

    def _run_mesh_rewriting(
        self,
        dir_exp,
        args,
        use_cycle_infer,
    ):
        """Rewrite every selected PLY with optional direction-1 cycling."""
        category = args.category
        supported_categories = set(
            PathConfig.Categorys_PartNet + PathConfig.Categorys_ShapeNet
        )
        if category not in supported_categories:
            raise ValueError(f'Unsupported mesh category: {category}')

        mesh_dir = Path(args.mapping_input_mesh_dir)
        if not mesh_dir.is_dir():
            raise FileNotFoundError(f'Mesh directory not found: {mesh_dir}')
        mesh_paths = sorted(
            path
            for path in mesh_dir.iterdir()
            if path.is_file() and path.suffix.lower() == '.ply'
        )
        if not mesh_paths:
            raise ValueError(f'No .ply meshes found in {mesh_dir}')
        if args.model_id_end > 0:
            mesh_paths = mesh_paths[args.model_id_start:args.model_id_end]
        else:
            mesh_paths = mesh_paths[args.model_id_start:]
        if not mesh_paths:
            raise ValueError('The selected mesh range is empty')
        mesh_by_output_name = {}
        for mesh_path in mesh_paths:
            output_name = mesh_path.stem.casefold()
            previous = mesh_by_output_name.get(output_name)
            if previous is not None:
                raise ValueError(
                    'Mesh output name collision: '
                    f'{previous.name} and {mesh_path.name}'
                )
            mesh_by_output_name[output_name] = mesh_path

        print(f'[{category}] Number of meshes to process: {len(mesh_paths)}')
        n_samples = (
            self.args.mapping_n_iter_rand
            if self.args.mapping_n_iter_rand is not None
            else 1
        )
        if n_samples < 1:
            raise ValueError('--mapping_n_iter_rand must be at least 1')
        cprint(f'Number of random sampling: {n_samples}.', 'yellow')

        failures = []
        processed = []
        for sample_index in tqdm(range(n_samples)):
            output_root = Path(dir_exp)
            sample_dir = str(
                output_root
                if n_samples == 1
                else output_root.parent / f'sample{sample_index:03d}'
            )
            print(f'Exp dir: {sample_dir}')

            for mesh_path in tqdm(mesh_paths):
                mesh_name = mesh_path.stem
                model_name = f'0000_single_{mesh_name}_complete'
                aug_name = 'aug_0000'
                view_name = 'view_0000'
                print(
                    f'\n\n\n Processing: model {category}, {model_name}. '
                    f'{aug_name}/{view_name}'
                )

                try:
                    with _temporary_mesh_inference_state(self):
                        self.mapping_direction_infer = 0
                        self.args.mapping_direction_infer = 0
                        self.args.struct_add_partcode_noise = (
                            args.struct_add_partcode_noise
                            if use_cycle_infer
                            else None
                        )
                        self.args.mapping_use_patch_condition_last_step = False
                        self._rewrite_one_mesh(
                            cat_curr=category,
                            model_name=model_name,
                            aug_name=aug_name,
                            view_name=view_name,
                            mode_geo='complete',
                            dir_exp_test=sample_dir,
                            path_mesh_test=str(mesh_path),
                            inp_mode_normal=(
                                args.mapping_inp_mode_normal or 'gt_normal'
                            ),
                            path_input_mesh=str(mesh_path),
                            path_input_pointcloud=None,
                            dir_data_root=str(mesh_dir),
                            use_cycle_infer=use_cycle_infer,
                        )
                    processed.append(str(mesh_path))
                except Exception as error:
                    message = (
                        f'{mesh_path}: {type(error).__name__}: {error}'
                    )
                    print(f'Error in: {message}')
                    failures.append(message)

        print(f'Failed models: {failures}')
        if failures:
            raise RuntimeError(
                f'Failed to rewrite {len(failures)} of '
                f'{len(processed) + len(failures)} mesh runs:\n'
                + '\n'.join(failures)
            )
        return {'processed': processed, 'failed': failures}

    def _run_mesh_encoding(self, dir_exp, args):
        """Encode each selected PLY into one shape-code HDF5 collection."""
        category = args.category
        mesh_dir = Path(args.mapping_input_mesh_dir)
        output_path = args.mapping_shape_code_path
        if not mesh_dir.is_dir():
            raise FileNotFoundError(f'Mesh directory not found: {mesh_dir}')

        n_samples = (
            self.args.mapping_n_iter_rand
            if self.args.mapping_n_iter_rand is not None
            else 1
        )
        if n_samples != 1:
            raise ValueError(
                'mesh_encode requires mapping_n_iter_rand=1 because each '
                'mesh stem maps to exactly one output code'
            )

        print(f'Exp dir: {dir_exp}')
        mesh_paths = sorted(
            path
            for path in mesh_dir.iterdir()
            if path.is_file() and path.suffix.lower() == '.ply'
        )
        if not mesh_paths:
            raise ValueError(f'No .ply meshes found in {mesh_dir}')

        seen_stems = set()
        duplicate_stems = set()
        for mesh_path in mesh_paths:
            stem = mesh_path.stem.casefold()
            if stem in seen_stems:
                duplicate_stems.add(stem)
            seen_stems.add(stem)
        if duplicate_stems:
            raise ValueError(
                'Duplicate mesh stems would overwrite shape codes: '
                f'{", ".join(sorted(duplicate_stems))}'
            )

        failures = []
        shape_codes = {}
        for mesh_path in tqdm(mesh_paths):
            mesh_name = mesh_path.stem
            model_name = f'0000_single_{mesh_name}_complete'
            print(
                f'Processing mesh encoding: {category}/{mesh_name}'
            )
            try:
                with _temporary_mesh_inference_state(self):
                    self.mapping_direction_infer = 1
                    self.args.mapping_direction_infer = 1
                    self.args.struct_add_partcode_noise = None
                    self.args.mapping_use_patch_condition_last_step = False
                    output = self._rewrite_one_mesh(
                        cat_curr=category,
                        model_name=model_name,
                        aug_name='aug_0000',
                        view_name='view_0000_test0',
                        mode_geo='complete',
                        dir_exp_test=dir_exp,
                        path_mesh_test=str(mesh_path),
                        inp_mode_normal='gt_normal',
                        path_input_mesh=str(mesh_path),
                        path_input_pointcloud=None,
                        dir_data_root=str(mesh_dir),
                        use_cycle_infer=False,
                        is_infer_part_code=True,
                    )
                    shape_codes[mesh_name] = (
                        output.detach().cpu().numpy()
                    )
            except Exception as error:
                message = (
                    f'{model_name}: {type(error).__name__}: {error}'
                )
                print(f'Error in: {message}')
                failures.append(message)

        if failures:
            raise RuntimeError(
                f'Failed to extract shape codes for {len(failures)} meshes: '
                + ', '.join(failures)
            )

        IOUtils.ensure_pdir_existence(output_path)
        IOUtils.write_dict_to_h5_recursively(output_path, shape_codes)
        return {
            'processed': list(shape_codes),
            'failed': failures,
            'output': output_path,
        }

    def _run_shape_code_decoding(self, dir_exp, args, source_mode):
        """Decode distribution samples or an interpolation through one loop."""
        samples = _load_shape_code_samples(args, source_mode)
        sample_count = len(samples)
        start = 0
        stop = sample_count
        if args.mapping_idx_seg is not None and args.mapping_idx_seg >= 0:
            start = args.mapping_idx_seg * 500
            stop = min(start + 500, sample_count)
        if start >= stop:
            raise ValueError(
                f'No samples selected by mapping_idx_seg={args.mapping_idx_seg}'
            )

        n_repeats = (
            self.args.mapping_n_iter_rand
            if self.args.mapping_n_iter_rand is not None
            else 1
        )
        failures = []
        processed = []
        output_root = Path(dir_exp).parent
        for repeat_index in tqdm(range(n_repeats)):
            sample_dir = str(output_root / f'sample{repeat_index}')
            print(f'Exp dir: {sample_dir}')
            for code_index in tqdm(range(start, stop)):
                model_name = f'0000_uncond_gen_sample{code_index:04d}_complete'
                try:
                    code = torch.from_numpy(
                        np.asarray(samples[code_index])
                    )[None, None, ...].to(
                        next(self.parameters()).device,
                        dtype=torch.float32,
                    )
                    with _temporary_mesh_inference_state(self):
                        self.mapping_direction_infer = 1
                        self.args.mapping_direction_infer = 1
                        self.args.struct_add_partcode_noise = None
                        self.args.mapping_use_patch_condition_last_step = False
                        self._decode_one_shape_code(
                            cat_curr=args.category,
                            model_name=model_name,
                            aug_name='aug_0000',
                            view_name='view_0000_test0',
                            mode_geo='complete',
                            dir_exp_test=sample_dir,
                            inp_mode_normal='gt_normal',
                            dir_data_root=sample_dir,
                            lis_code_infer=code,
                        )
                    processed.append((repeat_index, code_index))
                except Exception as error:
                    message = (
                        f'sample {code_index}: {type(error).__name__}: '
                        f'{error}'
                    )
                    print(f'Error in: {message}')
                    failures.append(message)

        if failures:
            raise RuntimeError(
                f'Failed to decode {len(failures)} of '
                f'{len(processed) + len(failures)} shape-code runs:\n'
                + '\n'.join(failures)
            )
        return {'processed': processed, 'failed': failures}

    def _decode_one_shape_code(
        self,
        *,
        cat_curr,
        model_name,
        aug_name,
        mode_geo,
        view_name,
        dir_exp_test,
        inp_mode_normal,
        dir_data_root,
        lis_code_infer,
    ):



        if dir_exp_test is None:
            raise ValueError('dir_exp_test is required')
        if self.mapping_direction_infer != 1:
            raise ValueError("Shape-code decoding requires direction 1")
        if lis_code_infer is None:
            raise ValueError("lis_code_infer is required")

        dict_conf = {
            'mode_geo': mode_geo,
            'inp_mode_normal': inp_mode_normal,
        }
        path_pair = f"{dir_data_root}/{PathConfig.Folder_Name_Hierarchy_Sparse_VAE}/{cat_curr}/{model_name}/{aug_name}/{view_name}/hierarchy_pair_0000.h5"


        noise_disturb = ModelIOUtils.perturb_code(lis_code_infer, 0.0)
        lis_code_infer = lis_code_infer + noise_disturb

        patch_output, lis_mesh = self.infer_partcode_2_patchcode(
            lis_code_infer,
            mode_decode_sdf='dense',
        )
        path_save_part = str(
            Path(dir_exp_test) / 'generated_parts' / f'{model_name}.ply'
        )
        IOUtils.ensure_pdir_existence(path_save_part)
        GeoUtils.write_triangle_mesh(path_save_part, lis_mesh[0])

        dict_code_sparse_vae = self.infer_sparse_vae_code_single_mesh(
            path_save_part,
            dict_conf=dict_conf,
        )
        box_curr = GeoUtils.get_aabb(
            GeoUtils.read_triangle_mesh(path_save_part),
            return_6Dbox=True,
        )
        box_lwh = GeoUtils.get_lwh_from_aabb(box_curr[None, ...])

        patch_bb_child = ModelIOUtils.convert_code_coords2bbox(
            patch_output['code_coords'].cpu().numpy()[0]
        )[None, ...].to(lis_code_infer.device)
        patch_code_child = patch_output['code']
        patch_code_idx_child = torch.zeros(
            (1, patch_code_child.shape[1], 1),
            dtype=torch.long,
            device=patch_code_child.device,
        )
        dict_pair_part = {
            'code_child': lis_code_infer[None, ...],
            'code_idx_child': torch.zeros((1, 1, 1)).long(),
            'bb_child': torch.from_numpy(box_lwh).float()[None, ...],
            'idx_dir_token': torch.ones((1, 1)).long(),
            'patch_bb_child': patch_bb_child,
            'patch_code_child': patch_code_child.unsqueeze(-2),
            'patch_code_idx_child': patch_code_idx_child,
        }
        dict_info_shape_inp = {
            'mesh_merge': lis_mesh[0],
            'lis_bbox_centerlwh': [box_lwh[0]],
            'lis_img_part': [None],
            'lis_mesh': lis_mesh,
            'img_gt': None,
            'img_part_merge': None,
            'lis_code_noise': noise_disturb.cpu().numpy(),
        }

        dict_code_sparse_vae['path_pair'] = path_pair
        dict_pair = self.prepare_hierarchy_pair_mapping(dict_code_sparse_vae)
        dict_pair['path'] = [path_pair]
        dict_pair.update(dict_pair_part)
        self.dict_data_inp = self.to_cuda(dict_pair)
        self.validate_hierarchy(
            dir_exp_test,
            dict_data_inp_hier=self.dict_data_inp,
            should_validate_hierarchy=True,
            dict_info_shape_inp=dict_info_shape_inp,
            should_render_mesh=False,
        )
