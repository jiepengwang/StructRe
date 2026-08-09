"""Runtime-free registry for StructRe inference capabilities."""

import json
from dataclasses import dataclass


TASK_SURFACES = ('dataloader', 'task_mode')
TASK_READINESS = (
    'prepared-data-contract',
    'public-file-contract',
    'legacy-path-contract',
)


@dataclass(frozen=True)
class TaskSpec:
    """Static contract for one public inference capability."""

    name: str
    surface: str
    required_stages: tuple
    direction: str
    input: str
    output: str
    summary: str
    readiness: str
    aliases: tuple = ()
    required_options: tuple = ()

    def to_dict(self):
        return {
            'task': self.name,
            'surface': self.surface,
            'required_stages': list(self.required_stages),
            'direction': self.direction,
            'input': self.input,
            'output': self.output,
            'summary': self.summary,
            'readiness': self.readiness,
            'aliases': list(self.aliases),
            'required_options': [
                option for _attribute, option in self.required_options
            ],
        }


INFERENCE_TASKS = (
    TaskSpec(
        'extract_patch_codes',
        'dataloader',
        ('stage1',),
        'none',
        'Stage 1 prepared point-cloud data',
        'patch-code HDF5 artifacts and optional reconstructed meshes',
        'Encode part geometry for Stage 2 training or reconstruction.',
        'prepared-data-contract',
    ),
    TaskSpec(
        'extract_part_codes',
        'dataloader',
        ('stage2',),
        'none',
        'Stage 1 patch-code artifacts',
        '512-d part/shape-code HDF5 artifacts and optional reconstructions',
        'Encode patch sequences for Stage 3 training.',
        'prepared-data-contract',
    ),
    TaskSpec(
        'validate_hierarchy',
        'dataloader',
        ('stage3',),
        '0',
        'prepared hierarchy-pair data',
        'forward metrics and optional hierarchy/tree artifacts',
        'Evaluate rewriting and optionally decode direction-0 hierarchies.',
        'prepared-data-contract',
    ),
    TaskSpec(
        'generation_noise_sweep',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '1',
        'implicit PartNet filelist; noise levels selected by '
        '--mapping_noise_levels',
        'evaluation samples under the experiment directory; no result manifest',
        'Generate evaluation samples across explicit code-noise levels.',
        'legacy-path-contract',
        ('generation',),
        required_options=(('category', '--category'),),
    ),
    TaskSpec(
        'completion',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '0',
        'JSONL manifest with id, category, partial_pointcloud, and optional gt_mesh',
        'completed hierarchy artifacts under the experiment directory',
        'Complete structured shapes from partial observations.',
        'public-file-contract',
        required_options=(
            ('mapping_input_manifest', '--mapping_input_manifest'),
        ),
    ),
    TaskSpec(
        'reconstruction',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '0',
        'JSONL manifest with id, category, and mesh',
        'structured reconstructions under the experiment directory',
        'Reconstruct a batch of structured shapes.',
        'public-file-contract',
        required_options=(
            ('mapping_input_manifest', '--mapping_input_manifest'),
        ),
    ),
    TaskSpec(
        'reconstruction_shapenet_batch',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '0',
        'JSONL mesh manifest restricted to ShapeNet categories',
        'structured reconstructions under the experiment directory',
        'Apply the ShapeNet preset to structured reconstruction.',
        'public-file-contract',
        ('reconstruction_shapenet_all',),
        required_options=(
            ('mapping_input_manifest', '--mapping_input_manifest'),
        ),
    ),
    TaskSpec(
        'reconstruction_partnet_batch',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '0',
        'JSONL mesh manifest restricted to PartNet categories',
        'structured reconstructions under the experiment directory',
        'Apply the PartNet preset to structured reconstruction.',
        'public-file-contract',
        ('reconstructio_partnet_all', 'reconstruction_partnet_all'),
        required_options=(
            ('mapping_input_manifest', '--mapping_input_manifest'),
        ),
    ),
    TaskSpec(
        'mesh_cycle',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '0 -> 1',
        'PLY directory selected by --mapping_input_mesh_dir',
        'cycle hierarchy artifacts under the experiment directory',
        'Compose direction-0 decomposition and direction-1 rewriting.',
        'public-file-contract',
        ('test_mesh_cycle_infer',),
        required_options=(
            ('mapping_input_mesh_dir', '--mapping_input_mesh_dir'),
            ('category', '--category'),
            ('struct_add_partcode_noise', '--struct_add_partcode_noise'),
        ),
    ),
    TaskSpec(
        'mesh_reconstruct',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '0',
        'PLY directory selected by --mapping_input_mesh_dir',
        'level meshes and hierarchy tree under the experiment directory',
        'Reconstruct structured meshes from complete input meshes.',
        'public-file-contract',
        ('test_mesh',),
        required_options=(
            ('mapping_input_mesh_dir', '--mapping_input_mesh_dir'),
            ('category', '--category'),
        ),
    ),
    TaskSpec(
        'mesh_encode',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '1',
        'PLY directory',
        'shape-code HDF5',
        'Encode custom meshes into a reusable shape-code collection.',
        'public-file-contract',
        (),
        required_options=(
            ('mapping_input_mesh_dir', '--mapping_input_mesh_dir'),
            ('category', '--category'),
            ('mapping_shape_code_path', '--mapping_shape_code_path'),
        ),
    ),
    TaskSpec(
        'shape_code_generate',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '1',
        'shape-code HDF5 and sample range',
        'generated hierarchies and meshes',
        'Sample and decode structures from a shape-code distribution.',
        'public-file-contract',
        ('test_mesh_uncond_gen',),
        required_options=(
            ('mapping_shape_code_path', '--mapping_shape_code_path'),
            ('category', '--category'),
        ),
    ),
    TaskSpec(
        'shape_code_interpolate',
        'task_mode',
        ('stage1', 'stage2', 'stage3'),
        '1',
        'shape-code HDF5, source key, target key, and interpolation steps',
        'interpolated hierarchies and meshes under the experiment directory',
        'Interpolate two shape codes and decode each step.',
        'public-file-contract',
        ('test_mesh_interp',),
        required_options=(
            ('mapping_shape_code_path', '--mapping_shape_code_path'),
            ('category', '--category'),
            ('mapping_shape_code_source', '--mapping_shape_code_source'),
            ('mapping_shape_code_target', '--mapping_shape_code_target'),
        ),
    ),
)


_TASKS_BY_NAME = {task.name: task for task in INFERENCE_TASKS}
_ALIASES = {
    alias: task.name
    for task in INFERENCE_TASKS
    for alias in task.aliases
}
INFERENCE_TASK_NAMES = tuple(task.name for task in INFERENCE_TASKS)
INFERENCE_TASK_ALIASES = tuple(_ALIASES)
INFERENCE_TASK_ACCEPTED_NAMES = INFERENCE_TASK_NAMES + INFERENCE_TASK_ALIASES


def resolve_inference_task_name(name):
    if name in _TASKS_BY_NAME:
        return name
    try:
        return _ALIASES[name]
    except KeyError as error:
        accepted = ', '.join(INFERENCE_TASK_NAMES)
        raise ValueError(
            f'Unknown inference task {name!r}. Canonical tasks: {accepted}'
        ) from error


def get_inference_task(name):
    return _TASKS_BY_NAME[resolve_inference_task_name(name)]


def format_inference_task(name):
    task = get_inference_task(name)
    aliases = ', '.join(task.aliases) if task.aliases else 'none'
    required_options = ', '.join(
        option for _attribute, option in task.required_options
    ) or 'none'
    stages = ', '.join(task.required_stages)
    return '\n'.join((
        f'Inference task: {task.name}',
        f'Summary: {task.summary}',
        f'Surface: {task.surface}',
        f'Required stages: {stages}',
        f'Direction: {task.direction}',
        f'Input: {task.input}',
        f'Output: {task.output}',
        f'Readiness: {task.readiness}',
        f'Required options: {required_options}',
        f'Aliases: {aliases}',
    ))


def format_inference_task_inventory():
    return json.dumps(
        {
            'schema_version': 1,
            'tasks': [task.to_dict() for task in INFERENCE_TASKS],
        },
        indent=2,
    )
