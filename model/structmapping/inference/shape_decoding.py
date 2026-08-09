"""Stage 2/1 decoding from structural part codes to assembled meshes."""

from datetime import datetime

import numpy as np
import torch
from termcolor import cprint

import confs.path as PathConfig
from confs.path import BBox_AABB
import model.structmapping.utils_model.utils_model_io as ModelIOUtils
from utils.utils_tsdf import extract_geometry


class ShapeDecodingMixin:
    def decode_shape(self, dict_data_inp, data_out_tmp, dict_conf_exp, should_render_mesh=True,
                            dict_info_tree=None, i_level_hier=0, mapping_direction_infer=0):

        exp_dir = dict_conf_exp['exp_dir']

        dict_out_memory_part_vae = None
        if dict_conf_exp['should_update_struct_code']:
            if mapping_direction_infer==0 and i_level_hier>0:
                lis_bbox_lwh = self.get_valid_bbox(data_out_tmp, thres_valid_part=self.thres_prob_valid_part_fix)
                n_parent_pred_re = len(lis_bbox_lwh)

                lis_mesh_merge_parent, mask_valid = ModelIOUtils.merge_mesh_child2parent_by_link(dict_info_tree[f'level_{i_level_hier-1}']['lis_mesh'],
                                                                dict_p2c=data_out_tmp['dick_link_cp']['dict_p2c'],
                                                                n_parent_pred=n_parent_pred_re)
                lis_mesh_merge_norm, mask_valid_norm = ModelIOUtils.normalize_lis_mesh2cube(lis_mesh_merge_parent, lis_bbox_lwh,
                                                                                            dict_pointcloud=self.dict_pointcloud)
                n_valid_norm = mask_valid_norm.sum()
                if n_parent_pred_re>n_valid_norm:
                    print(f'Mesh number is different. {mask_valid.sum()} {mask_valid_norm.sum()}')
                mask_valid = mask_valid_norm

                sparse_vae_output = self.infer_sparse_vae_code_lis_mesh(lis_mesh_merge_norm)

                dict_out_memory_part_vae_valid = self.part_vae.infer_part_vae_memory_mesh(sparse_vae_output)


                # randomly mask memory
                ratio_mask = self.args.mapping_ratio_mask_memory
                cprint(f'Randomly mask memory. Ratio: {ratio_mask}', 'yellow')
                dict_out_memory_part_vae_valid['memory_key_padding_mask'] = ModelIOUtils.random_mask_patch_memory(dict_out_memory_part_vae_valid['memory_key_padding_mask'],
                                                                                    ratio_mask=ratio_mask)

                n_parent_pred_pointer = len(lis_mesh_merge_norm)
                if n_parent_pred_pointer != n_parent_pred_re:
                    print('NOT equal to pointer number')
                    dict_out_memory_part_vae = ModelIOUtils.fix_memory_part_vae(dict_out_memory_part_vae_valid, mask_valid)
                else:
                    dict_out_memory_part_vae = dict_out_memory_part_vae_valid

        dict_out_shape = self.decode_output_mapping(exp_dir, data_out_tmp, dict_data_inp,
                                                    dict_info_memory_part_vae=dict_out_memory_part_vae,
                                                    should_render_mesh=should_render_mesh)

        return dict_out_shape


    def decode_output_mapping(self, exp_dir, dict_out, dict_data_inp,
                                is_decode_gt_code = False,
                                dict_info_memory_part_vae=None,
                                lis_trans_parts_world2reg=None,
                                should_render_mesh=True,
                                should_decode_part_code_only=False):
        '''
        paras:
            out_class_prob: t*b*n, or t*b
            bboxes: t*b*6
            mask_key_padding_mask: t*b
        '''

        out_part_codes = dict_out['out_class'].detach()
        bboxes = dict_out['out_bbox'].detach()
        out_status = dict_out['out_status'].detach()

        B = bboxes.shape[1]
        if B != 1:
            raise ValueError(
                'Structured hierarchy decoding requires batch size 1, '
                f'got {B}'
            )

        thres_valid_part = self.thres_prob_valid_part_fix
        print(f'out_status shape: {out_status.shape}')
        out_prob_max = out_status.softmax(-1)[...,0]
        mask_valid_batch = out_prob_max > thres_valid_part
        n_valid = mask_valid_batch.sum()

        print(
            '[No random sampling]. Valid parts. '
            f'Threshold: {thres_valid_part}. Parts: {n_valid} / '
            f'{dict_data_inp["bb_parent"].shape[0] - 1 + 1e-6}'
        )

        sample_index = 0
        mask_valid = mask_valid_batch[:, sample_index]
        lis_part_bbox = bboxes[:, sample_index][mask_valid]
        lis_part_codes = out_part_codes[:, sample_index][mask_valid]
        query_idx = torch.arange(
            len(mask_valid), device=mask_valid.device
        )
        lis_color_idx_parts = (
            query_idx[mask_valid].detach().cpu().numpy()
            if not is_decode_gt_code
            else None
        )

        if lis_trans_parts_world2reg:
            # Update bbox by registration.
            for part_index, trans_i in enumerate(
                lis_trans_parts_world2reg
            ):
                scale_i = trans_i[0, 0]
                translate_i = torch.from_numpy(trans_i[:3, 3]).to(
                    device=lis_part_bbox.device,
                    dtype=lis_part_bbox.dtype,
                )
                lis_part_bbox[part_index][:3] += translate_i
                lis_part_bbox[part_index][3:6] *= scale_i

        if is_decode_gt_code:
            cprint('Decode GT data...', 'yellow')
            lis_part_bbox = dict_data_inp['bb_parent'][:, sample_index][:-1]
            lis_part_codes = dict_data_inp['code_parent'][:, sample_index][:-1]
        else:
            cprint('Decode inference data...', 'yellow')

        dict_info_shape = ModelIOUtils.get_model_info_from_path(
            dict_data_inp['path'][sample_index], exp_dir
        )
        lis_part_bbox = lis_part_bbox.detach().cpu().numpy()
        dict_mesh_parts = self.decode_lis_partcode2mesh(
            lis_part_codes,
            lis_part_bbox,
            dict_info_memory_part_vae,
            should_decode_part_code_only=should_decode_part_code_only,
            should_render_mesh=should_render_mesh,
            lis_color_idx_parts=lis_color_idx_parts,
            path_gt_mesh=dict_info_shape['path_gt_mesh'],
        )
        dict_mesh_parts['lis_color_idx_parts'] = lis_color_idx_parts
        dict_mesh_parts['lis_part_codes'] = lis_part_codes.detach()
        return dict_mesh_parts


    def decode_lis_partcode2mesh(self, lis_partcode, lis_bbox_lwh,
                                    dict_info_memory_part_vae,
                                    should_decode_part_code_only=False,
                                    should_render_mesh=True,
                                    lis_color_idx_parts=None,
                                    path_gt_mesh=None):
        '''
        paras:
            lis_partcode: T*1*512, tensor, cuda
            lis_bbox_lwh: T*8, numpy
        '''

        # 1. inference
        t0 = datetime.now()
        dict_out_chunk = self.decode_partcode2tsdf_chunk(lis_partcode, dict_info_memory_part_vae,
                                            should_decode_part_code_only)

        t1 = datetime.now()
        t_infer = (t1-t0).total_seconds()

        if should_decode_part_code_only:
            return dict_out_chunk



        # 2. mc to extract geometry
        lis_recon_tf, lis_vec_occ = dict_out_chunk['lis_recon_tf'], dict_out_chunk['lis_vec_occ']
        mask_valid = np.ones(len(lis_recon_tf), dtype=bool)
        for i in range(len(lis_recon_tf)):
            if lis_recon_tf[i] is None:
                mask_valid[i] = False

        lis_mesh = ModelIOUtils.extract_geometry_lis2(lis_recon_tf, threshold=0.05,
                                                        resolution=128,
                                                        lis_vec_occu=lis_vec_occ,
                                                        mode_mc='cpu')
        t2 = datetime.now()
        t_mc = (t2-t1).total_seconds()

        # 3. paint parts
        if lis_color_idx_parts is not None:
            lis_mesh = ModelIOUtils.paint_lis_mesh(
                lis_mesh, lis_color_idx_parts, PathConfig.get_color_map()
            )
        t3 = datetime.now()
        t_paint = (t3-t2).total_seconds()

        # 4. merge parts
        lis_bbox = lis_bbox_lwh
        dict_mesh_parts = ModelIOUtils.merge_lis_parts(lis_mesh, lis_bbox,
                                                            get_max_cluster_part=False)

        t4 = datetime.now()
        t_merge = (t4-t3).total_seconds()



        # 5. render parts and gt mesh
        if should_render_mesh:
            dict_img_render = ModelIOUtils.render_lis_parts(self.renderer, dict_mesh_parts, path_gt_mesh=path_gt_mesh)
            dict_mesh_parts.update(dict_img_render)
        t5 = datetime.now()
        t_render = (t5-t4).total_seconds()

        print(f'\n[Decode part] Infer: {t_infer:.04f}, MC: {t_mc:.04f}, Paint: {t_paint:.04f}, Merge: {t_merge:.04f}, Render: {t_render:.04f}')
        torch.cuda.empty_cache()

        dict_mesh_parts['mask_valid_parts'] = mask_valid

        return dict_mesh_parts



    def decode_partcode2tsdf_chunk(
        self,
        part_code,
        dict_info_memory_part_vae,
        should_decode_part_code_only=False,
    ):
        """Decode non-empty ``[parts, 1, dim]`` codes in bounded chunks."""
        if part_code.ndim != 3 or part_code.shape[1] != 1:
            raise ValueError(
                'Part-code decode expects shape [parts, 1, dim], '
                f'got {tuple(part_code.shape)}'
            )
        n_parts = part_code.shape[0]
        if n_parts == 0:
            raise ValueError('Part-code decode requires at least one part')

        chunk_size = self.decode_chunk_size
        print(
            f'[Chunk {chunk_size}] valid_struct_codes '
            f'(max, min, mean, mean_abs): '
            f'{part_code.max().item():.04f}, '
            f'{part_code.min().item():.04f}, '
            f'{part_code.mean().item():.04f}, '
            f'{part_code.abs().mean().item():.04f}'
        )
        code_chunks = part_code.split(chunk_size)

        if dict_info_memory_part_vae is not None:
            required_keys = {'memory_inp', 'memory_key_padding_mask'}
            missing_keys = required_keys - dict_info_memory_part_vae.keys()
            if missing_keys:
                raise ValueError(
                    'PartVAE memory is missing: '
                    + ', '.join(sorted(missing_keys))
                )
            memory_inp = dict_info_memory_part_vae['memory_inp']
            memory_mask = dict_info_memory_part_vae[
                'memory_key_padding_mask'
            ]
            if memory_inp.ndim != 3 or memory_inp.shape[1] != n_parts:
                raise ValueError(
                    'PartVAE memory_inp must have shape [T, parts, D] '
                    f'for {n_parts} parts, got {tuple(memory_inp.shape)}'
                )
            expected_mask_shape = (
                n_parts,
                memory_inp.shape[0] + part_code.shape[1],
            )
            if (
                memory_mask.ndim != 2
                or tuple(memory_mask.shape) != expected_mask_shape
            ):
                raise ValueError(
                    'PartVAE memory_key_padding_mask must have shape '
                    f'[parts, T+1]; expected {expected_mask_shape}, got '
                    f'{tuple(memory_mask.shape)}'
                )
            memory_chunks = memory_inp.transpose(0, 1).split(chunk_size)
            memory_mask_chunks = memory_mask.split(chunk_size)
        else:
            memory_chunks = (None,) * len(code_chunks)
            memory_mask_chunks = (None,) * len(code_chunks)

        reconstructions = []
        occupancies = []
        patch_chunks = {}
        for code_chunk, memory_chunk, memory_mask_chunk in zip(
            code_chunks, memory_chunks, memory_mask_chunks
        ):
            memory_current = (
                None
                if memory_chunk is None
                else memory_chunk.transpose(0, 1)
            )
            decoder_output = self.part_vae.inference_struct_code(
                code_chunk.transpose(0, 1),
                memory_inp=memory_current,
                memory_key_padding_mask=memory_mask_chunk,
                should_decode_part_code_only=should_decode_part_code_only,
            )
            if should_decode_part_code_only:
                for key, value in decoder_output.items():
                    patch_chunks.setdefault(key, []).append(value)
            else:
                reconstructions.extend(decoder_output['sdf_pred'])
                occupancies.extend(decoder_output['vec_occu'])

        if should_decode_part_code_only:
            return {
                key: torch.cat(values, dim=1)
                for key, values in patch_chunks.items()
            }
        return {
            'lis_recon_tf': reconstructions,
            'lis_vec_occ': occupancies,
        }



    def infer_partcode_2_patchcode(self, lis_code_infer, dict_out_memory_part_vae=None, mode_decode_sdf='sparse',
                                   use_memory_inp = False):
        '''
        Args:
            lis_code_infer: T*1*C
        '''

        memory_inp_curr = None; memory_key_padding_mask_curr = None
        if use_memory_inp and dict_out_memory_part_vae is not None:
            memory_inp_curr = dict_out_memory_part_vae['memory_inp']
            memory_key_padding_mask_curr = dict_out_memory_part_vae['memory_key_padding_mask']

        patch_output, geometry_output = self.part_vae.inference_struct_code(
            lis_code_infer.transpose(0, 1),
            memory_inp=memory_inp_curr,
            memory_key_padding_mask=memory_key_padding_mask_curr,
            mode_decode_sdf=mode_decode_sdf,
            return_patch_code=True,
        )

        recon_tf = geometry_output['sdf_pred']
        n_mesh = len(recon_tf)
        tsdf_resolution_mc = 128
        thres_mc = 0.005
        lis_mesh = []
        for j_mesh in range(n_mesh):
            tsdf_i = recon_tf[j_mesh]
            mesh_i = extract_geometry(None,  tsdf_i, threshold = thres_mc, bbox=BBox_AABB, resolution=tsdf_resolution_mc,
                                        should_paint_mesh=False)
            lis_mesh.append(mesh_i)


        return patch_output, lis_mesh
