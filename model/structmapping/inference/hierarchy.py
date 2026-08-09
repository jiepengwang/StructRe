"""Iterative hierarchy decoding and structured-mesh assembly control."""

import os
from copy import copy
from datetime import datetime
from pathlib import Path

import torch
from einops import rearrange
from termcolor import cprint

from model.data_contracts import normalize_parent_eos, reverse_child_parent_info
import model.structmapping.utils_model.utils_model_io as ModelIOUtils
import model.structmapping.utils_model.utils_model_sampling as ModelSamplingUtils
import model.structmapping.utils_model.utils_model_tree as ModelTreeUtils
import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils


def _load_hierarchy_data(h5_file, mapping_direction_infer=None):
    # refer to load_hierarchy_data_sparse_vae
    keys_to_skip=['path_child', 'code_path_child', 'code_path_parent',
                    'node_idx_child', 'node_idx_parent',
                    'link_p2c_dict', 'rotation_angle',
                    'nodeID_link_p2c_dict',
                    'nodeID_link_c2p',
                    'code_enc_child']

    dict_pair = IOUtils.read_h5_to_dict(h5_file, keys_to_skip=keys_to_skip,
                                                use_tensor=True, keys_skip2tensor = ['link_p2c_dict'],
                                                should_expand_tensor=True)


    if 'is_patch' in dict_pair:
        # patch code
        is_patch = dict_pair['is_patch']
        if is_patch.numel() != 1 or is_patch.item() != 1:
            raise ValueError(
                "Patch hierarchy input requires scalar is_patch=1: "
                f"{h5_file}; got shape={tuple(is_patch.shape)}, "
                f"values={is_patch.reshape(-1).tolist()}"
            )

        dict_pair['is_patch'] = torch.tensor([[1]])

        if dict_pair['code_child'].ndim == 3:
            dict_pair['code_child'] = rearrange(dict_pair['code_child'], 'b t c -> b t 1 c')

        if  dict_pair['code_idx_child'].ndim == 2:
            dict_pair['code_idx_child'] = rearrange(dict_pair['code_idx_child'], 'b t -> b t 1')

        # align input
        dict_pair['patch_bb_child'] = copy(dict_pair['bb_child'])
        dict_pair['patch_code_child'] = copy(dict_pair['code_child'])
        dict_pair['patch_code_idx_child'] = copy(dict_pair['code_idx_child'])

        dict_pair['bb_child'] = torch.zeros((1,0,6))
        dict_pair['code_child'] = torch.zeros((1,0,1, 512))
        dict_pair['code_idx_child'] = torch.zeros((1,0,1))

    else:
        # part code
        dict_pair['is_patch'] = torch.tensor([[0]])

        if 'patch_code_child' not in dict_pair:
            raise ValueError(
                f"Part hierarchy input requires patch_code_child: {h5_file}"
            )
        if dict_pair['patch_code_child'].ndim == 3:
            dict_pair['patch_code_child'] = rearrange(dict_pair['patch_code_child'], 'b t c -> b t 1 c')

        if  dict_pair['patch_code_idx_child'].ndim == 2:
            dict_pair['patch_code_idx_child'] = rearrange(dict_pair['patch_code_idx_child'], 'b t -> b t 1')


    dict_pair['rotation_angle'] = torch.tensor([[0.0]])
    dict_pair['code_parent'] = dict_pair['code_parent'].float()
    dict_pair['path'] = h5_file

    angle_rot = dict_pair['rotation_angle']

    # child
    bbox_lwh_child = GeoUtils.get_lwh_from_aabb(dict_pair['bb_child'].numpy(), angle_rot)
    dict_pair['bb_child'] = torch.from_numpy(bbox_lwh_child)
    bbox_lwh_patch = GeoUtils.get_lwh_from_aabb(dict_pair['patch_bb_child'].numpy(), angle_rot)
    dict_pair['patch_bb_child'] = torch.from_numpy(bbox_lwh_patch)

    # parent
    bbox_lwh_parent = GeoUtils.get_lwh_from_aabb(dict_pair['bb_parent'].numpy(), angle_rot)
    dict_pair['bb_parent'] = torch.from_numpy(bbox_lwh_parent)

    # rotation
    dict_pair["bb_rot"] = torch.cat(
        [torch.cos(angle_rot), torch.sin(angle_rot)], dim=-1
    )[None, ...]


    dict_pair = normalize_parent_eos(dict_pair, sample_path=h5_file)

    # Explicit inference direction replaces the old augmentation-ID contract.
    _, _, hierarchy_name = h5_file.split("/")[-3:]
    if mapping_direction_infer not in (None, 0, 1):
        raise ValueError(
            f"Unknown mapping direction: {mapping_direction_infer}"
        )

    idx_dir = 0
    if (
        mapping_direction_infer == 1
        and hierarchy_name != "hierarchy_pair_0000.h5"
    ):
        dict_pair = reverse_child_parent_info(dict_pair)
        idx_dir = 1

    dict_pair["idx_dir_token"] = torch.tensor([[idx_dir]])

    return dict_pair



class HierarchyInferenceMixin:
    @staticmethod
    def select_hierarchy_sample(dict_data_inp, sample_index=0):
        paths = dict_data_inp.get('path')
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError(
                'Hierarchy batch selection requires a non-empty path sequence'
            )
        batch_size = len(paths)
        if not 0 <= sample_index < batch_size:
            raise IndexError(
                f'Hierarchy sample index {sample_index} is outside batch size '
                f'{batch_size}'
            )

        selected = {}
        for key, value in dict_data_inp.items():
            if torch.is_tensor(value):
                if value.ndim < 2 or value.shape[1] != batch_size:
                    raise ValueError(
                        f'Hierarchy tensor {key!r} must use sequence-first '
                        f'batch dimension {batch_size}, got {tuple(value.shape)}'
                    )
                selected[key] = value[:, sample_index:(sample_index + 1)]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                selected[key] = value[sample_index:(sample_index + 1)]
            else:
                selected[key] = value
        return selected

    def check_stop_condition_hier_infer(self, dict_data_inp, dict_out_dec, mapping_direction_infer, dict_conf_exp):
        should_stop = False
        n_child_new = dict_data_inp['bb_child'].shape[0]
        n_parent_new = self.get_number_valid_parts(dict_out_dec, thres_valid_part=dict_conf_exp['thres_valid_part'])
        if mapping_direction_infer not in (0, 1):
            raise ValueError(
                f"Unknown mapping direction: {mapping_direction_infer}"
            )

        if mapping_direction_infer==0:
            if n_child_new == 1:
                should_stop = True



        elif mapping_direction_infer==1:
            if n_child_new==n_parent_new:
                cprint(f'\n\n\n[Info] No new components (parent->child): {n_child_new} -> {n_parent_new}. Stop here', 'yellow')
                should_stop = True
            elif n_child_new > n_parent_new:
                cprint(f'\n\n\n[Warning]: {n_child_new} > {n_parent_new}. Stop here', 'yellow')
                should_stop = True

        return should_stop


    def set_dict_conf_exp(self, dict_conf_exp, args, thres_prob_valid_part_fix):
        if args.mode_thres_prob_valid_part == "fixed":
            thres_valid_part = thres_prob_valid_part_fix
        else:
            try:
                thres_valid_part = float(args.mode_thres_prob_valid_part)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Unknown mode_thres_prob_valid_part: "
                    f"{args.mode_thres_prob_valid_part}"
                ) from error
        cprint(f'thres_valid_part: {thres_valid_part}', 'yellow')

        dict_conf_exp['thres_valid_part'] = thres_valid_part
        dict_conf_exp['should_decode_shape_gt'] = False
        dict_conf_exp['should_update_struct_code'] = args.mapping_update_struct_code
        dict_conf_exp['use_patch_condition_last_step'] = args.mapping_use_patch_condition_last_step



        return dict_conf_exp


    def validate_hierarchy(
        self,
        exp_dir,
        dict_data_inp_hier=None,
        should_validate_hierarchy=False,
        dict_info_shape_inp=None,
        should_render_mesh=False,
    ):

        dict_conf_exp = self.set_dict_conf_exp(
            {},
            self.args,
            self.thres_prob_valid_part_fix,
        )
        dict_conf_exp['exp_dir'] = exp_dir
        mapping_direction_infer = self.mapping_direction_infer


        dict_info_tree = {}
        if mapping_direction_infer==1:
            lis_tree_nodes = ModelTreeUtils.construct_lis_tree_nodes(dict_info_shape_inp, curr_level_hier=-1)
            dict_info_shape_inp['lis_tree_nodes'] = lis_tree_nodes
            dict_info_tree[f'level_{-1}'] = dict_info_shape_inp

        # Hierarchical inference
        data_inp_tmp = dict_data_inp_hier; data_out_tmp = None; dict_out_shape = None
        dict_info_shape = ModelIOUtils.get_model_info_from_path(data_inp_tmp['path'][0], exp_dir)
        path_img_hier = f'{dict_info_shape["dir_mesh_render_merge"]}_hier.png'


        # should_load_inp
        self.dict_pointcloud = None


        n_max_hier_infer = 2 if mapping_direction_infer==0 else 5
        if self.args.mapping_n_hier_infer is not None:
            n_max_hier_infer = self.args.mapping_n_hier_infer

            if n_max_hier_infer==1:
                should_validate_hierarchy = False


        t_start_hier = datetime.now()
        for i in range(n_max_hier_infer):
            cprint(f'\nHierarchy inference level: {i} ...', 'yellow')
            t1 = datetime.now()

            # 1. prepare input
            dict_data_inp, is_succeed = self.prepare_pair_input_from_last_level(data_inp_tmp, data_out_tmp, dict_out_shape,
                                                                            curr_level_hier=i,
                                                                            dict_conf_exp=dict_conf_exp)
            if not is_succeed:
                break

            # 2. model inference
            _, data_out_tmp = self.inference_one_step(dict_data_inp, dict_conf=dict_conf_exp)

            t2 = datetime.now()
            t_infer = (t2-t1).total_seconds()

            # 3. check stop condition
            should_stop = self.check_stop_condition_hier_infer(dict_data_inp, data_out_tmp,
                                                                    mapping_direction_infer=self.mapping_direction_infer,
                                                                    dict_conf_exp=dict_conf_exp)


            if should_stop and should_validate_hierarchy:
                break

            # 5. Decode shape
            dict_out_shape_tree = self.decode_shape(dict_data_inp, data_out_tmp, dict_conf_exp,
                                                        should_render_mesh=should_render_mesh,
                                                        dict_info_tree=dict_info_tree,
                                                        i_level_hier=i,
                                                        mapping_direction_infer=self.mapping_direction_infer,)






            # 4. perturb part code
            if (not should_stop) and self.args.struct_add_partcode_noise:
                code_noise_perturb = ModelIOUtils.perturb_code(data_out_tmp['out_class'], self.args.struct_add_partcode_noise)
                data_out_tmp['out_class'] += code_noise_perturb


                mask_valid_parts = self.get_number_valid_parts(data_out_tmp, thres_valid_part=dict_conf_exp['thres_valid_part'],
                                                                return_mask=True)
                dict_info_shape['lis_code_noise'] = code_noise_perturb[mask_valid_parts].cpu().numpy()

                # 5. Decode shape
                dict_out_shape = self.decode_shape(dict_data_inp, data_out_tmp, dict_conf_exp,
                                                        should_render_mesh=should_render_mesh,
                                                        dict_info_tree=dict_info_tree,
                                                        i_level_hier=i,
                                                        mapping_direction_infer=self.mapping_direction_infer,)
            else:
                dict_out_shape = dict_out_shape_tree


            t3 = datetime.now()
            t_decode = (t3-t2).total_seconds()

            # 6. build tree
            dict_info_tree = ModelTreeUtils.update_tree_hierarchy(
                dict_info_tree,
                i,
                data_out_tmp,
                dict_out_shape_tree,
                self.mapping_direction_infer,
            )
            t4 = datetime.now()
            t_tree = (t4-t3).total_seconds()
            print(f'[Rewriting Hier-{i}] Inference time: {t_infer:.04f}, decode time: {t_decode:.04f}, tree time: {t_tree:.04f}. Total: {t_infer+t_decode+t_tree:.04f}')

            # 7. Stop here
            if should_stop:
                break

            data_inp_tmp = dict_data_inp



        # Direction 0 may stop before producing a single explicit root.
        if mapping_direction_infer==0:
            dict_info_tree = ModelTreeUtils.merge_root_tree(dict_info_tree)

        should_save_mesh = True
        if n_max_hier_infer == 1:
            mesh_merge = dict_info_tree["level_0"]["mesh_merge"]
            path_mesh_merge = dict_info_shape["dir_mesh_merge"] + "/mesh_merge.ply"
            GeoUtils.write_triangle_mesh(path_mesh_merge, mesh_merge)
            should_save_mesh = False



        t_end = datetime.now()
        print(f'[Rewriting] Total time: {(t_end-t_start_hier).total_seconds():.02f}')
        ModelTreeUtils.dump_tree(
            path_tree=f'{dict_info_shape["dir_mesh_merge"]}/tree_pred.pkl',
            dict_info_tree=dict_info_tree,
            mapping_direction_infer=self.mapping_direction_infer,
            should_dump_tree=True,
            path_img_hier=path_img_hier,
            should_save_images=should_render_mesh,
            dir_mesh_merge=dict_info_shape['dir_mesh_merge'] + '/../mesh_merge',
            should_save_mesh=should_save_mesh,
        )



        self.dict_data_inp = dict_data_inp
        return dict_data_inp, dict_out_shape



    def decode_hierarchy_pointer(self, dict_data_inp, dict_data_out, thres_valid_part=0.2, mapping_direction_infer=0):
        if mapping_direction_infer not in (0, 1):
            raise ValueError(
                f"Unknown mapping direction: {mapping_direction_infer}"
            )
        # input latent
        latent_tf_inp = dict_data_out['latent_tf_inp']
        T_seq_part = len(dict_data_inp['bb_child'])
        latent_tf_inp_valid = latent_tf_inp[:T_seq_part]

        # output latent
        latent_tf_out = dict_data_out['latent_tf_out']
        out_status = dict_data_out['out_status']
        out_prob_max = out_status.softmax(-1)[...,0]
        mask_valid = (out_prob_max > thres_valid_part).squeeze(1) # N*1 -> N
        latent_tf_out_valid = latent_tf_out[mask_valid]


        latent_tf_inp_emb = self.criterion.pointer.emb_latent(latent_tf_inp_valid)
        latent_tf_out_emb = self.criterion.pointer.emb_latent(latent_tf_out_valid)

        latent_tf_inp_emb = latent_tf_inp_emb.permute(1,0,2) # T*B*D -> B*T*D
        latent_tf_out_emb = latent_tf_out_emb.permute(1,2,0)  # N*B*D -> B*D*N

        prob_link = torch.matmul(latent_tf_inp_emb, latent_tf_out_emb) # B*T*N
        prob_link = torch.sigmoid(prob_link).transpose(0,1) # B*T*N -> T*B*N

        # choose argmax at each row
        bs = prob_link.shape[1]
        if mapping_direction_infer==0:
            idx_valid = torch.argmax(prob_link, dim=2)  # T*B
        elif mapping_direction_infer==1:
            idx_valid = torch.argmax(prob_link, dim=0)  # T*B

        for i in range(bs):
            # link c2p
            if mapping_direction_infer==0:
                idx_valid_curr = idx_valid[:, i]
                idx_seq = torch.arange(
                    len(idx_valid_curr), device=idx_valid_curr.device
                )
                link_sorted = torch.stack([idx_seq, idx_valid_curr], dim=-1).cpu().numpy() # N*2
            elif mapping_direction_infer==1:
                idx_valid_curr = idx_valid[i,:]
                idx_seq = torch.arange(
                    len(idx_valid_curr), device=idx_valid_curr.device
                )
                link_sorted = torch.stack([idx_valid_curr, idx_seq], dim=-1).cpu().numpy() # N*2

            dict_p2c = {}; dict_c2p = {}
            for idx_link in range(len(link_sorted)):
                link_i = link_sorted[idx_link]
                idx_c, idx_p = link_i
                if idx_p not in dict_p2c:
                    dict_p2c[idx_p] = []
                dict_p2c[idx_p].append(idx_c)

                if idx_c not in dict_c2p:
                    dict_c2p[idx_c] = []
                dict_c2p[idx_c].append(idx_p)

            dict_link_cp = {
                'link_inp2out': link_sorted,
                'dict_p2c': dict_p2c,
                'dict_c2p': dict_c2p
            }


        return dict_link_cp



    def iterative_decode_parts(
        self,
        dict_conf_valid,
        dict_data_inp,
        dict_out_enc,
        dict_out_dec,
    ):
        thres_valid_part = dict_conf_valid['thres_valid_part']
        memory = dict_out_enc['memory']
        memory_key_padding_mask = dict_out_enc['mask_padding']
        input_tf_decoder = dict_out_dec['input_tf_decoder']
        sampling_strategy = self.sampling_strategy_infer
        filter_strategy = self.filter_strategy_infer

        n_valid, lis_prob_dist = self.get_number_valid_parts(
            dict_out_dec,
            thres_valid_part,
            return_prob_dist=True,
        )
        n_gt = dict_data_inp['bb_parent'].shape[0] - 1
        cprint(
            f'Iteration [0] Parts: {n_valid}/{n_gt}. '
            f'Prob(0.1->0.5): {lis_prob_dist}',
            'yellow',
        )

        dict_out_iter = dict_out_dec
        mask_attach_prev = None
        for i_iter in range(self.n_iter):
            if sampling_strategy == 'gt_matcher':
                dict_matcher_v1 = self.matcher(
                    dict_out_iter,
                    dict_data_inp,
                    dict_attach=None,
                )
                dict_attach = ModelSamplingUtils.random_sampling_output(
                    dict_out_iter,
                    dict_matcher_v1,
                    step_iter=i_iter,
                    filter_strategy=filter_strategy,
                    lis_schedule=self.lis_schedule,
                )
            else:
                dict_attach, mask_attach_prev, _ = (
                    ModelSamplingUtils.random_sampling_output_infer(
                        dict_out_iter,
                        sampling_strategy=sampling_strategy,
                        step_iter=i_iter,
                        mask_prev=mask_attach_prev,
                        filter_strategy=filter_strategy,
                        lis_schedule=self.lis_schedule,
                    )
                )
            dict_out_iter = self.decoder(
                input_tf_decoder,
                memory,
                memory_key_padding_mask,
                dict_attach,
            )
            dict_out_iter['latent_tf_inp'] = dict_out_enc['latent_tf_inp']

            n_valid, lis_prob_dist = self.get_number_valid_parts(
                dict_out_iter,
                thres_valid_part,
                return_prob_dist=True,
            )
            n_gt = dict_data_inp['bb_parent'].shape[0] - 1
            cprint(
                f'Iteration [{i_iter + 1}] Parts: {n_valid}/{n_gt}. '
                f'Prob(0.1->0.5): {lis_prob_dist}',
                'yellow',
            )

        return dict_out_iter

    def get_number_valid_parts(self, data_output, thres_valid_part, return_mask=False, return_prob_dist=False):
        out_prob_max = data_output['out_status'].softmax(-1)[...,0]
        mask_valid = (out_prob_max > thres_valid_part).squeeze(1)
        n_valid = mask_valid.sum()
        if return_mask:
            return mask_valid

        if return_prob_dist:
            lis_n_part = []
            for thres in [0.1, 0.2, 0.3, 0.4, 0.5]:
                mask_valid = (out_prob_max > thres).squeeze(1)
                n_valid = mask_valid.sum().item()
                lis_n_part.append(n_valid)
            return n_valid, lis_n_part
        return n_valid


    def get_valid_bbox(self, data_output, thres_valid_part):
        out_prob_max = data_output['out_status'].softmax(-1)[...,0]
        mask_valid = (out_prob_max > thres_valid_part) #.squeeze(1)
        bbox_valid = data_output['out_bbox'][mask_valid].detach().cpu().numpy()
        return bbox_valid


    def prepare_pair_input_from_last_level(self, dict_data_inp, dict_out_dec, dict_out_shape,
                                                curr_level_hier,
                                                dict_conf_exp):

        thres_valid_part = dict_conf_exp['thres_valid_part']

        dict_data_inp_new = {}
        if curr_level_hier==0:
            dict_data_inp_new, is_succeed = dict_data_inp, True
        else:
            mask_valid_parts = None
            dict_data_inp_new, is_succeed = self.prepare_hierarchy_input_from_output(dict_data_inp, dict_out_dec,
                                                                                    use_random_sample=False,
                                                                                    mask_valid_sample=None,
                                                                                    thres_valid_part = thres_valid_part,
                                                                                    mask_valid_parts=mask_valid_parts)

            if dict_conf_exp['use_patch_condition_last_step']:
                cprint(f'Using patch condtion of last step', 'yellow')
                dict_conf_curr = {
                    'mesh_inp': dict_out_shape['mesh_merge'],
                    'mode_geo': 'complete',
                    'inp_mode_normal': 'gt_normal'
                }
                dict_pair_patch_new = self.prepare_pair_patch_from_mesh(path_mesh=None, dict_conf=dict_conf_curr)
                dict_data_inp_new.update(dict_pair_patch_new)

        return dict_data_inp_new, is_succeed


    def prepare_hierarchy_input_from_output(self, dict_data_inp, data_output, use_random_sample, mask_valid_sample=None,
                                                    thres_valid_part = 0.5,
                                                    should_update_input_by_prediction=False,
                                                    mask_valid_parts=None):

        '''
        paras:
            mask_valid_parts: T-element boolean tensor
        '''
        if mask_valid_sample is None:
            out_prob_max = data_output['out_status'].softmax(-1)[...,0]
            mask_valid = (out_prob_max > thres_valid_part).squeeze(1)
        else:
            mask_valid = mask_valid_sample.squeeze(-1)

        code_pred_valid = data_output['out_class'][mask_valid]
        code_pred_idx_valid = torch.zeros(
            code_pred_valid.shape[:-1],
            dtype=torch.int,
            device=code_pred_valid.device,
        )
        bbox_pred_valid = data_output['out_bbox'][mask_valid]

        if mask_valid_parts is not None:
            if mask_valid_parts.sum() != len(mask_valid_parts):
                print(f'[{len(mask_valid_parts)- mask_valid_parts.sum()}] parts are None and removed')
            code_pred_valid = code_pred_valid[mask_valid_parts]
            code_pred_idx_valid = code_pred_idx_valid[mask_valid_parts]
            bbox_pred_valid = bbox_pred_valid[mask_valid_parts]


        path_ = dict_data_inp['path'][0]
        dir_file, stem_file = str(Path(path_).parent), Path(path_).stem,

        stem_file_id_next = int(stem_file[-4:]) + 1
        file_parent = dir_file + f'/hierarchy_pair_{stem_file_id_next:04d}.h5'
        if (not os.path.exists(file_parent)) or (should_update_input_by_prediction):
            dict_inp_new = copy(dict_data_inp)
            dict_inp_new.update({
                'bb_child': bbox_pred_valid[...,:6].float(),
                'code_child': code_pred_valid,
                'code_idx_child': code_pred_idx_valid,
                'path': [file_parent],
            })


            return dict_inp_new, True
        # load default data
        dict_data_parent = _load_hierarchy_data(file_parent,
                                                            mapping_direction_infer=self.mapping_direction_infer)


        if use_random_sample:
            mask_max_ratio = 0.5
            mask_ratio = mask_max_ratio*torch.rand(1)

            n_code = len(bbox_pred_valid)
            n_mask = (mask_ratio*n_code).floor().item()
            idx_to_mask = torch.randint(0, n_code, [int(n_mask)])

            code_pred_idx_valid[idx_to_mask, :] = -4


        dict_inp_new = copy(dict_data_inp)
        dict_inp_new.update({
            'bb_child': bbox_pred_valid[...,:6].float(),
            'code_child': code_pred_valid,
            'code_idx_child': code_pred_idx_valid,

            'path': [file_parent]
        })

        # attach patch input
        for key in ['patch_bb_child', 'patch_code_child', 'patch_code_idx_child',
                        'bb_parent', 'code_parent', 'code_idx_parent', 'idx_dir_token', 'link_c2p']:
            if key in dict_data_parent:
                dict_inp_new[key] = dict_data_parent[key].transpose(0, 1).to(
                    bbox_pred_valid.device, dtype=torch.float32
                )

            if key in ['code_idx_parent', 'patch_code_idx_child']:
                dict_inp_new[key] = dict_inp_new[key].long()





        return dict_inp_new, True
