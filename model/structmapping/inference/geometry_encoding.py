"""Geometry sampling and Stage 1/2 encoding capabilities for StructMapping."""

from copy import copy

import numpy as np
import torch
from einops import rearrange

import confs.path as PathConfig
from model.dataloader_config import collate_fn_padding
import model.structmapping.utils_model.utils_adapter_vae as VAEAdapterUtils
import model.structmapping.utils_model.utils_model_io as ModelIOUtils
from model.data_contracts import reverse_child_parent_info
import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils


class GeometryEncodingMixin:
    def prepare_sparse_vae_input_lis_mesh(self, lis_mesh):
        if not lis_mesh:
            raise ValueError("At least one mesh is required")
        lis_pointcloud, lis_normal, lis_mask, lis_path = [], [], [], []
        for mesh in lis_mesh:
            dict_data_i = self.prepare_data_sparse_vae_input(mesh=mesh)

            lis_pointcloud.append(dict_data_i['pointcloud'])
            lis_normal.append(dict_data_i['normal'])
            lis_mask.append(dict_data_i['mask_valid_pointcloud'])
            lis_path.append(dict_data_i['path'][0])
        dict_data_i['pointcloud'] = torch.cat(lis_pointcloud, dim=0)
        dict_data_i['mask_valid_pointcloud'] = torch.cat(lis_mask, dim=0)
        dict_data_i['normal'] = torch.cat(lis_normal, dim=0)
        dict_data_i['path'] = lis_path

        return dict_data_i


    def prepare_data_sparse_vae_input(self, path_mesh=None, mesh=None, n_sample = 65536, dict_conf=None):
        path_input_pointcloud = (
            dict_conf.get('path_input_pointcloud')
            if dict_conf is not None
            else None
        )
        if path_input_pointcloud and IOUtils.check_existence(path_input_pointcloud):
            print('Load preprocessed pointcloud and normal')
            cached = IOUtils.read_h5_to_dict(path_input_pointcloud)
            pointcloud = cached['pointcloud']
            normal = cached['normal']
            lis_cam_sample = cached.get('lis_cam_sample')
        else:
            pointcloud, normal, lis_cam_sample = (
                VAEAdapterUtils.sample_points_on_mesh(
                    path_mesh,
                    mesh,
                    n_sample,
                    dict_conf,
                )
            )

        inp_mode_normal = dict_conf['inp_mode_normal'] if dict_conf is not None else 'gt_normal'
        pointcloud, normal = VAEAdapterUtils.fix_input_normal(pointcloud, normal, inp_mode_normal, lis_cam_sample)

        pointcloud, _, mask_valid_pointcloud = VAEAdapterUtils.quantize_pointcloud2grid_mink(pointcloud)

        dict_data_inp_sparse_vae = {
            'pointcloud': torch.from_numpy(pointcloud)[None, ...],
            'normal': torch.from_numpy(normal)[None, ...],
            'mask_valid_pointcloud': torch.from_numpy(mask_valid_pointcloud)[None, ...],
            'path': [path_mesh]
        }

        return dict_data_inp_sparse_vae


    def sort_sparse_vae_code(self, dict_code_sparse_vae):
        code_coords2 = torch.from_numpy(dict_code_sparse_vae['code_coords'])[None, ...]

        n_reso_bottleneck = 8
        coeff_idx = torch.tensor([1, n_reso_bottleneck, n_reso_bottleneck**2])
        code_coords_1d = code_coords2*coeff_idx[None, None, ...]
        code_coords_1d = code_coords_1d.sum(dim=2).squeeze(dim=0)

        mask_nonempty= np.zeros(n_reso_bottleneck**3)
        mask_nonempty[code_coords_1d] = 1
        dict_code_sparse_vae['mask_nonempty'] = mask_nonempty

        # sort code and coordinates
        _, indice_sorted = torch.sort(code_coords_1d)
        indice_sorted = indice_sorted.numpy()

        dict_code_sparse_vae['code_coords'] = dict_code_sparse_vae['code_coords'][indice_sorted]
        dict_code_sparse_vae['code'] = dict_code_sparse_vae['code'][indice_sorted]

        n_code = len(code_coords_1d)
        mask_valid = np.ones(n_code)
        dict_code_sparse_vae['mask_valid'] = mask_valid

        return dict_code_sparse_vae


    def _infer_sorted_patch_codes(self, dict_sparse_vae_inp):
        return [
            self.sort_sparse_vae_code(dict_code)
            for dict_code in self.part_vae.sparse_vae.infer_patch_code(dict_sparse_vae_inp)
        ]


    def infer_sparse_vae_code_single_mesh(self, path_mesh, dict_conf=None):
        '''
        Args:

        Returns:
            dict_code_sparse_vae:
                code: T*D, narray
                code_coords: T*3, narray
                mask_nonempty: 512, narray
        '''
        dict_sparse_vae_inp = self.prepare_data_sparse_vae_input(path_mesh, dict_conf=dict_conf)
        lis_out_patch_code = self._infer_sorted_patch_codes(dict_sparse_vae_inp)

        return lis_out_patch_code[0]


    def convert_patchcode_to_pairinput(self, dict_code_sparse_vae):
        # numpy format
        patch_bb_child = ModelIOUtils.convert_code_coords2bbox(dict_code_sparse_vae['code_coords'], return_tensor=False)
        # numpy format
        dict_pair_patch  ={
            'patch_bb_child': patch_bb_child[None, ...],
            'patch_code_child': dict_code_sparse_vae['code'][None, ...],
            'patch_code_idx_child': np.zeros((patch_bb_child.shape[0], 1))[None, ...]
        }

        for key in dict_pair_patch:
            dict_pair_patch[key] = (
                torch.from_numpy(dict_pair_patch[key])
                .to(next(self.parameters()).device)
                .transpose(0, 1)
            )

        dict_pair_patch['patch_code_child'] = dict_pair_patch['patch_code_child'].unsqueeze(-2)
        return dict_pair_patch


    def prepare_pair_patch_from_mesh(self, path_mesh, dict_conf=None):
        dict_code_sparse_vae = self.infer_sparse_vae_code_single_mesh(path_mesh, dict_conf=dict_conf)
        dict_pair_patch = self.convert_patchcode_to_pairinput(dict_code_sparse_vae)
        return dict_pair_patch


    def infer_sparse_vae_code_lis_mesh(self, lis_mesh):
        '''
        Args:
            lis_mesh: list of open3d mesh
        Return:
            dict_out:
                code: B*T*D, tensor
                code_coords: B*T*3, tensor
                mask_nonempty: B*512, tensor
                path: B, list
        '''
        dict_inp_sparse_vae = self.prepare_sparse_vae_input_lis_mesh(lis_mesh)
        lis_out_patch_code = self._infer_sorted_patch_codes(dict_inp_sparse_vae)

        lis_dict = []

        for i in range(len(lis_mesh)):
            dict_code_i = lis_out_patch_code[i]

            # preprocess code
            dict_code_i_new = {}
            for key in dict_code_i:
                if key in ['code_idx', 'code_enc']:
                    continue
                dict_code_i_new[key] = torch.from_numpy(dict_code_i[key])[None, ...]


            # valid mask

            path = f'in_memory_mesh_{i:04d}'
            dict_code_i_new['path'] = path

            n_code = len(dict_code_i['code_coords'])
            mask_valid = torch.ones(1, n_code)
            dict_code_i_new['mask_valid'] = mask_valid

            lis_dict.append(dict_code_i_new)

        # collate function
        dict_out = collate_fn_padding(lis_dict)


        return dict_out


    def prepare_hierarchy_pair_mapping(self, dict_code_sparse_vae):
        # convert code to mapping format
        def convert_sparse_coords_to_bbox(code_coords_np):
            code_coords = code_coords_np
            grid_size = 2.3/(PathConfig.TSDF_Resolution-1)
            size_vox = (2.3+grid_size)/8
            half_cube = 1.15
            kernel_topright = np.array([[size_vox, size_vox, size_vox]])
            code_coords_norm = code_coords*size_vox
            code_coords_vert_bottomleft = code_coords_norm - half_cube # move 0~2.3 ->+-1.15
            code_coords_vert_topright = code_coords_vert_bottomleft + kernel_topright
            return np.concatenate(
                (code_coords_vert_bottomleft, code_coords_vert_topright),
                axis=1,
            )

        bbox_patch = convert_sparse_coords_to_bbox(dict_code_sparse_vae['code_coords'])


        # dataloader
        dict_pair = {}
        dict_pair['code_child'] = torch.from_numpy(dict_code_sparse_vae['code'])[None, ...]
        dict_pair['bb_child'] = torch.from_numpy(bbox_patch)[None, ...]
        dict_pair['code_idx_child'] = torch.zeros((len(bbox_patch)))[None, ...].long()

        dict_pair['is_patch'] = torch.tensor([[1]])
        dict_pair['rotation_angle'] = torch.tensor([[0.0]])
        dict_pair['idx_dir_token'] = torch.tensor([[0]])

        if dict_pair['code_child'].ndim == 3:
            dict_pair['code_child'] = rearrange(dict_pair['code_child'], 'b t c -> b t 1 c')

        if dict_pair['code_idx_child'].ndim == 2:
            dict_pair['code_idx_child'] = rearrange(dict_pair['code_idx_child'], 'b t -> b t 1')

        # align input
        dict_pair['patch_bb_child'] = copy(dict_pair['bb_child'])
        dict_pair['patch_code_child'] = copy(dict_pair['code_child']).float()
        dict_pair['patch_code_idx_child'] = copy(dict_pair['code_idx_child'])

        dict_pair['bb_child'] = torch.zeros((1,0,6))
        dict_pair['code_child'] = torch.zeros((1,0,1, 512)).float()
        dict_pair['code_idx_child'] = torch.zeros((1,0,1))

        dict_pair['bb_parent'] = torch.zeros((1,2,6))
        dict_pair['code_parent'] = torch.zeros((1,2,1, 512)).float()
        dict_pair['code_idx_parent'] = torch.zeros((1,2,1))

        angle_rot = 0
        bbox_lwh_child = GeoUtils.get_lwh_from_aabb(dict_pair['bb_child'].numpy(), angle_rot)
        dict_pair['bb_child'] = torch.from_numpy(bbox_lwh_child)
        patch_bb_child = GeoUtils.get_lwh_from_aabb(dict_pair['patch_bb_child'].numpy(), angle_rot)
        dict_pair['patch_bb_child'] = torch.from_numpy(patch_bb_child)

        angle_rot = torch.tensor([[0.0]])
        dict_pair["bb_rot"] = torch.cat(
            [torch.cos(angle_rot), torch.sin(angle_rot)], dim=-1
        )[None, ...]


        # reverse
        path = dict_code_sparse_vae['path_pair']
        _, view_name, hierarchy_name = path.split('/')[-3:]
        idx_dir = 0

        if self.mapping_direction_infer is not None:
            if self.mapping_direction_infer==1:
                if hierarchy_name!='hierarchy_pair_0000.h5':
                    idx_dir = 1
            elif self.mapping_direction_infer == 0:
                idx_dir = 0
            else:
                raise ValueError(
                    f"Unknown mapping direction: {self.mapping_direction_infer}"
                )
        else:
            if (view_name=='view_0000') and (hierarchy_name!='hierarchy_pair_0000.h5'):
                # except partial data, and patch->leaf
                prob_reverse = np.random.rand(1)[0]
                if prob_reverse>0.5:
                    dict_pair_reverse = reverse_child_parent_info(dict_pair)
                    dict_pair = dict_pair_reverse

                    idx_dir = 1
        dict_pair['idx_dir_token'] = torch.tensor([[idx_dir]])


        return dict_pair
