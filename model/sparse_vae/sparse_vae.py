
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

import MinkowskiEngine as ME

from model.sparse_vae.modules import ResidualSparseEncoder3D, ResidualSparseDecoder3D
from model.sparse_vae.sdf_decoder import SDFDecoder

from model.sparse_vae.loss import VAELoss

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import utils.utils_training as TrainingUtils
import utils.utils_image as ImageUtils
import utils.utils_render as RenderUtils

from termcolor import cprint
from einops import repeat
from pathlib import Path
from multiprocessing import Pool, pool

from confs.path import (
    BBox_AABB,
    TSDF_Resolution_Part,
)
import confs.path as PathConfig

def save_one_file(tuple_data):
    path, dict_data = tuple_data
    try:
        IOUtils.ensure_pdir_existence(path)
        IOUtils.write_dict_to_h5(path, dict_data)
    except Exception as error:
        raise RuntimeError(f'Failed to save code: {path}') from error


def save_code_chunk(lis_chunk):
    n_thread = 8
    with pool.ThreadPool(n_thread) as p_pool:
        p_pool.map(save_one_file, lis_chunk)    
    
def _validation_output_paths(sample_path, mode_shape, exp_dir, phase):
    path_parts = Path(sample_path).parts
    if mode_shape in {'shape_parts', 'shape_aug_rotation'}:
        category, model_id, filename = path_parts[-3:]
        sample_name = Path(filename).stem
        mesh_relative = Path(category) / model_id / sample_name
        render_name = f'{model_id}_{sample_name}.png'
    elif mode_shape == 'shape_partial':
        category, model_id, part_name, filename = path_parts[-4:]
        view_name = Path(filename).stem
        mesh_relative = Path(category) / model_id / part_name / view_name
        render_name = f'{model_id}_{part_name}_{view_name}.png'
    else:
        raise ValueError(f'Unsupported Stage 1 shape mode: {mode_shape}')

    output_root = Path(exp_dir) / phase
    return (
        str(output_root / 'mesh' / mesh_relative),
        str(output_root / 'mesh_render' / render_name),
    )


def _require_feature_dim(feat_sample, expected_dim, context):
    actual_dim = feat_sample.shape[1] if feat_sample.ndim == 2 else None
    if actual_dim != expected_dim:
        raise ValueError(
            f"{context} expected feature dim {expected_dim}, "
            f"got shape {tuple(feat_sample.shape)}"
        )


class SparseVAE(nn.Module):
    def __init__(self, args, conf):
        nn.Module.__init__(self)
        
        self.args = args
        self.conf = conf

        
        self.scale_cube = self.conf['dataset.scale_cube']

            
        self.resolution = TSDF_Resolution_Part
        self.enc_ch = conf['model.channels_encoder']
        cprint(f'TSDF resolution: {TSDF_Resolution_Part}', 'yellow')
        self.n_blocks_downsample = self.conf['model.n_blocks_downsample']
        self.n_blocks_upsample = self.conf['model.n_blocks_upsample']
        
        self.d_out_decoder = self.enc_ch[self.n_blocks_downsample-self.n_blocks_upsample]

        
        self.scale_gt_sdf = self.conf['train.scale_gt_sdf']

        self.sdf_thres_trunc_decoder = self.conf['model.tsdf_thres_trunc_decoder']
        cprint(f'[SparseVAE Decoder] TSDF truncation threshold: {self.sdf_thres_trunc_decoder}', 'red')

        # 1. encoder
        self.encoder = ResidualSparseEncoder3D(conf)
        
        # 2. decoder
        self.decoder = ResidualSparseDecoder3D(conf)


        # 3. SDF
        self.max_sample_points_grid = self.conf['train.max_sample_points_grid']  
        
        self.interp = ME.MinkowskiInterpolation(return_kernel_map=True, return_weights=False)
        self.dim_feat = self.d_out_decoder
        self.sdf_decoder = SDFDecoder(
            dim_feat=self.dim_feat,
            hidden_size=256,
        )
        
        # 4. loss
        self.criterion = VAELoss()
        self.weight_sdf = self.conf['train.sdf_weight']
        self.weight_occu = self.conf['train.occu_weight']
        self.weight_klreg = self.conf['train.klreg_weight']
        
        # save code
        self.n_process = 4
        self.process_pool = None

        # render
        size_img, dist, elev, azim = 384, 3, 25, 35
        try:
            self.renderer = RenderUtils.init_mesh_renderer(image_size=size_img, dist=dist, elev=elev, azim=azim, device='cuda')
        except Exception as error:
            print(f'Fail to create renderer: {error}')
            self.renderer = None


    def get_coordinates_grid(self, resolution, should_return_tensor = True):
        '''
        return:
            grid: d*h*w*3
        '''
        X = np.arange(0, resolution).astype(np.int32)
        Y = np.arange(0, resolution).astype(np.int32)
        Z = np.arange(0, resolution).astype(np.int32)
        yy, xx, zz = np.meshgrid(Y, X, Z)
        pts_grid_idx = np.stack([xx, yy, zz], axis=-1)

        if should_return_tensor:
            pts_grid_idx = torch.from_numpy(pts_grid_idx)
        return pts_grid_idx        

    def get_batched_coordinates(self, data_dict, sdf_thres_trunc, dtype_coord=torch.int32, device='cuda'): 
        
        data_dict['sdf'] = data_dict['sdf'].cuda()
        mask_valid_sdf = torch.logical_and(data_dict['sdf']<sdf_thres_trunc, 
                                            data_dict['sdf']>-sdf_thres_trunc)

        # get valid sdf and corresponding coordinates
        in_feat = data_dict['sdf'][mask_valid_sdf][..., None]
        
        coordinates_grid = self.get_coordinates_grid(self.resolution).cuda()
        coordinates_grid = repeat(coordinates_grid, 'd h w c -> b 1 d h w c', b=self.cur_bs)
        coordinates_grid_valid = coordinates_grid[mask_valid_sdf]
            
        vec_n_valid_batch = mask_valid_sdf.reshape(self.cur_bs, -1).sum(dim=1)
        vec_n_valid_batch_cumsum = torch.cumsum(vec_n_valid_batch, dim=0)
        bcoords = torch.ones((vec_n_valid_batch_cumsum[-1], 1), dtype=dtype_coord).cuda()
        for idx_b in range(len(vec_n_valid_batch_cumsum)):
            ind_start = vec_n_valid_batch_cumsum[idx_b-1] if idx_b > 0 else 0
            ind_end = vec_n_valid_batch_cumsum[idx_b]
            
            bcoords[ind_start:ind_end] *= idx_b
        
        bcoords = torch.cat([bcoords, coordinates_grid_valid], dim=1)
        
        sparse_tensor_inp = ME.SparseTensor(
            features=in_feat,
            coordinates=bcoords,
            device=device
        )

        dict_data_out = {
            'in_feat': in_feat,
            'bcoords': bcoords,
            'mask_valid': mask_valid_sdf
        }

        return sparse_tensor_inp, dict_data_out

    def get_batched_coordinates_pointcloud(self, data_dict, dtype_coord=torch.int32, device='cuda'):
        # 1. get valid point cloud
        pointcloud = data_dict['pointcloud']
        mask_valid =  data_dict['mask_valid_pointcloud']
        pointcloud_valid = pointcloud[mask_valid]

        # 2. build batch idx
        vec_n_valid_batch = mask_valid.sum(dim=1)
        vec_n_valid_batch_cumsum = torch.cumsum(vec_n_valid_batch, dim=0)
        bcoords = torch.ones((vec_n_valid_batch_cumsum[-1], 1), dtype=dtype_coord)
        for idx_b in range(len(vec_n_valid_batch_cumsum)):
            ind_start = vec_n_valid_batch_cumsum[idx_b-1] if idx_b > 0 else 0
            ind_end = vec_n_valid_batch_cumsum[idx_b]
            
            bcoords[ind_start:ind_end] *= idx_b
        
        # 3. build sparse tensor
        bcoords = torch.cat([bcoords.cuda(), pointcloud_valid.cuda()], dim=1).cuda()
        normal_valid = data_dict['normal'][mask_valid]
        in_feat = normal_valid.cuda().float()
            
        sparse_tensor_inp = ME.SparseTensor(
            features=in_feat,
            coordinates=bcoords.int(),
            device=device
        )
        
        # 4. output: dict_data
        dict_data_out = {
            'in_feat': in_feat,
            'bcoords': bcoords,
            'mask_valid_pointcloud': mask_valid
        }
            
        return sparse_tensor_inp, dict_data_out
                
    def set_input(self, data_dict, dtype_coord=torch.int32, device='cuda'):
        self.cur_bs = len(data_dict['pointcloud'])
        if 'mask_valid_pointcloud' in data_dict:
            data_dict['mask_valid_pointcloud'] = (
                data_dict['mask_valid_pointcloud'] > 0
            )

        sin, dict_data_inp_encoder = self.get_batched_coordinates_pointcloud(
            data_dict, dtype_coord=dtype_coord, device=device
        )

        coordinate_manager = sin.coordinate_manager
        if 'sdf' in data_dict:
            _, dict_data_inp_decoder = self.get_batched_coordinates(
                data_dict, self.sdf_thres_trunc_decoder,
                dtype_coord=dtype_coord, device=device,
            )
            data_dict['bcoords'] = dict_data_inp_decoder['bcoords']
            data_dict['in_feat'] = dict_data_inp_decoder['in_feat']
            target_coordinates = dict_data_inp_decoder['bcoords']
        else:
            target_coordinates = dict_data_inp_encoder['bcoords'].int()

        target_key, _ = coordinate_manager.insert_and_map(
            coordinates=target_coordinates,
            string_id='target',
        )
        return data_dict, sin, target_key

    def forward(self, data_dict, device, iter_steps):
        # 0. data
        bs = len(data_dict['path'])
        data_dict, sin, target_key = self.set_input(data_dict, dtype_coord=torch.int32, device='cuda')
        
        # 1. encoder
        out_encoder, loss_klreg = self.encoder(sin)
        
        # The sparse encoder already applies KL regularization.
        F_quant = out_encoder.F
        self.feat_quant = F_quant.detach()

        n_sparse_code = len(F_quant) / bs
        if self.conf["train.should_extract_code"]:
            batch_coords, batch_feats_quant = (
                out_encoder.decomposed_coordinates_and_features
            )
            stride_xyz = out_encoder.coordinate_map_key.get_tensor_stride()
            return F_quant.new_zeros(1), {
                "code_quant": batch_feats_quant,
                "code_coords": batch_coords,
                "stride_xyz": stride_xyz,
            }
        
        # 3. decoder
        out_cls, targets, self.out_feat_sparse = self.decoder(out_encoder, target_key)
        
        if 'bcoords' not in data_dict:
            return 0, {}
        
        # 4. sdf
        bcoords = data_dict['bcoords'].float()
        coords_sample = bcoords[:, 1:]
        sdf_gt_sample = data_dict['in_feat']
        
        max_sample_points = self.max_sample_points_grid   # sample on cube space
        bs = len(data_dict['path'])
        
        if self.training and (len(bcoords) > max_sample_points * bs):
            choice = torch.randint(len(bcoords), (max_sample_points * bs, )).cuda()
            bcoords = torch.index_select(bcoords, 0, choice)
            sdf_gt_sample = torch.index_select(sdf_gt_sample, 0, choice)
            
        feat_sample, _ = self.interp(self.out_feat_sparse, bcoords)
        _require_feature_dim(feat_sample, self.dim_feat, "SparseVAE interpolation")
        
        n_feat = len(feat_sample)
        if self.training and (n_feat > max_sample_points * bs):
            choice = torch.randint(n_feat, (max_sample_points * bs, )).cuda()

            feat_sample = torch.index_select(feat_sample, 0, choice)
            coords_sample = torch.index_select(coords_sample, 0, choice)
            sdf_gt_sample = torch.index_select(sdf_gt_sample, 0, choice)
        
        self.sdf_pred = self.sdf_decoder(coords_sample, feat_sample)

        # 5. loss 
        sdf_gt_sample_scale = sdf_gt_sample*self.scale_gt_sdf
        loss_occu, loss_sdf = self.criterion(
            self.sdf_pred,
            sdf_gt_sample_scale,
            out_cls,
            targets,
        )
        loss_all = self.weight_occu*loss_occu + \
                    self.weight_sdf*loss_sdf + \
                    self.weight_klreg * loss_klreg
                    
        # 6. logs        
        phase = 'train' if self.training else 'test'
        _, stat_code = TrainingUtils.get_tensor_statistics(F_quant.detach(), tag=f'logs/{phase}_code')
        err_sdf = (sdf_gt_sample - self.sdf_pred.detach()/self.scale_gt_sdf).abs().mean().item()
        if not self.training:
            print(f'SDF Error: {err_sdf}')
        logs = {
            f'loss/loss_all_{phase}': loss_all.detach().item(),
            f'loss/loss_occu_{phase}': loss_occu.detach().item(),
            f'loss/loss_sdf_{phase}': loss_sdf.detach().item(),
            f'loss/loss_klreg_{phase}': loss_klreg.detach().item(),
            f'logs/n_sparse_code_{phase}': n_sparse_code,
            f'logs/err_sdf_{phase}': err_sdf,
            
        }
        logs.update(stat_code)
        
        return loss_all, logs
    
    def save_code(self, data_dict, data_out):
        category = self.args.category
        if not category:
            raise ValueError('Stage1 code extraction requires --category')
        if category in PathConfig.Categorys_PartNet:
            dir_data_root = PathConfig.DIR_ROOT_PARTNET
        elif category in PathConfig.Categorys_ShapeNet:
            dir_data_root = PathConfig.DIR_DATA_SHAPENET_OURS
        else:
            raise ValueError(f'Unsupported Stage1 category: {category}')

        dataset_name = self.args.sparse_vae_dataset_name
        mode_shape = self.conf['dataset.mode_shape']
        if dataset_name == 'partnet' and mode_shape == 'shape_parts':
            folder_code = PathConfig.Folder_Code_Sparse_VAE
        else:
            raise ValueError(
                'Unsupported Stage1 extraction contract: '
                f'dataset={dataset_name!r}, mode_shape={mode_shape!r}'
            )
        dir_out = f'{dir_data_root}/{folder_code}'

        code_quant = data_out['code_quant']
        code_coords = data_out['code_coords']
        stride_xyz = data_out['stride_xyz']
        vec_out_data = []

        for index, path in enumerate(data_dict['path']):
            if mode_shape == 'shape_parts':
                synset_id, model_id, file_name = path.split('/')[-3:]
                file_stem = Path(file_name).stem
                model_dir = f'{dir_out}/{synset_id}/{model_id}/{file_stem}'
            else:
                synset_id, model_id, aug_name, view_name = path.split('/')[-4:]
                model_dir = (
                    f'{dir_out}/{synset_id}/{model_id}/'
                    f'{aug_name}/{Path(view_name).stem}'
                )

            code_quant_i = code_quant[index].detach().cpu().numpy()
            coords_i = (
                code_coords[index].detach().cpu().numpy() // stride_xyz[0]
            )
            if len(code_quant_i) != len(coords_i):
                raise ValueError(
                    "Stage1 code/coordinate count mismatch for "
                    f"{path}: code={len(code_quant_i)}, coords={len(coords_i)}"
                )

            vec_out_data.append((
                f'{model_dir}/code.h5',
                {'code': code_quant_i, 'code_coords': coords_i},
            ))

        if not vec_out_data:
            return
        chunk_size = (len(vec_out_data) + self.n_process - 1) // self.n_process
        chunks = [
            vec_out_data[start:start + chunk_size]
            for start in range(0, len(vec_out_data), chunk_size)
        ]
        if self.process_pool is None:
            self.process_pool = Pool(self.n_process)
        self.process_pool.map(save_code_chunk, chunks)

    def validate_sdf(
        self, dict_data_inp, phase, exp_dir='./test', resolution_validate=256
    ):
        
        bs = len(dict_data_inp['path'])
        _, sin, target_key = self.set_input(dict_data_inp, dtype_coord=torch.int32, device='cuda')
        
        # 1. encoder
        out_encoder, _ = self.encoder(sin)
        
        # The encoder output is already the KL-regularized feature.
        F_quant = out_encoder.F
        self.feat_quant = F_quant.detach()
            
        # 3. decoder
        _, _, self.out_feat_sparse = self.decoder(out_encoder, target_key)
        
        if resolution_validate == 512:
            ratio_reso = 1.0
        elif resolution_validate == 256:
            ratio_reso = 1.0 * self.scale_cube
        elif resolution_validate == 128:
            ratio_reso = 2.0 * self.scale_cube
        else:
            ratio_reso = PathConfig.TSDF_Resolution_Minkowski_Grid / resolution_validate
        coordinates_grid = self.get_coordinates_grid(resolution_validate).cuda()*ratio_reso
        coordinates_grid = repeat(coordinates_grid, 'd h w c -> b d h w c', b=bs)

        batch_idx = torch.arange(bs).reshape(bs, 1, 1, 1, 1) * torch.ones((bs, resolution_validate, resolution_validate, resolution_validate, 1))
        bcoords_all = torch.cat([batch_idx.cuda(), coordinates_grid], dim=-1)
        bcoords_all = bcoords_all.reshape(-1, 4).cuda()

        # TODO: Split data to chunks
        feat_sample, _ = self.interp(self.out_feat_sparse, bcoords_all)
        _require_feature_dim(feat_sample, self.dim_feat, "SparseVAE interpolation")
        max_sample_points = 65536
        lis_chunks_coords = coordinates_grid.reshape(-1,3).split(max_sample_points)        
        lis_chunks_feat = feat_sample.split(max_sample_points)
        lis_sdf_pred = []
        for j_c in range(len(lis_chunks_coords)):
            sdf_pred_j = self.sdf_decoder(lis_chunks_coords[j_c], lis_chunks_feat[j_c]).detach().cpu()
            lis_sdf_pred.append(sdf_pred_j)
        sdf_pred = torch.cat(lis_sdf_pred, dim=0)
        sdf_pred = sdf_pred.reshape(bs, resolution_validate, resolution_validate, resolution_validate)

        mode_shape = self.conf['dataset.mode_shape']

        img_render_pred = RenderUtils.render_sdf(self.renderer, sdf_pred.unsqueeze(1).cuda())

        for i in range(bs):
            dir_save, path_save_render = _validation_output_paths(
                dict_data_inp['path'][i],
                mode_shape,
                exp_dir,
                phase,
            )
            IOUtils.ensure_dir_existence(dir_save)
            IOUtils.ensure_pdir_existence(path_save_render)

            path_sdf = f'{dir_save}/sdf_pred_feat.ply'
            sdf_inp = sdf_pred[i].detach().cpu().numpy() / self.scale_gt_sdf
            t1 = datetime.now()
            GeoUtils.extract_geometry_tsdf256(
                path_sdf,
                sdf_inp,
                threshold=0.005,
                bbox=BBox_AABB,
                resolution=resolution_validate,
                should_paint_mesh=False,
                vec_occu=None,
                reso_mink_grid=256,
            )
            print(f'MC time: {(datetime.now() - t1).total_seconds()}')
            torch.cuda.empty_cache()

            ImageUtils.write_image_lis(
                path_save_render,
                [img_render_pred[i]],
                interval_img=20,
                cat_mode='horizontal',
                color_space='RGB',
                add_black_border=True,
            )
    
            


    
    def decode_code_sequence(self, dict_data_inp, dict_out_part_vae, dtype_coord=torch.int32, device='cuda',
                                    should_extract_gt_feat = False, 
                                    tsdf_resolution_mc=128,
                                    mode_decode_sdf = 'sparse'):
        
        '''Decode code sequence from UniStruct
        
        '''
        # prepare sparse tensor from part_vae
        if not dict_out_part_vae and not should_extract_gt_feat:
            raise ValueError(
                "Stage1 code-sequence decode requires predicted or ground-truth code"
            )
        if len(dict_out_part_vae)>0:
            for key in dict_out_part_vae:
                dict_out_part_vae[key] = dict_out_part_vae[key].transpose(0,1) # t*b -> b*t
                if dict_data_inp is not None:
                    dict_data_inp[key] = dict_data_inp[key].transpose(0,1) # t*b -> b*t

            mask_nonempty = dict_out_part_vae['mask_nonempty']
                
            code = dict_out_part_vae['code']
            code_coords = dict_out_part_vae['code_coords']

            if code.shape[1] < mask_nonempty.shape[1]:
                print(f'Already remove empty code: {code.shape[1]} -> {mask_nonempty.shape[1]}')
                mask_nonempty = mask_nonempty[:,:code.shape[1]]
                sparse_code = code.squeeze(0)
                sparse_code_coords = code_coords.squeeze(0)
            else:
                sparse_code = code[mask_nonempty]
                sparse_code_coords = code_coords[mask_nonempty]
       
            if len(sparse_code) == 0:
                raise ValueError(
                    "Stage1 code-sequence decode received no non-empty patch codes"
                )
       
        if should_extract_gt_feat:
            mask_nonempty = dict_data_inp['mask_nonempty']
            if 'mask_valid' in dict_data_inp:
                mask_valid = (dict_data_inp['mask_valid']>0).transpose(0,1)
                sparse_code = dict_data_inp['code'][mask_valid]
                sparse_code_coords = dict_data_inp['code_coords'][mask_valid]
            else:
                sparse_code = dict_data_inp['code'][mask_nonempty]
                sparse_code_coords = dict_data_inp['code_coords'][mask_nonempty]
        
        bs = len(mask_nonempty)
        
        vec_n_valid_batch = mask_nonempty.sum(dim=1)
        vec_n_valid_batch_cumsum = torch.cumsum(vec_n_valid_batch, dim=0)
        bcoords = torch.ones((vec_n_valid_batch_cumsum[-1], 1), dtype=dtype_coord).cuda()
        for idx_b in range(len(vec_n_valid_batch_cumsum)):
            ind_start = vec_n_valid_batch_cumsum[idx_b-1] if idx_b > 0 else 0
            ind_end = vec_n_valid_batch_cumsum[idx_b]
            
            bcoords[ind_start:ind_end] *= idx_b
        
        if PathConfig.TSDF_Resolution==256:
            stride_tensor = 32
        elif PathConfig.TSDF_Resolution==128:
            stride_tensor = 16
        else:
            raise ValueError(
                "Stage1 sparse decoder supports TSDF resolution 128 or 256, "
                f"got {PathConfig.TSDF_Resolution}"
            )
            
        sparse_code_coords = sparse_code_coords*stride_tensor
        bcoords = torch.cat([bcoords, sparse_code_coords], dim=1).int()


        sparse_tensor_infer = ME.SparseTensor(
            features=sparse_code,
            coordinates=bcoords,
            device=device,
            tensor_stride=stride_tensor
        )
        # use decoder to decode

        # 3. decoder
        cm = sparse_tensor_infer.coordinate_manager
        target_key, _ = cm.insert_and_map(
            coordinates=bcoords,
            string_id="target",
        )
        out_cls, _, out_feat_sparse = self.decoder(sparse_tensor_infer, target_key)
        
        # 4. sdf
        # tsdf_resolution_mc
        if tsdf_resolution_mc == 256:
            ratio_reso = 1.0
        elif tsdf_resolution_mc == 128:
            ratio_reso = 2.0
        else:
            ratio_reso = PathConfig.TSDF_Resolution_Minkowski_Grid / tsdf_resolution_mc

        if mode_decode_sdf=='dense':
            coordinates_grid = self.get_coordinates_grid(tsdf_resolution_mc).cuda()*ratio_reso
            coordinates_grid = repeat(coordinates_grid, 'd h w c -> b d h w c', b=bs)
        
            batch_idx = torch.arange(bs).reshape(bs, 1, 1, 1, 1) * torch.ones((bs, tsdf_resolution_mc, tsdf_resolution_mc, tsdf_resolution_mc, 1))
            bcoords_all = torch.cat([batch_idx.cuda(), coordinates_grid], dim=-1) 
         
            bcoords_all = bcoords_all.reshape(-1, 4).cuda()    

            # TODO: Split data to chunks
            feat_sample, _ = self.interp(out_feat_sparse, bcoords_all)
            _require_feature_dim(feat_sample, self.dim_feat, "SparseVAE interpolation")
            max_sample_points = 65536  

            coordinates_grid_flat = coordinates_grid.reshape(-1,3)
            chunks_coordinates = coordinates_grid_flat.split(max_sample_points)
            chunks_feat_sample = feat_sample.split(max_sample_points)
            lis_sdf_pred = []
            for i_chunk in range(len(chunks_coordinates)):
                sdf_pred_i = self.sdf_decoder(chunks_coordinates[i_chunk], chunks_feat_sample[i_chunk])
                lis_sdf_pred.append(sdf_pred_i.detach().cpu().numpy())

            sdf_pred = np.concatenate(lis_sdf_pred, axis=0)

            sdf_pred = sdf_pred.reshape(bs, tsdf_resolution_mc, tsdf_resolution_mc, tsdf_resolution_mc)

            vec_occu = []
            batch_coords_cls, batch_feats_cls = out_cls[-1].decomposed_coordinates_and_features
            for i in range(bs):  
                # occupancy
                mask_occu_i = batch_feats_cls[i].squeeze(dim=-1)>0
                coords_occu = batch_coords_cls[i][mask_occu_i]
                vec_occu.append(coords_occu.cpu().numpy())
            
        elif mode_decode_sdf=='sparse':
            if tsdf_resolution_mc != 128:
                raise ValueError(
                    "Sparse Stage1 SDF decode requires resolution 128, "
                    f"got {tsdf_resolution_mc}"
                )
            idx_level_occ = -1
            batch_coords_cls, batch_feats_cls = out_cls[idx_level_occ].decomposed_coordinates_and_features

            lis_tsdf = []; vec_occu = []
            for i in range(bs):  
                # occupancy
                thres_prune = 0 if self.training else -0.5
                mask_occu_i = batch_feats_cls[i].squeeze(dim=-1)>thres_prune
                coords_occu_curr = batch_coords_cls[i][mask_occu_i].long()
                
                if len(coords_occu_curr)==0:
                    print(f'Empty occupancy at batch {i}')
                    lis_tsdf.append(None)
                    vec_occu.append(None)
                    continue
            
                
                coords_occu_curr = repeat(coords_occu_curr, 'n d -> n b d', b=8)
                kernel_cube_grid = torch.tensor(
                    [[[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                        [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
                    ]).cuda()
                coords_occu_curr = (coords_occu_curr + kernel_cube_grid).reshape(-1,3)
                
                
                batch_idx_curr = i*torch.ones((len(coords_occu_curr), 1)).cuda().long()
                bcoords_curr = torch.cat([batch_idx_curr, coords_occu_curr], dim=1)
                
                feat_sample_curr, _ = self.interp(out_feat_sparse, bcoords_curr.float())
                
                sdf_pred = self.sdf_decoder(coords_occu_curr.reshape(-1,3), feat_sample_curr).detach().cpu().numpy()
                sdf_pred = sdf_pred.squeeze(axis=1)
                
                # re-fill dense tsdf
                tsdf_grid_curr = np.ones((tsdf_resolution_mc, tsdf_resolution_mc, tsdf_resolution_mc))*0.11
                
                scale_factor = 256 // tsdf_resolution_mc
                coords_occu_valid = coords_occu_curr.cpu().numpy().astype(int)//scale_factor
                tsdf_grid_curr[coords_occu_valid[:,0], coords_occu_valid[:,1], coords_occu_valid[:,2]] = sdf_pred
                
                lis_tsdf.append(tsdf_grid_curr)
                vec_occu.append(coords_occu_curr.cpu().numpy())
                
 
            sdf_pred = lis_tsdf # np.stack(lis_tsdf, axis=0)   
            
        else:
            raise ValueError(
                f"Unsupported Stage1 SDF decode mode: {mode_decode_sdf!r}"
            )
            
                          

        dict_out_decoder = {
            'sdf_pred': sdf_pred,
            'vec_occu': vec_occu
        }
        return dict_out_decoder

    def infer_patch_code(self, data_dict):
        _, sin, _ = self.set_input(data_dict, dtype_coord=torch.int32, device="cuda")
        out_encoder, _ = self.encoder(sin)

        batch_coords, batch_feats = out_encoder.decomposed_coordinates_and_features
        stride_xyz = out_encoder.coordinate_map_key.get_tensor_stride()[0]
        return [
            {
                "code": feats.detach().cpu().numpy(),
                "code_coords": coords.detach().cpu().numpy() // stride_xyz,
                "code_idx": None,
            }
            for coords, feats in zip(batch_coords, batch_feats)
        ]
            
