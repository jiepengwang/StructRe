import torch
import numpy as np

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import utils.utils_render as RenderUtils

from tqdm import tqdm
from copy import copy

import confs.path as PathConfig
from scipy.spatial.transform import Rotation as R

from datetime import datetime
from termcolor import cprint

import mcubes

def read_dict_tn_from_pkl(path_pkl):
    data_inp_tmp_np = IOUtils.read_pickle_to_class_instance(path_pkl)
    data_inp_tmp = {}
    for key in data_inp_tmp_np:
        if isinstance(data_inp_tmp_np[key], np.ndarray):
            data_inp_tmp[key] = torch.from_numpy(data_inp_tmp_np[key]).cuda()
        else:
            data_inp_tmp[key] = data_inp_tmp_np[key]
    return data_inp_tmp                


def perturb_code(code, noise_level):
    code_noise = torch.rand_like(code) * noise_level * 2 - noise_level
    return code_noise

def voxelize_bbox_to_vertices(inp_bbox, bbox_size, vox_num, mode='vertices'):
    '''
    paras:
        mode: indices or vertices
    '''
    vox_size = bbox_size / float(vox_num-1)
    
    n_emd_half = vox_num // 2
    inp_vox = inp_bbox / vox_size
    inp_vox = inp_vox.round().long()

    # only clamp center
    inp_vox[:,:, :3] = inp_vox[:,:, :3].clamp(-n_emd_half, n_emd_half-1)
    
    if mode=='vertices':
        inp_vertices = inp_vox*vox_size
        return inp_vertices
    elif mode == 'indices':
        inp_indices = inp_vox + n_emd_half
        return inp_indices.long()
    else:
        raise ValueError(
            f"Unsupported bbox voxelization mode {mode!r}; "
            "expected 'vertices' or 'indices'"
        )

def reverse_cossin_to_angle(pred_cossin, return_degree=True):
    '''
    paras:
        pred_arc: N*2, cos
    '''
    cos_angle = pred_cossin[...,0:1].mean().clamp(-1,1)
    sin_angle = pred_cossin[...,1:].mean().clamp(-1,1)
    
    angle_arccos = torch.arccos(cos_angle)  # 0~pi
    angle_arcsin = torch.arcsin(sin_angle)  # -pi/2~pi/2
    
    theta1 = angle_arcsin
    if sin_angle>0:
        # 0~pi
        theata2 = (np.pi-theta1)
        theta_cos = angle_arccos
    else:
        # -pi~0
        theata2 = (-np.pi - theta1) 
        theta_cos = -angle_arccos
        
    theta_sin = theta1 if cos_angle>0 else theata2
    theata_final = (theta_sin+theta_cos)/2.0
    
    if return_degree:
        theata_final = theata_final/3.14 * 2 * 90.0
    return theata_final

def transform_mesh_from_bbox_center_lwh(mesh, center, lwh, angle_rotation, should_paint_mesh=True, return_trans=False):
    scale = lwh.max() / 2.0
    trans_scale = np.diag([scale, scale, scale, 1.0])

    trans_mv2 = np.identity(4)
    trans_mv2[:3,3] = center   
    
    trans_rot_3x3 = R.from_rotvec(angle_rotation * np.array([0, 1, 0])).as_matrix()
    trans_rot = np.identity(4)
    trans_rot[:3, :3] = trans_rot_3x3
    
    trans_all = trans_mv2@trans_rot@trans_scale
    mesh = mesh.transform(trans_all)
    if should_paint_mesh:
        color = np.random.rand(3)*0.7+0.3
        mesh.paint_uniform_color(color)
    if return_trans:
        return mesh, trans_all
    return mesh

def convert_code_coords2bbox(code_coords_np, n_reso_bottleneck=8, return_tensor=True):
    '''
    Args:
        code_coords: N*3, numpy array
    '''
    # check parameters  
    code_coords = code_coords_np
      
    cube_side = 2.3
    grid_size = cube_side/(PathConfig.TSDF_Resolution-1)
    size_vox = (cube_side+grid_size)/n_reso_bottleneck
    half_cube = cube_side/2.0
    if PathConfig.scale_cube_side != 1.0:
        cube_side = cube_side * PathConfig.scale_cube_side
        n_reso_bottleneck *= PathConfig.scale_cube_side

        grid_size = cube_side / (PathConfig.TSDF_Resolution*PathConfig.scale_cube_side-1)
        size_vox = (cube_side+grid_size)/n_reso_bottleneck
        half_cube = cube_side/2.0

    kernel_topright = np.array([[size_vox, size_vox, size_vox]])
    code_coords_vert_bottomleft = code_coords*size_vox - half_cube # move 0~2.3 ->+-1.15
    code_coords_vert_topright = code_coords_vert_bottomleft + kernel_topright
    
    
    bb_patches = np.concatenate((code_coords_vert_bottomleft, code_coords_vert_topright), axis=1)
    bb_patches_lwh = GeoUtils.get_lwh_from_aabb(bb_patches, 0)

    if return_tensor:
        patch_bb_child = torch.from_numpy(bb_patches_lwh)
        return patch_bb_child

    return bb_patches_lwh

def get_model_info_from_path(path, exp_dir):     
    
    comp_path_inp = path.split('/')   
    if 'aug_' in path:
        cat, model_id, aug_name, view_name, pair_name = comp_path_inp[-5:]
    else:
        cat, model_id, view_name, pair_name = comp_path_inp[-4:]
        aug_name = ''
    
    pair_name = pair_name[:-3]
    folder_shape = f'{cat}/{model_id}/{aug_name}/{view_name}/{pair_name}'
    folder_shape_render = f'{cat}/{model_id}_{aug_name}_{view_name}_{pair_name}'

    # check dataset type
    if cat in PathConfig.Categorys_ShapeNet:
        name_dataset = 'shapenet'
        dir_root_dataset_ours = PathConfig.DIR_DATA_SHAPENET_OURS
    elif cat in PathConfig.Categorys_PartNet:
        name_dataset = 'partnet'
        dir_root_dataset_ours = PathConfig.DIR_ROOT_PARTNET
    else:
        raise ValueError(
            f"Unsupported dataset category {cat!r} in model path: {path}"
        )

    # check view name
    if view_name =='view_0000':
        path_gt_mesh = f'{dir_root_dataset_ours}/{PathConfig.Folder_Mesh_Aug}/{cat}/{model_id}/{aug_name}.ply'
    else:
        path_gt_mesh = f'{dir_root_dataset_ours}/{PathConfig.Folder_Mesh_Aug_Partial}/{cat}/{model_id}/{aug_name}/{view_name}.ply'
    
    dict_info_model = {
        'cat': cat,
        'model_id': model_id,
        'aug_name': aug_name,
        'view_name': view_name,
        'pair_name': pair_name,
        'folder_shape': folder_shape,
        'folder_shape_render': folder_shape_render,
        
        'name_dataset': name_dataset,
        'path_gt_mesh': path_gt_mesh,
        
        'dir_mesh_merge': f'{exp_dir}/mesh_merge/{folder_shape}',
        'dir_mesh_render_merge': f'{exp_dir}/mesh_render_merge/{folder_shape_render}',
        'path_mesh_render_merge2': f'{exp_dir}/mesh_render_merge/{folder_shape_render}.png',
    }
    return dict_info_model

def extract_geometry_lis2(lis_tsdf, threshold, resolution = 64,  lis_vec_occu=None, bbox_side=2.3, mode_mc='cpu'):
    n_tsdf = len(lis_tsdf)
    lis_mesh = []
    
    if mode_mc=='gpu':
        t1 = datetime.now()
        lis_vertices, lis_triangles  = GeoUtils.extract_geometry_gpu(lis_tsdf, threshold)
        t2 = datetime.now()
        print(f'[time] Extract mesh: {(t2-t1).total_seconds():.2f}')
    
    for i in tqdm(range(n_tsdf)):
        if lis_tsdf[i] is None:
            lis_mesh.append(None)
            print(f'Fail to extract mesh from None. [{i}]')
            continue
        
        if mode_mc=='gpu':
            vertices, triangles = lis_vertices[i], lis_triangles[i]
            t2 = datetime.now()
        elif mode_mc=='cpu':
            t1 = datetime.now()
            vertices, triangles = mcubes.marching_cubes(-lis_tsdf[i], -threshold)
            t2 = datetime.now()
        
        if lis_vec_occu is not None:
            scale_factor = 256 / resolution
            lis_occ_curr = lis_vec_occu[i] / scale_factor
            mesh_mc = GeoUtils.create_mesh_from_verts_faces(None, vertices, triangles)
            mesh_mc_fix = GeoUtils.fix_mesh_mc_by_occ(mesh_mc, lis_occ=lis_occ_curr, mode_quant='round')
            vertices, triangles = np.asarray(mesh_mc_fix.vertices), np.asarray(mesh_mc_fix.triangles)

        vertices = vertices / (resolution - 1.0) * bbox_side - bbox_side*0.5
        mesh_mc_fix_norm = GeoUtils.create_mesh_from_verts_faces(None, vertices, triangles)
        lis_mesh.append(mesh_mc_fix_norm)
        
        t3 = datetime.now()

        print(f'[{i}] Extract mesh total: {(t3-t1).total_seconds():.2f}')
    return lis_mesh

def paint_lis_mesh(lis_mesh, lis_idx_color, color_map):
    for i in range(len(lis_mesh)):
        if lis_mesh[i] is None:
            continue
        
        mesh_i = lis_mesh[i]
        try:
            color_i = color_map[lis_idx_color[i]]
        except (IndexError, KeyError, TypeError) as error:
            print(f'Use random color: {error}')
            color_i = np.random.rand(3)*0.7+0.3
                        
        mesh_i.paint_uniform_color(color_i)
    return lis_mesh

def merge_lis_parts(lis_parts_mesh_norm, lis_parts_bbox_lwh, get_max_cluster_part=False, should_paint_mesh=True):
    
    mesh_merge = None
    lis_parts_mesh_world = []; lis_trans_parts_norm2world = []
    for j_mesh in range(len(lis_parts_mesh_norm)):
        if lis_parts_mesh_norm[j_mesh] is None:
            continue
        
        bbox_center = lis_parts_bbox_lwh[j_mesh][:3]
        bbox_lwh = lis_parts_bbox_lwh[j_mesh][3:6]
        rotation_angle = 0
               
        mesh_i = lis_parts_mesh_norm[j_mesh]
        if get_max_cluster_part:
            mesh_i = GeoUtils.get_mesh_max_cluster(mesh_i, min_faces_cluster = 1000)
        
        mesh_i_trans, trans_norm2world = transform_mesh_from_bbox_center_lwh(copy(mesh_i), bbox_center, bbox_lwh, rotation_angle, 
                                                                            should_paint_mesh=False, 
                                                                            return_trans=True)

        if should_paint_mesh:
            mesh_i_trans = GeoUtils.paint_mesh_uniformly(mesh_i_trans, j_mesh)
        
        lis_trans_parts_norm2world.append(trans_norm2world)
        lis_parts_mesh_world.append(copy(mesh_i_trans))
        
        mesh_merge = (mesh_merge+mesh_i_trans) if j_mesh>0 else copy(mesh_i_trans)
        
    dict_info_mesh = {
        'mesh_merge': mesh_merge,
        'lis_mesh': lis_parts_mesh_world,
        'lis_mesh_norm': lis_parts_mesh_norm,
        'lis_bbox_centerlwh': lis_parts_bbox_lwh[:,:6],
        'lis_trans_norm2world': lis_trans_parts_norm2world,
    }
    return dict_info_mesh


def render_lis_parts(renderer, dict_info_mesh, path_gt_mesh):
    lis_mesh_world = dict_info_mesh['lis_mesh']
    mesh_merge = dict_info_mesh['mesh_merge']

    # render gt parts
    img_part_merge = RenderUtils.render_mesh(renderer, mesh_p3d=None, path_mesh=None, mesh_o3d=mesh_merge)[0]
    try:
        img_render_gt = RenderUtils.render_mesh(renderer, mesh_p3d=None, path_mesh=path_gt_mesh)[0]
    except Exception as error:
        cprint(f'Fail to render GT: {error}', 'yellow')
        img_render_gt = np.ones((384, 384, 3), dtype=np.uint8)*255

    # render predicted parts
    lis_img_part = []
    for i in range(len(lis_mesh_world)):
        mesh_world_i = lis_mesh_world[i]
        try:
            img_render_part = RenderUtils.render_mesh(renderer, mesh_p3d=None, mesh_o3d=[mesh_world_i], path_mesh=None)[0]
            lis_img_part.append(img_render_part*0.5 + 0.5*img_render_gt)
        except Exception as error:
            img_render_part = np.ones((384, 384, 3), dtype=np.uint8)*255
            print(f'Fail to render part: {error}')
            lis_img_part.append(img_render_part)
            
    dict_img_render ={
        'img_part_merge': img_part_merge,
        'img_gt': img_render_gt,
        'lis_img_part': lis_img_part
    }
    return dict_img_render

def merge_mesh_child2parent_by_link(lis_mesh_child, dict_p2c, n_parent_pred):
    
    dict_mesh_parent = {}
    for key in dict_p2c:
        lis_idx_child = dict_p2c[key]
        
        mesh_merge_curr = None
        for idx_child in lis_idx_child:
            mesh_curr = lis_mesh_child[idx_child]
            if mesh_curr is None:
                continue
            
            if mesh_merge_curr is None:
                mesh_merge_curr = copy(mesh_curr)
            else:
                mesh_merge_curr += mesh_curr
        dict_mesh_parent[key] = mesh_merge_curr
    
    # convert to list
    mask_valid = torch.ones(n_parent_pred).cuda().bool()
    if n_parent_pred != len(dict_mesh_parent):
        print('Predicted parent and parent in pointer is not equal.')
    lis_mesh_merge_parent = []
    for i in range(n_parent_pred):
        if i not in dict_mesh_parent:
            mask_valid[i]=False
            lis_mesh_merge_parent.append(None)
            continue
        
        if len(dict_mesh_parent[i].vertices)<1000:
            n_vertices = len(dict_mesh_parent[i].vertices)
            print(f'No suffient (1000) points in mesh. Vert:{n_vertices}')
            mask_valid[i]=False
            lis_mesh_merge_parent.append(None)
            continue    
                
        lis_mesh_merge_parent.append(dict_mesh_parent[i])
    
    return lis_mesh_merge_parent, mask_valid

def normliaze_bbox_lwh_to_cube(bbox_lwh):
    center, lwh = bbox_lwh[:3], bbox_lwh[3:6]
    scale = 2.0 / lwh.max()
    
    trans_scale = np.diag([scale, scale, scale, 1.0])
    
    trans_mv2orig = np.identity(4)
    trans_mv2orig[:3,3] = -center
    
    trans_w2n = trans_scale @ trans_mv2orig
    return trans_w2n

def normalize_lis_mesh2cube(lis_mesh, lis_bbox_lwh=None, dict_pointcloud=None):
    mask_valid_curr = np.ones(len(lis_mesh)).astype(bool)
    lis_mesh_new = []

    cloud_input = GeoUtils.create_pointcloud_from_points(dict_pointcloud['pointcloud']) if dict_pointcloud is not None else None
    for i in range(len(lis_mesh)):
        if lis_mesh[i] is None:
            mask_valid_curr[i] = False
            continue
        
        mesh_i = lis_mesh[i]
        bbox_lwh_i = lis_bbox_lwh[i]
        
        # find nn input
        if dict_pointcloud is not None:
            cloud_curr_part = GeoUtils.create_pointcloud_from_mesh(mesh_i)
            dists_curr = cloud_curr_part.compute_point_cloud_distance(cloud_input)
            dists_curr = np.asarray(dists_curr)
            
            thres_dist = 0.03
            mask_remove = dists_curr>thres_dist
            n_remove2 = mask_remove.sum()
            mesh_i.remove_vertices_by_mask(mask_remove)
            print(f'[{i}] Removed points {n_remove2}/{len(mask_remove)}.')
            n_points_curr = len(mesh_i.vertices)
            if n_points_curr<1000:
                mask_valid_curr[i] = False
                continue

            
        trans_bbox = normliaze_bbox_lwh_to_cube(bbox_lwh_i)
        
        mesh_i = mesh_i.transform(trans_bbox)
        
        
        # crop outside 
        thres_crop = 1.0
        points_norm = np.array(mesh_i.vertices)
        mask_outside = np.logical_or(points_norm<-thres_crop, points_norm>thres_crop)
        mask_remove = mask_outside.sum(axis=1)>0
            
        mesh_i.remove_vertices_by_mask(mask_remove)
        n_pts_clean = len(mesh_i.vertices)
        
        if n_pts_clean<1000:
            print(f'No sufficient points after cleanning. {n_pts_clean}<1000')
            mask_valid_curr[i] = False
            continue
                
        lis_mesh_new.append(copy(mesh_i))
    return lis_mesh_new, mask_valid_curr

def fix_memory_part_vae(dict_out_memory_part_vae_valid, mask_valid):
    '''
    paras:
        memory_inp: (T-1)*B
        memory_key_padding_mask: B*T
    '''
    n_parent_pred_re = len(mask_valid)
    
    t_seq = len(dict_out_memory_part_vae_valid['memory_inp']); dim_tf = 768
    mask_shape = dict_out_memory_part_vae_valid['memory_key_padding_mask'].shape
    if len(mask_shape) != 2 or mask_shape[1] != t_seq + 1:
        raise ValueError(
            "Stage2 memory padding mask must include one leading token: "
            f"memory_shape={tuple(dict_out_memory_part_vae_valid['memory_inp'].shape)}, "
            f"mask_shape={tuple(mask_shape)}"
        )
    
    dict_out_memory_part_vae = {}
    dict_out_memory_part_vae['memory_inp'] = torch.zeros((t_seq, n_parent_pred_re, dim_tf), dtype=dict_out_memory_part_vae_valid['memory_inp'].dtype).cuda()
    dict_out_memory_part_vae['memory_key_padding_mask'] = torch.ones((n_parent_pred_re, t_seq+1), dtype=dict_out_memory_part_vae_valid['memory_key_padding_mask'].dtype).cuda()
    dict_out_memory_part_vae['memory_key_padding_mask'][:,:1] = False # B*T
    
    dict_out_memory_part_vae['memory_inp'][:, mask_valid] = dict_out_memory_part_vae_valid['memory_inp']
    dict_out_memory_part_vae['memory_key_padding_mask'][mask_valid] = dict_out_memory_part_vae_valid['memory_key_padding_mask']
    
    return dict_out_memory_part_vae

def random_mask_patch_memory(memory_key_padding_mask, ratio_mask=0.5):
    if ratio_mask == 0.0:
        return memory_key_padding_mask
    
    b, _ = memory_key_padding_mask.shape
    for i in range(b):
        mask_curr = memory_key_padding_mask[i]
        n_valid = (mask_curr==False).sum().item()
        n_mask = int(n_valid*ratio_mask)

        idx_padding_curr = np.random.choice(n_valid, n_mask, replace=False)
        memory_key_padding_mask[i][idx_padding_curr] = True
        
        n_valid_mask = (memory_key_padding_mask[i]==False).sum().item()
        print(f'[{i}] {n_valid}-{n_mask} --> {n_valid_mask}.')
        
    memory_key_padding_mask[:,:1] = False
    return memory_key_padding_mask
