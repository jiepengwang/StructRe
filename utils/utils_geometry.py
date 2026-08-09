import glob, trimesh, logging
import open3d as o3d
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from copy import copy

# from torchmcubes import marching_cubes as marching_cubes_gpu
try:
    from pytorch3d.ops.marching_cubes import marching_cubes as marching_cubes_p3d
except (ImportError, OSError) as error:
    print(f'Failed to import PyTorch3D marching cubes: {error}')

from termcolor import cprint

try:
    from pytorch3d.ops import knn_points
    from utils.utils_chamfer_p3d import chamfer_distance_p3d
except (ImportError, OSError) as error:
    print(f'Failed to import PyTorch3D ops: {error}')


from einops import rearrange

import utils.utils_io as IOUtils
import point_cloud_utils as pcu

try:
    import mcubes, igl
except (ImportError, OSError) as error:
    print(f'Failed to import igl: {error}')

from scipy import stats

import confs.path as PathConfig

import trimesh
# import pymeshlab as ml



'''Content
    1. Geometry intersection

'''


'''Geometry intersection
'''
def quantify_points_to_grid_idx(pts, grid_reso = 32, cube_side=2.3, remove_duplicates=True, flat_grid_1d=True):
    '''
    paras:
        pts: N*3, numpy array, located inside a cube with side {cube_side}
    '''
    # quantify point cloud
    cube_side_half = cube_side/2
    grid_size = cube_side/grid_reso
    pts_grid = (pts + cube_side_half) / grid_size
    pts_grid = np.floor(pts_grid).astype(np.int64)
    
    pts_grid_1d = pts_grid
    if flat_grid_1d:
        coeff_idx = np.array([1, grid_reso, grid_reso**2])
        pts_grid_1d = pts_grid * coeff_idx[None, ...]
        pts_grid_1d = pts_grid_1d.sum(axis=1) #.squeeze(axis=1)
    
    if remove_duplicates:
        pts_grid_1d = np.unique(pts_grid_1d, axis=0)

    return pts_grid_1d

def check_iou_two_pointcloud(pts_src, pts_tgt, ratio_keep = 0.1, grid_reso = 16, cube_side=2.3, digit_round=2, verbose=False):
    pts_src_grid = quantify_points_to_grid_idx(pts_src, grid_reso = grid_reso, cube_side=cube_side, remove_duplicates=True, flat_grid_1d=True)
    pts_tgt_grid = quantify_points_to_grid_idx(pts_tgt, grid_reso = grid_reso, cube_side=cube_side, remove_duplicates=True, flat_grid_1d=True)

    intersect = np.intersect1d(pts_src_grid, pts_tgt_grid)
    union = np.union1d(pts_src_grid, pts_tgt_grid)
    iou_src = len(intersect) / len(pts_src_grid)
    iou_tgt = len(intersect) / len(pts_tgt_grid)
    iou = len(intersect) / len(union)

    should_keep = (iou<ratio_keep) and (iou_src<ratio_keep) and (iou_tgt<ratio_keep)
    
    return (round(iou, digit_round), round(iou_src, digit_round), round(iou_tgt, digit_round)), should_keep

def check_iou_two_tsdf(tsdf_src, dict_tsdf, ratio_keep = 0.1, verbose=False):
    mask_neg_src = tsdf_src < 0
    mask_neg_tgt = dict_tsdf < 0
    n_neg_src = mask_neg_src.sum()
    n_neg_tgt = mask_neg_tgt.sum()
    if verbose:
        print(f'Negative points (src, tgt): {n_neg_src}, {n_neg_tgt}.')

    mask_neg_union = mask_neg_src | mask_neg_tgt
    mask_neg_inter = mask_neg_src & mask_neg_tgt
    n_union = mask_neg_union.sum()
    n_inter = mask_neg_inter.sum()

    iou = n_inter / n_union
    iou_src = n_inter / (n_neg_src+1)
    iou_tgt = n_inter / (n_neg_tgt+1)

    should_keep = (iou<ratio_keep) and (iou_src<ratio_keep) and (iou_tgt<ratio_keep)

    digit_round = 2
    return (round(iou, digit_round), round(iou_src, digit_round), round(iou_tgt, digit_round)), should_keep

'''Mesh preprocessing
'''

def get_connected_components_mesh(path_mesh, thres_min_vert=1, should_paint_mesh=False, remove_duplicated_vertices=False,
                                    remove_unreferenced_vertices=False):
    mesh = read_triangle_mesh(path_mesh)
    if remove_duplicated_vertices:
        mesh.remove_duplicated_vertices()
    
    if remove_unreferenced_vertices:
        mesh.remove_unreferenced_vertices()

    faces = np.array(mesh.triangles)
    n_faces_all = len(faces)
    mat_con = igl.adjacency_matrix(faces)
    n_comp, arr_idx_comp, arr_size_comp = igl.connected_components(mat_con)
    if n_comp==1:
        return [mesh], mesh

    debug_ = False
    if debug_:
        for thres in range(0, 200, 10):
            mask_thres = arr_size_comp > thres
            n_valid = mask_thres.sum()
            print(f'Thres, valid: {thres} {n_valid}')

    lis_mesh = []; mesh_merge = None; n_count = 0
    for i in range(n_comp):
        size_curr = arr_size_comp[i]
        if size_curr < thres_min_vert:
            continue

        mask_faces_curr = (arr_idx_comp!=i)
        
        n_verts = len(mesh.vertices)
        n_mask = len(mask_faces_curr)
        if n_verts != n_mask:
            print(f'[Error] Number of vertices, mask: {n_verts} {n_mask}')
        
        mesh_i = copy(mesh)
        mesh_i.remove_vertices_by_mask(mask_faces_curr)
        if len(mesh_i.triangles)==0:
            continue
        
        debug_ = False
        if debug_:
            dir_parts = './test/parts'
            IOUtils.ensure_dir_existence(dir_parts)
            mesh_i = write_triangle_mesh(f'{dir_parts}/{n_count:04d}.ply', mesh_i, should_paint_mesh=True)
        
        if should_paint_mesh:
            color = np.random.rand(3)*0.7+0.3
            mesh_i.paint_uniform_color(color)
        lis_mesh.append(copy(mesh_i))

        if mesh_merge is None:
            mesh_merge = copy(mesh_i)
        else:
            mesh_merge += copy(mesh_i)
        n_count += 1

    debug_ = False
    if debug_:
        dir_parts = './test/parts'
        IOUtils.ensure_dir_existence(dir_parts)
        write_triangle_mesh(f'{dir_parts}/mesh_merge_{thres_min_vert}_count{n_count}.ply', mesh_merge)
        print(f'[{thres_min_vert}] Number of parts: {n_count}')

    return lis_mesh, mesh_merge


def get_chamfer_distance(path_src=None, path_tgt=None, cloud_src=None, cloud_tgt=None, use_sq_dist=True, voxel_size_downsample=-1):
    '''
    paras:
        use_sqrt_dist: default, True
    '''
    if cloud_src is None:
        s = read_point_cloud(path_src)
        t = read_point_cloud(path_tgt)
    else:
        s = cloud_src
        t = cloud_tgt

    if voxel_size_downsample>0:
        s=s.voxel_down_sample(voxel_size=voxel_size_downsample)
        t=t.voxel_down_sample(voxel_size=voxel_size_downsample)
        
    dist1 = s.compute_point_cloud_distance(t)
    dist1 = np.asarray(dist1)

    dist2 = t.compute_point_cloud_distance(s)
    dist2 = np.asarray(dist2)

    if use_sq_dist:
        dist1 = np.square(dist1)
        dist2 = np.square(dist2)

    n1 = len(dist1)
    n2 = len(dist2)

    m1 = dist1.mean()
    m2 = dist2.mean()

    # m = (m1*n1+m2*n2)/(n1+n2)
    m = 0.5*(m1+m2)
    return m

def get_earth_mover_distance(path_src=None, path_tgt=None, cloud_src=None, cloud_tgt=None, voxel_size_downsample=-1, 
                                    n_sample=1024, max_iters=100,
                                    mode_geometry='open3d'): 
    '''
    paras:
        point cloud: N*3, numpy array
    ''' 

    use_downsampled_points = False
    if use_downsampled_points:
        if cloud_src is None:
            s = o3d.io.read_point_cloud(path_src)
            t = o3d.io.read_point_cloud(path_tgt)
        else:
            s = cloud_src
            t = cloud_tgt
        
        if voxel_size_downsample>0:
            s=s.voxel_down_sample(voxel_size=voxel_size_downsample)
            t=t.voxel_down_sample(voxel_size=voxel_size_downsample)
        

        point_cloud_1 = np.array(s.points) #[:100]
        point_cloud_2 = np.array(t.points) #[:100]\
        n_sample = min(len(point_cloud_1), len(point_cloud_2))

        if n_sample ==0:
            print(f'No points in mesh')
            return np.NAN

    if path_src is None and mode_geometry not in {'mesh', 'pointcloud'}:
        raise ValueError(
            f"Unsupported EMD geometry mode {mode_geometry!r}; "
            "expected 'mesh' or 'pointcloud'"
        )

    try:
        if path_src is not None:
            point_cloud_1, _ = sample_points_on_mesh(read_triangle_mesh(path_src), n_sample=n_sample, return_points=True)
            point_cloud_2, _ = sample_points_on_mesh(read_triangle_mesh(path_tgt), n_sample=n_sample, return_points=True)
        elif mode_geometry == 'mesh':
            point_cloud_1, _ = sample_points_on_mesh(cloud_src, n_sample=n_sample, return_points=True)
            point_cloud_2, _ = sample_points_on_mesh(cloud_tgt, n_sample=n_sample, return_points=True)
        else:
            point_cloud_1 = sample_points_on_pointcloud(cloud_src, n_sample=n_sample)
            point_cloud_2 = sample_points_on_pointcloud(cloud_tgt, n_sample=n_sample)
    except (RuntimeError, ValueError) as error:
        print(f'Failed to sample EMD points: {error}')
        return np.nan


    mode_cal_emd = 'fwilliams'
    # if mode_cal_emd=='gpu':
    #     point_cloud_1 = torch.from_numpy(point_cloud_1).cuda()[None, ...]
    #     point_cloud_2 = torch.from_numpy(point_cloud_2).cuda()[None, ...]
    #     with torch.no_grad():
    #         emd1 = earth_mover_distance(point_cloud_1, point_cloud_2, transpose=False).item()
    #         torch.cuda.empty_cache()

    #     emd = emd1 / n_sample

    # if mode_cal_emd=='cpu':
    #     import ot  
    #     # Compute the cost matrix  
    #     cost_matrix = ot.dist(point_cloud_1, point_cloud_2, metric='euclidean')  
    
    #     # Normalize the weights of the points  
    #     weights1 = np.ones(point_cloud_1.shape[0]) / point_cloud_1.shape[0]  
    #     weights2 = np.ones(point_cloud_2.shape[0]) / point_cloud_2.shape[0]  
    
    #     # Compute the Earth Mover's Distance  
    #     emd = ot.emd2(weights1, weights2, cost_matrix)  
    
    if mode_cal_emd=='fwilliams':
        # point_cloud_2 = point_cloud_1 + 0.01
        # Compute the chamfer distance between p1 and p2
        emd2, pi2 = pcu.earth_movers_distance(point_cloud_1, point_cloud_2, max_iters=max_iters)
        emd = np.square(emd2)
        # GeoUtils.save_points('./test/p1.ply', point_cloud_1)
        # GeoUtils.save_points('./test/p2.ply', point_cloud_2)
    return emd  


'''TSDF grid
'''
def fix_mesh_mc_by_occ(mesh_mc, lis_occ, mode_quant='round'):

    lis_occ = torch.from_numpy(lis_occ).cuda()
    
    vertices = np.asarray(mesh_mc.vertices)
    if mode_quant=='floor':
        vertices_quant = np.floor(vertices)
    elif mode_quant=='round':
        vertices_quant = np.round(vertices, decimals=0) 
    else:
        raise NotImplementedError
    vertices_quant = torch.from_numpy(vertices_quant).cuda()
    
    dist, _ = chamfer_distance_p3d(vertices_quant.float()[None, ...], lis_occ.float()[None, ...], batch_reduction=None, point_reduction=None, single_directional=True)
    dist_min = dist.cpu()[0]
    # dist = torch.cdist(vertices_quant.float(), lis_occ.float(), p=2)
    # dist_min = dist.min(dim=1)[0]
    thres_rm = 8e-4
    mask_remove = (dist_min > thres_rm).cpu().numpy()
    # n_remove2 = (dist_min > 0).cpu().numpy().sum()
        
    # n_remove = mask_remove.sum()
    # print(f'Number of vertices removed: {n_remove} {n_remove2}')
    mesh_mc.remove_vertices_by_mask(mask_remove)
        
    return mesh_mc

def extract_geometry_gpu(tsdf_vol, vol):
    '''
    paras:
        tsdf_vol: torch tensor, N*H*W*D
        
    '''
    if isinstance(tsdf_vol, np.ndarray):
        tsdf_vol = torch.from_numpy(tsdf_vol).cuda()
    if not tsdf_vol.is_cuda:
        tsdf_vol = tsdf_vol.cuda()
    
    verts, faces = marching_cubes_p3d(tsdf_vol.float().permute(0,3,2,1), vol, return_local_coords=False)

    for i in range(len(verts)):
        verts[i] = verts[i].cpu().numpy()
        faces[i] = faces[i].cpu().numpy()
    return verts, faces

    
def extract_geometry(path_mesh, tsdf_vol, threshold, bbox, resolution = 64, lis_occ=None, mode_mc='gpu'):
    # logging.info('threshold: {}'.format(threshold))
    # u = extract_fields(bound_min, bound_max, resolution, query_func)
    if mode_mc=='gpu':
        vertices, triangles  = extract_geometry_gpu(tsdf_vol[None, ...], threshold)
    elif mode_mc=='cpu':
        vertices, triangles = mcubes.marching_cubes(-tsdf_vol, -threshold)
    else:
        raise NotImplementedError

    if lis_occ is not None:
        mesh_mc = create_mesh_from_verts_faces(None, vertices, triangles)
        mesh_mc_fix = fix_mesh_mc_by_occ(mesh_mc, lis_occ, mode_quant='round')
        vertices, triangles = np.asarray(mesh_mc_fix.vertices), np.asarray(mesh_mc_fix.triangles)

    b_max_np = bbox[3:]
    b_min_np = bbox[:3]
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    # mesh = trimesh.Trimesh(vertices, triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices) 
    mesh.triangles = o3d.utility.Vector3iVector(triangles) 

    if path_mesh is not None:
      write_triangle_mesh(path_mesh, mesh)
      # mesh1.triangle_normals = o3d.utility.Vector3dVector(
      #     np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])

      # mesh.export(path_mesh)
    return mesh

def extract_geometry_tsdf256(path_mesh, tsdf_vol, threshold, bbox, resolution = 256, should_paint_mesh=False, 
                                    vec_occu=None, 
                                    reso_mink_grid=128):
    
    
    
    if resolution==128 and reso_mink_grid==128:
        ratio_mc=1.0
        ratio_mink_grid = 1.0
    elif resolution==256 and reso_mink_grid==128:
        ratio_mc=2.0
        ratio_mink_grid = 0.5
    elif resolution==256 and reso_mink_grid==256:
        # ratio_mc=1.0
        ratio_mink_grid = 1.0
        n_grid_mc = 255
        ratio_upsample = 2.0
    elif resolution==128 and reso_mink_grid==256:
        # ratio_mc=1.0
        ratio_mink_grid = 2.0
        n_grid_mc = 255
        ratio_upsample = 2.0
        
    vertices, triangles = mcubes.marching_cubes(-tsdf_vol, -threshold)
    vertices = vertices*ratio_mink_grid

    b_max_np = bbox[3:]
    b_min_np = bbox[:3]
    
    if vec_occu is not None:
      # 2.3 method 3
        # cdist version
        vertices_quant = np.round(vertices/ratio_upsample, decimals=0) * ratio_upsample   # near vertices
        vertices_quant = torch.from_numpy(vertices_quant).cuda().float()
        vec_occu = torch.from_numpy(vec_occu).cuda().float()
        
        vertices_quant2 = torch.from_numpy( (vertices//ratio_upsample) * ratio_upsample ).cuda().float() # quan grid
        
        mode_dist = 'knn'
        if mode_dist == 'cdist':
            # 3. get valid vertices
            lis_mask = []
            lis_n_remove = []
            for vectices_quant_i in [vertices_quant, vertices_quant2]:
                n_chunk = 65536
                chunks = vectices_quant_i.split(n_chunk)
                lis_mask_remove_i = []
                for chunk_i in chunks:
                    dist = torch.cdist(chunk_i.float().cuda(), vec_occu.float(), p=2).cpu()
                    dist_min = dist.min(dim=1)[0]
                    mask_remove_chunk = (dist_min > 0).cpu()
                    
                    lis_mask_remove_i.append(mask_remove_chunk)
                del chunks
                del vectices_quant_i
                
                mask_remove = torch.cat(lis_mask_remove_i, dim=0).numpy()
                n_remove = mask_remove.sum()
                
                lis_mask.append(mask_remove)
                lis_n_remove.append(n_remove)
                
            mask_remove = np.logical_or(lis_mask[0], lis_mask[1])
            #   mask_remove = lis_mask[0]
            n_remove = mask_remove.sum()
        elif mode_dist=='knn':
            dist1_round = knn_points(vertices_quant[None, ...], vec_occu[None, ...]).dists.cpu()
            dist2_occu = knn_points(vertices_quant2[None, ...], vec_occu[None, ...]).dists.cpu()

            # to cpu
            vertices_quant = vertices_quant.cpu()
            vertices_quant2 = vertices_quant2.cpu()
            vec_occu = vec_occu.cpu()

            mask1 = dist1_round.squeeze().numpy()>0
            mask2 = dist2_occu.squeeze().numpy()>0

            n_remove1 = mask1.sum()
            n_remove2 = mask2.sum()

            mask_remove = np.logical_or(mask1, mask2)
            n_remove = mask_remove.sum()
            
    vertices = vertices / (n_grid_mc) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    # mesh = trimesh.Trimesh(vertices, triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices) 
    mesh.triangles = o3d.utility.Vector3iVector(triangles) 
    
    if vec_occu is not None:
        mesh.remove_vertices_by_mask(mask_remove)

    if should_paint_mesh:
        color = np.random.rand(3)*0.7+0.3
        mesh.paint_uniform_color(color)

    if path_mesh is not None:
        write_triangle_mesh(path_mesh, mesh)
    return mesh


def unfold_to_cubes(x, cube_size=8, stride=8, should_return_tensor=False):
    """ 
        assume x.shape: b, c, d, h, w
        x: torch tensor
        return: x_cubes: (b cubes), numpy array
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
    x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
    x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

    if should_return_tensor:
        return x_cubes
    else:
        return x_cubes.cpu().numpy()

def visualize_voxel_grid(path_voxel, grid_points, kernel_size=1, side_size=8, scale_cube=1):
    kernel = np.array([[1,1,1]])*kernel_size
    grid_points_upper = grid_points + kernel
    lis_voxels = np.concatenate([grid_points, grid_points_upper], axis=1)

    grid_size = 2.3/255
    lis_voxels = lis_voxels/side_size * (2.3 + grid_size )- 1.15
    lis_voxels = lis_voxels*scale_cube
    create_boxes_from_bbox_lis(path_voxel, lis_voxels)

def read_triangle_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh

def write_triangle_mesh(path_save, mesh, should_paint_mesh=False, check_existence=True):
    # assert IOUtils.ensure_dir_existence(os.path.dirname(path_save))
    if check_existence:
        IOUtils.ensure_pdir_existence(path_save)
        
    if should_paint_mesh:
        color = np.random.rand(3)*0.7+0.3
        mesh.paint_uniform_color(color)
    o3d.io.write_triangle_mesh(path_save, mesh)
    return mesh
    
def resize_cam_intrin(intrin, resolution_level):
    if resolution_level != 1.0:
        logging.info(f'Resize instrinsics, resolution_level: {resolution_level}')
        intrin[:2,:3] /= resolution_level
    return intrin

def read_cam_matrix(path, data = ''):
    '''load camera intrinsics or extrinsics
    '''
    if path is not None:
        data = open(path)
    
    lines = [[float(w) for w in line.strip().split()] for line in data]
    assert len(lines) == 4
    return np.array(lines).astype(np.float32)

# camera pose related functions
def get_camera_origins(poses_homo):
    '''
    Args:
        poses_homo: world to camera poses
    '''
    if not isinstance(poses_homo, np.ndarray):
        poses_homo = np.array(poses_homo)
    cam_centers = []
    poses_homo = np.array(poses_homo)
    num_cams = poses_homo.shape[0]
    for i in range(num_cams):
        rot = poses_homo[i, :3,:3]
        trans = poses_homo[i, :3,3]
        trans = trans.reshape(3,1)
        cam_center = - np.linalg.inv(rot) @ trans
        cam_centers.append(cam_center)
    cam_centers = np.array(cam_centers).squeeze(axis=-1)
    return cam_centers
        
def get_world_points(depth, intrinsics, extrinsics):
    '''
    Args:
        depthmap: H*W
        intrinsics: 3*3 or 4*4
        extrinsics: 4*4, world to camera
    Return:
        points: N*3, in world space 
    '''
    if intrinsics.shape[0] ==4:
        intrinsics = intrinsics[:3,:3]
        
    height, width = depth.shape

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # valid_points = np.ma.masked_greater(depth, 0.0).mask
    # x, y, depth = x[valid_points], y[valid_points], depth[valid_points]

    x = x.reshape((1, height*width))
    y = y.reshape((1, height*width))
    depth = depth.reshape((1, height*width))

    xyz_ref = np.matmul(np.linalg.inv(intrinsics),
                        np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(extrinsics),
                            np.vstack((xyz_ref, np.ones_like(x))))[:3]
    xyz_world = xyz_world.transpose((1, 0))

    return xyz_world

def get_aabb(points, scale=1.0, return_6Dbox=False):
    '''
    Args:
        points; 1) numpy array (converted to '2)'; or 
                2) open3d cloud
    Return:
        min_bound
        max_bound
        center: bary center of geometry coordinates
    '''
    if isinstance(points, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        points = point_cloud
    # if os.path.isfile(points):
    #     if points.endswith('.ply'):
    #         points = o3d.io.read_point_cloud(points)    
    #     else:
    #         points = o3d.io.read_triangle_mesh(points)    
    min_max_bounds = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(points)
    min_bound, max_bound = min_max_bounds.min_bound, min_max_bounds.max_bound
    center = (min_bound+max_bound)/2
    # center = points.get_center()
    if scale != 1.0:
        min_bound = center + scale * (min_bound-center)
        max_bound = center + scale * (max_bound-center)

    if return_6Dbox:
        return np.concatenate([min_bound, max_bound])
    # logging.info(f"min_bound, max_bound, center: {min_bound, max_bound, center}")
    return min_bound, max_bound, center

def get_obb(points, scale=1.0):
    '''
    Args:
        points; 1) numpy array (converted to '2)'; or 
                2) open3d cloud, mesh
    Return:
        obb
    '''
    if isinstance(points, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        points = point_cloud

    # aabb = points.get_axis_aligned_bounding_box()
    # aabb.color = (1, 0, 0)

    obb = points.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    # o3d.visualization.draw_geometries([points, aabb, obb],
    #                                 zoom=0.7,
    #                               front=[0.5439, -0.2333, -0.8060],
    #                               lookat=[2.4615, 2.1331, 1.338],
    #                               up=[-0.1781, -0.9708, 0.1608])

    return obb

def get_lwh_from_aabb(bbox_aabb, angle_rotation=0):
    vert_min, vert_max = bbox_aabb[...,:3], bbox_aabb[...,3:]
    center = (vert_min+vert_max)/2.0
    XX = vert_max[...,2]-vert_min[...,2]   # z -> L
    YY = vert_max[...,0]-vert_min[...,0]   # x -> W
    ZZ = vert_max[...,1]-vert_min[...,1]   # y -> H
    
    L = np.cos(angle_rotation)*(XX) + np.sin(angle_rotation)* (YY)
    W = -np.sin(angle_rotation)*(XX) + np.cos(angle_rotation)* (YY)
    H = ZZ
    L = abs(L)
    W = abs(W)
    
    bbox_lwh = np.concatenate([center, L[..., None], W[..., None], H[..., None]], axis=-1)
    return bbox_lwh

def read_poses(dir, lis_stem = None, ext = '.txt'):
    '''Read camera poses
    '''
    vec_path = sorted(glob.glob(f"{dir}/**{ext}"))
    if lis_stem is not None:
        vec_path = []
        for stem_curr in lis_stem:
            vec_path.append(f'{dir}/{stem_curr}{ext}')
        vec_path = sorted(vec_path)

    poses = []
    stems_all =  []
    for i in range(len(vec_path)):
        pose_i = read_cam_matrix(vec_path[i])
        poses.append(pose_i)

        _, stem_, _ = IOUtils.get_path_components(vec_path[i])
        stems_all.append(stem_)
    if lis_stem is not None:
        return np.array(poses), stems_all
    else:
        return np.array(poses)

def fuse_depthmaps_to_points_world(depthmaps, intrinsics, extrinsics):
    '''
    args:
        extrinsics: world to camera
    return:
        merged depth map points
    '''
    points_fuse = None
    for i in range(depthmaps.shape[0]):
        cam_ext, depth =  extrinsics[i], depthmaps[i]
        cam_int = intrinsics if intrinsics.ndim == 2 else intrinsics[i]
        #     depth = scale * depth
        points = get_world_points(depth, cam_int, cam_ext)
        if points_fuse is None:
            points_fuse = points
        else:
            points_fuse = np.concatenate((points_fuse, points), axis = 0)
    return points_fuse

def create_mesh_from_verts_faces(path_mesh, verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts) 
    mesh.triangles = o3d.utility.Vector3iVector(faces) 
    if path_mesh is not None:
        write_triangle_mesh(path_mesh, mesh)
    return mesh

def convert_mesh2vf(mesh):
    vert = np.array(mesh.vertices)
    face = np.array(mesh.triangles)
    return (vert, face)

def convert_vf2mesh(vf):
    vert, face = vf
    mesh = create_mesh_from_verts_faces(None, vert, face)
    return mesh

def convert_dict_mesh2vf(dict_mesh):
    dict_vf = {}
    for key in dict_mesh:
        dict_vf[key] = convert_mesh2vf(dict_mesh[key])
    return dict_vf

def convert_dict_vf2mesh(dict_vf):
    dict_mesh = {}
    for key in dict_vf:
        dict_mesh[key] = convert_vf2mesh(dict_vf[key])
    return dict_mesh

def save_points(path_save, pts, colors = None, normals = None, BRG2RGB=False, check_pdir_existence=True):
    '''save points to point cloud using open3d
    '''
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors) 
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals) 
    if check_pdir_existence:
        IOUtils.ensure_pdir_existence(path_save)
    o3d.io.write_point_cloud(path_save, cloud)

def write_point_cloud(path_save, cloud, ensure_pdir_existence=True):
    if ensure_pdir_existence:
        IOUtils.ensure_pdir_existence(path_save)
    o3d.io.write_point_cloud(path_save, cloud)
    
def read_point_cloud(path_cloud):
    assert IOUtils.checkExistence(path_cloud)
    cloud = o3d.io.read_point_cloud(path_cloud)
    # o3d.visualization.draw_geometries([cloud])
    return cloud

def create_pointcloud_from_points(points, normals=None, colors=None):
    '''
    paras:
        points: N*3
    '''
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud

def create_pointcloud_from_mesh(mesh):
    '''
    paras:
        points: N*3
    '''
    points = np.array(mesh.vertices)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud

def merge_lis_points_to_cloud(lis_points, use_color_map=True):
    cloud_merge = None
    for i in range(len(lis_points)):
        points_i = lis_points[i].reshape(-1,3)
        cloud_i = create_pointcloud_from_points(points_i)
        cloud_i = paint_mesh_uniformly(cloud_i, idx_color= (i if use_color_map else None))
        
        if cloud_merge is None:
            cloud_merge = copy(cloud_i)
        else:
            cloud_merge += cloud_i
    return cloud_merge

def merge_lis_mesh(lis_mesh, is_paint_mesh=True):
    mesh_merge = None
    for i in range(len(lis_mesh)):
        mesh_i = lis_mesh[i]
        
        if is_paint_mesh:
            mesh_i = paint_mesh_uniformly(mesh_i, idx_color=i)
        
        if mesh_merge is None:
            mesh_merge = copy(mesh_i)
        else:
            mesh_merge += mesh_i
    return mesh_merge

def generate_rays(img_size, intrin, pose = None, normalize_dir = True):
    '''Generate rays with specified size, intrin and pose.
    Args:
        intrin: 4*4
        pose: 4*4, (default: None, identity), camera to world
    Return:
        rays_o, rays_d: H*W*3, numpy array
    '''
    if pose is None:
        pose = np.identity(4)
    pose = torch.tensor(pose)
    intrin = torch.tensor(intrin)

    W,H = img_size
    tu = torch.linspace(0, W - 1, W)
    tv = torch.linspace(0, H - 1, H)
    pixels_v, pixels_u = torch.meshgrid(tv, tu)
    p = torch.stack([pixels_u, pixels_v, torch.ones_like(pixels_v )], dim=-1) # W, H, 3
    p = torch.matmul(torch.linalg.inv(intrin)[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    if normalize_dir:
        # for volume rendering, depth along ray
        rays_v = p  / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    else:
        # for reprojection of depthmap, depth along z axis
        rays_v = p
    rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
    rays_o = pose[None, None, :3, 3].expand(rays_v.shape) 
    return rays_o.cpu().numpy(), rays_v.cpu().numpy()

def reproject_points_W2I(points_homo, intrin, extrin, idx_mode = 'xy'):
    '''
    Args:
        points_homo: N*4
    Return:
        img_indices: default xy mode else uv mode
        depth: of all points
    '''
    # reproject points to camera coordinates
    if points_homo.shape[1] == 3:
        points_homo = np.concatenate( (points_homo, np.ones( (points_homo.shape[0],1) ) ), axis=1 )
    points_homo_cam = extrin@points_homo.transpose((1, 0))
    # depth = points_homo_cam[2]

    points_homo_image = intrin @ points_homo_cam
    depth = points_homo_image[2]
    points_homo_image = (points_homo_image / depth)[:2] # x,y: u,v
    
    if idx_mode == 'xy':
        points_homo_image = np.stack([points_homo_image[1], points_homo_image[0]], axis=0)

    return points_homo_image.transpose(), depth

def reproject_depth_I2W(depth, intrin, extrin):
    '''
    Args:
        depth: depthmap, not volume rendering depth
        points_homo: N*4
    '''
    # reproject points to camera coordinates
    size_img = depth.shape[::-1]
    rays_o, rays_d = generate_rays(size_img, intrin, np.linalg.inv(extrin), normalize_dir = False)
    pts = rays_o + rays_d * depth[..., None]
    return pts

def sample_sphere(point_N, dim=3, use_half_sphere_y=False):
    """
        uniformly sample points on a sphere 
        https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    """
    vec = np.random.rand(point_N, dim) * 2 - 1  # sample range: -1~1
    vec /= np.linalg.norm(vec, axis=1)[...,None]
    if use_half_sphere_y:
        vec[0,1] = np.abs(vec[0, 1])
        # print(f'Flip y axis')   
    return vec

def sphericalFlip(points, center, param):
    n = len(points) # total n points
    points = points - np.repeat(center, n, axis = 0) # Move C to the origin
    normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere
    flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
    flippedPoints += points 
    return flippedPoints

def hidden_point_removal(cloud, campos):
    C = campos # View Point, which is well above the point cloud in z direction
    # C = np.array([campos]) # View Point, which is well above the point cloud in z direction
    flippedCloud = sphericalFlip(cloud, C, np.pi) # Reflect the point cloud about a sphere centered at C
    points = np.append(flippedCloud, [[0,0,0]], axis = 0) # All points plus origin
    hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.
    visible = hull.vertices[:-1] # remove origin
    return cloud[visible], visible

def get_camera_pose(cam_center):
    view_dir_z = -cam_center.squeeze()
    view_dir_z_proj = np.array([view_dir_z[0], view_dir_z[1], 0])
    dir_x = np.cross(view_dir_z, view_dir_z_proj)

    head_dir_y = np.cross(view_dir_z, dir_x)
    rot = np.stack([dir_x, head_dir_y, view_dir_z], axis=0).transpose()
    rot = rot / np.linalg.norm(rot, axis=0, keepdims=True)

    trans = np.identity(4)
    trans[:3, :3] = rot  #  camera to world
    trans[:3, 3] = cam_center
    
    return trans

def get_visible_points_virtual_scan(points_cloud, mode_camera_sample='random_bbox_corner', half_cube_side=3, lis_idx_cam=None, normals=None):
    '''
    paras:
        points: N*3, numpy array
    return:
        mask_vis_merge: (n, ), bool
        dict_info_view: 
    '''
    
    if mode_camera_sample=='random_sphere':
        cam_center = 4*sample_sphere(1, use_half_sphere_y=True)

        debug_ = False 
        if debug_:
            cam_center = np.array([[-0.58, 0.58, 0.58]]) * 4           
            # cam_center = np.array([[0.1, 0.1,1]]) * 4           
            # cam_center = np.array([[0.7, 0.7,0.1]]) * 4 
                            
        camera_pose = get_camera_pose(cam_center)
        dict_info_view = {
            'cam_center': cam_center.tolist(),
            'camera_pose': camera_pose.tolist() # camera to world
        }

        lis_cam_sample = [cam_center[0]]

    elif mode_camera_sample=='random_bbox_corner':
        # randomly select
        
        radius_view = np.sqrt(half_cube_side*half_cube_side*3)
        lis_bbox_center  = np.array([[-half_cube_side, half_cube_side, -half_cube_side],
                                        [ half_cube_side, half_cube_side, -half_cube_side],
                                        [-half_cube_side, half_cube_side,  half_cube_side],
                                        [ half_cube_side, half_cube_side,  half_cube_side],
                                        [ 0,              0,               radius_view  ],  # idx 4
                                        [ 0,              0,              -radius_view  ],  # idx 5
                                        [ 0,              radius_view,     0            ],  # idx 6
                                        [ 0,             -radius_view,     0            ],  # idx 7
                                        [ radius_view,    0,               0            ],  # idx 8
                                        [-radius_view,    0,               0            ]   # idx 9
                                        ])
        # lis_idx_sample = [2, 3]
        n_sample = np.random.choice(4,1)+1
        lis_idx_sample = np.random.choice(4, n_sample, replace=False).tolist()
        if lis_idx_cam is not None:
            lis_idx_sample = lis_idx_cam
            cprint(f'Use given camera idx: {lis_idx_sample}', 'yellow')

        else:
            print(f'Sample camera: {lis_idx_sample}')


        lis_cam_sample = lis_bbox_center[lis_idx_sample]
        dict_info_view = {
            'mode_camera_sample': mode_camera_sample,
            'lis_idx_sample': lis_idx_sample,
            'lis_cam_sample': np.array(lis_cam_sample).tolist(),
        }
    elif mode_camera_sample=='fps':
        lis_viewpoints_sphere = copy(PathConfig.get_viewpoints_sphere())
        np.random.shuffle(lis_viewpoints_sphere)
        
        lis_n_views = np.array([3,4,5])
        idx_view_select = np.random.choice(3,1)
        n_views_curr = lis_n_views[idx_view_select][0]
        
        cloud_view_points = create_pointcloud_from_points(lis_viewpoints_sphere)
        cloud_sample = cloud_view_points.farthest_point_down_sample(n_views_curr)
        lis_cam_sample = np.array(cloud_sample.points)

        dict_info_view = {
            'mode_camera_sample': mode_camera_sample,
            'lis_cam_sample': lis_cam_sample,
        }
    else:
        assert False
    
    mask_vis_merge = get_visible_points_cameras(points_cloud, lis_cam_sample)
    debug_ = False
    if debug_:
        write_point_cloud('./test/vis2/cam.ply', cloud_sample)
        save_points('./test/vis2/cam_vis.ply', points_cloud[mask_vis_merge])
            
    return mask_vis_merge, dict_info_view

def get_visible_points_cameras(points_cloud, lis_cameras):
    # lis_cameras = np.array(lis_cameras)
    mask_vis_merge = np.zeros(len(points_cloud)).astype(bool)
    n_cam = len(lis_cameras)
    for i in range(n_cam):
        cam_center = lis_cameras[i]
        _, mask_vis = hidden_point_removal(copy(points_cloud), cam_center[None, ...])
        mask_vis_merge[mask_vis] = True
    return mask_vis_merge

def create_box_from_bbox(vert_min, vert_max):
    '''Create box
    '''
    size_box = vert_max - vert_min
    size_box = np.where(size_box > 0, size_box, 0.01)
    # center = (vert_max - vert_min) / 2.0
    box = o3d.geometry.TriangleMesh.create_box(width=size_box[0], height=size_box[1], depth=size_box[2])
    box_origin = box.translate(vert_min)
    return box_origin

def create_boxes_from_bbox_lis(path_mesh, lis_bboxes, should_paint_mesh=True, map_color=None):
    '''Create box with center located at origin
    paras:
        lis_bboxes: N*6
    '''
    num_boxes = len(lis_bboxes)
    if should_paint_mesh and map_color is None:
        map_color = PathConfig.get_color_map()
    for i in range(num_boxes):
        vert_min, vert_max = lis_bboxes[i][:3], lis_bboxes[i][3:]
        mesh_box_i = create_box_from_bbox(vert_min, vert_max)
        if should_paint_mesh:                
            try:
                color = map_color[i]
            except (IndexError, KeyError, TypeError):
                color = np.random.rand(3)*0.7+0.3
            mesh_box_i.paint_uniform_color(color)

        if i == 0:
            mesh = mesh_box_i
        else:
            mesh += mesh_box_i
            
    if path_mesh is not None and num_boxes > 0:
        write_triangle_mesh(path_mesh, mesh)

def create_box_from_center_lwh(center, lwh, angle_rotation, should_paint_mesh=True, color_paint=None):
    # rotation along Y axix
    # bbox_lwh = np.concatenate([center, L[..., None], W[..., None], H[..., None]], axis=-1)
    L, W, H = (lwh.clip(0, 10)+1e-5)
    box = o3d.geometry.TriangleMesh.create_box(width=W, height=H , depth=L)
    trans_mv = np.identity(4)
    trans_mv[:3,3] = np.array([-W/2.0, -H/2.0, -L/2.0])
    # trans_mv[:3,3] = np.array([-L/2.0, -W/2.0, -H/2.0])

    trans_mv2 = np.identity(4)
    trans_mv2[:3,3] = center   
    
    trans_rot_3x3 = R.from_rotvec(angle_rotation * np.array([0, 1, 0])).as_matrix()
    trans_rot = np.identity(4)
    trans_rot[:3, :3] = trans_rot_3x3
    
    box = box.transform(trans_mv2@trans_rot@trans_mv)
    if should_paint_mesh:
        if color_paint is not None:
            color=color_paint
        else:
            color = np.random.rand(3)*0.7+0.3
        box.paint_uniform_color(color)
    return box

def create_boxes_from_lwh_lis(path_mesh, lis_bboxes, angle_rotation=0):
    '''Create box with center located at origin
    paras:
        lis_bboxes: N*6
    '''
    num_boxes = len(lis_bboxes)
    map_color = PathConfig.get_color_map()
    for i in range(num_boxes):
        center, lwh = lis_bboxes[i][:3], lis_bboxes[i][3:6]

        if (map_color is not None) and i < map_color.shape[0]:
            color_paint = map_color[i]
        else:
            color_paint = None

        if i == 0:
            mesh = create_box_from_center_lwh(center, lwh, angle_rotation, color_paint=color_paint)
        else:
            mesh += create_box_from_center_lwh(center, lwh, angle_rotation, color_paint=color_paint)
    if path_mesh is not None and num_boxes > 0:
        write_triangle_mesh(path_mesh, mesh)
    return mesh
        
def sample_pc_trimesh(mesh_o3d, n_points=2048):
    v, f = np.array(mesh_o3d.vertices), np.array(mesh_o3d.triangles)

    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points

def sample_points_on_mesh(
    mesh,
    n_sample,
    return_dict_data=False,
    return_points=False,
    calcualte_normal=True,
):
    if calcualte_normal:
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    cloud_from_mesh = mesh.sample_points_uniformly(
        number_of_points=int(n_sample * 1.1)
    )
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=n_sample,
        pcl=cloud_from_mesh,
    )

    if return_points:
        return np.asarray(pcd.points), np.asarray(pcd.normals)

    if return_dict_data:
        return pcd, {
            "normal": np.asarray(pcd.normals),
            "pointcloud": np.asarray(pcd.points),
        }

    return pcd

def sample_points_on_pointcloud(cloud, n_sample=1024):
    points = np.array(cloud.points)
    n_sample = min(len(points), n_sample)
    idx_selected_c_inp = np.random.choice(len(points), n_sample, replace=False)
    points_sample = points[idx_selected_c_inp]
    return points_sample

def normalize_mesh_to_cube(mesh, unit_cube_size = 2.0):
    # mesh = o3d.io.read_triangle_mesh(path_mesh)
    # centroid = mesh.get_center()
    # trans: world to norm

    box_aabb = mesh.get_axis_aligned_bounding_box()
    max_size = (box_aabb.max_bound-box_aabb.min_bound).max()
    scale = unit_cube_size / max_size

    if scale==1.0:
        return mesh, {'center': [0,0,0],
                        'scale': 1.0,
                        'trans': np.identity(4)}

    centroid = (box_aabb.max_bound+box_aabb.min_bound)/2.0

    trans_mv = np.identity(4)
    trans_mv[:3, 3] = -centroid
    trans_scale = np.diag([scale, scale, scale, 1.0])
    trans_all = trans_scale@trans_mv
    mesh_norm = mesh.transform(trans_all)

    # if path_mesh_norm is not None:
    #     if should_ensure_pdir_existence:
    #         IOUtils.ensure_pdir_existence(path_mesh_norm)
    #     o3d.io.write_triangle_mesh(path_mesh_norm, mesh_norm)

    dict_norm = {
        'center': centroid.tolist(),
        'scale': scale,
        'trans': trans_all
    }
    return mesh_norm, dict_norm     

def parse_mesh_color_group(mesh_inp):
    vertex_colors = np.array(mesh_inp.vertex_colors)
    lis_color_unique = np.unique(vertex_colors, axis=0)

    lis_mesh = []
    for i in range(len(lis_color_unique)):
        color_i = lis_color_unique[i]
        mask_curr = np.all(np.equal(vertex_colors, color_i), axis=1) 
        n_verts_curr = mask_curr.sum()
        # print(f'Curr verts: {n_verts_curr}/{len(mask_curr)}')
        # matching_row_indices = np.where(mask)[0]

        # mask_remove_curr = np.ones_like(matching_row_indices).astype(bool)
        # mask_remove_curr[matching_row_indices]=False

        mesh_curr = copy(mesh_inp)
        mesh_curr.remove_vertices_by_mask(mask_curr==False)
        lis_mesh.append(mesh_curr)

    return lis_mesh

def transfer_colors_tgt2part(target_mesh, lis_mesh_src, mode_geo='mesh', 
                                check_duplicated_colors=False):  
    # Convert target mesh vertices and vertex colors to numpy arrays  
    target_vertices = np.asarray(target_mesh.vertices)  
    target_colors = np.asarray(target_mesh.vertex_colors)  
  
    # Create a KDTree for nearest-neighbor search  
    cloud = create_pointcloud_from_points(target_vertices)
    kdtree = o3d.geometry.KDTreeFlann(cloud)  
  
    # Iterate through source meshes  
    lis_color = []; lis_color_count = []; mesh_merge_new = None
    for i in range(len(lis_mesh_src)): 
        if mode_geo=='mesh': 
            source_mesh = lis_mesh_src[i]
            source_vertices_sample, _ = sample_points_on_mesh(source_mesh, n_sample=1024, return_points=True)
        elif mode_geo=='pointcloud':  
            source_vertices_sample = np.array(lis_mesh_src[i].points)
        else:
            assert False
          
        # Initialize an array to store the new vertex colors  
        source_colors = np.zeros_like(source_vertices_sample)  
  
        # Find the nearest vertex in the target mesh for each vertex in the source mesh  
        for idx_v, vertex in enumerate(source_vertices_sample):  
            _, idx_nn, _ = kdtree.search_knn_vector_3d(vertex, 1)  
            source_colors[idx_v] = target_colors[idx_nn[0]]  
  
        # Assign the new vertex colors to the source mesh  
        values, lis_count = np.unique(source_colors, axis=0, return_counts=True)
        idx_max = np.argmax(lis_count)
        color_curr = values[idx_max]
        count_curr = lis_count[idx_max]

        lis_mesh_src[i].paint_uniform_color(color_curr)
        lis_color.append(color_curr)
        lis_color_count.append(count_curr)

        if mesh_merge_new is None:
            mesh_merge_new = copy(lis_mesh_src[i])
        else:
            mesh_merge_new += copy(lis_mesh_src[i])

    # TODO: Add linear assignment
    if check_duplicated_colors:
        lis_color = np.array(lis_color)
        values_unique, count_unique = np.unique(lis_color, axis=0, return_counts=True)

        mask_reset = count_unique>1
        lis_values_unique_reset = values_unique[mask_reset]

        for i_reset in range(len(lis_values_unique_reset)):
            color_curr = lis_values_unique_reset[i_reset]

            mask_curr = np.all(np.equal(lis_color, color_curr), axis=1) 
            lis_idx_matching = np.where(mask_curr)[0]
            print(f'Curr parts with duplicated colors: {len(lis_idx_matching)}')
            for j_mesh in range(1, len(lis_idx_matching)):
                idx_reset_mesh = int(lis_idx_matching[j_mesh])
                lis_mesh_src[idx_reset_mesh] = paint_mesh_uniformly(lis_mesh_src[idx_reset_mesh], np.random.choice(256, 1)[0])

        mesh_merge_new = None
        for i in range(len(lis_mesh_src)):
            if mesh_merge_new is None:
                mesh_merge_new = copy(lis_mesh_src[i])
            else:
                mesh_merge_new += copy(lis_mesh_src[i])

    return mesh_merge_new, lis_mesh_src, lis_color

def normalize_points_sphere(pc, mv2origin=True):
    '''
    paras:
            pc: N*3
    '''
    centroid = np.mean(pc, axis=0)
    if mv2origin:
        pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def normalize_points_to_unit_sphere(vertices, mv_back=False):  
    # Compute the mesh vertices as a numpy array        
    # Calculate the mesh centroid  
    centroid = np.mean(vertices, axis=0)  
      
    # Translate the mesh so that the centroid is at the origin  
    vertices -= centroid  
      
    # Calculate the maximum distance from the origin to any vertex  
    max_distance = np.max(np.linalg.norm(vertices, ord=2, axis=1))  
      
    # Scale the mesh vertices so that the maximum distance is 1  
    vertices /= max_distance  

    # vertices *= 0.6

    if mv_back:
        vertices += centroid
      
    # Update the mesh vertices with the normalized positions        
    return vertices 


def get_transformation_rotation(angle_rot, axis='y'):
    '''
    Paras:
        angle_rot: degrees
    '''
    angle_rotate = angle_rot/360.0*(np.pi*2)
    trans_rot_3x3 = R.from_rotvec(angle_rotate * np.array([0, 1, 0])).as_matrix()
    trans_rot = np.identity(4)
    trans_rot[:3, :3] = trans_rot_3x3
    
    return trans_rot

def get_transformation_rotation_x(angle_rot, axis='x'):
    '''
    Paras:
        angle_rot: degrees
    '''
    angle_rotate = angle_rot/360.0*(np.pi*2)
    trans_rot_3x3 = R.from_rotvec(angle_rotate * np.array([1, 0, 0])).as_matrix()
    trans_rot = np.identity(4)
    trans_rot[:3, :3] = trans_rot_3x3
    
    return trans_rot

# def write_points2ply(path_ply, points, pts_type = 'xyz', mode='w'):
#     '''
#     Args:
#         path_ply: path to save file
#         points: N*3
#         pts_type: 'xyz' (default), 'xyz_nxyz', 'xyz_nxyz_rgb'
#     '''

#     num_points = points.shape[0]
#     header_xyz = f'''ply
# format ascii 1.0
# comment VCGLIB generated
# element vertex {num_points}
# property float x
# property float y
# property float z
# end_header
# '''
#     header_xyz_nxyz = f'''ply
# format ascii 1.0
# comment VCGLIB generated
# element vertex {num_points}
# property float x
# property float y
# property float z
# property float nx
# property float ny
# property float nz
# element face 0
# property list uchar int vertex_indices
# end_header
# '''

#     header_xyz_nxyz_rgb = f'''ply
# format ascii 1.0
# comment VCGLIB generated
# element vertex {num_points}
# property float x
# property float y
# property float z
# property float nx
# property float ny
# property float nz
# property uchar red
# property uchar green
# property uchar blue
# property uchar alpha
# element face 0
# property list uchar int vertex_indices
# end_header
# '''

#     fPly = open(path_ply, mode)
#     if pts_type == 'xyz':
#         fPly.write(header_xyz)
#     elif pts_type == 'xyz_nxyz':
#         fPly.write(header_xyz_nxyz)
#     elif pts_type == 'xyz_nxyz_rgb':
#         fPly.write(header_xyz_nxyz_rgb)
#     else:
#         NotImplementedError
#     fPly.write(pd.DataFrame(points).to_string(header=False, index=False))
#     fPly.close()



'''Mesh preprocessing
'''

def build_scene_from_lis_mesh(lis_mesh, remove_textures=False):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_ids = {}
    mesh_all = None
    for i in range(len(lis_mesh)):
        mesh_i = lis_mesh[i]
        if remove_textures:
            mesh_i.textures = []
        mesh_i_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_i)
        mesh_ids[scene.add_triangles(mesh_i_t)] = i

        # if i == 0:
        #     mesh_all = copy(mesh_i)
        # else:
        #     mesh_all += copy(mesh_i) 
    return scene, mesh_ids

def paint_mesh_uniformly(mesh, idx_color=None, color=None):
    if color is not None:
        mesh.paint_uniform_color(color)
        return mesh
    try:
        color = PathConfig.get_color_map()[idx_color]
    except (IndexError, KeyError, TypeError):
        color = np.random.rand(3)*0.7+0.3
    mesh.paint_uniform_color(color)
    return mesh

def get_mesh_max_cluster(mesh, min_faces_cluster=200, mode='remove_small_clusters'):
    triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # only get max cluster
    # mesh_1 = copy(mesh)
    # largest_cluster_idx = cluster_n_triangles.argmax()
    # triangles_to_remove = triangle_clusters != largest_cluster_idx
    # mesh_1.remove_triangles_by_mask(triangles_to_remove)
    # mesh_1.remove_unreferenced_vertices()

    # remove small cluster
    if min_faces_cluster>0:
        mesh_1 = copy(mesh)
        n1 = np.array(mesh_1.vertices).shape[0]

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_faces_cluster
        mesh_1.remove_triangles_by_mask(triangles_to_remove)
        mesh_1.remove_unreferenced_vertices()

        n2 = np.array(mesh_1.vertices).shape[0]
        # print(f'Removed vertex {n1-n2} = {n1} - {n2}')
    else:
        mesh_1 = copy(mesh)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_1.remove_triangles_by_mask(triangles_to_remove)
        mesh_1.remove_unreferenced_vertices()

    return mesh_1


def split_mesh_to_parts(mesh_inp, lis_meshes, min_verts_part=0, should_paint_part=True):
    points_query = np.array(mesh_inp.vertices)
    scene, mesh_ids = build_scene_from_lis_mesh(lis_meshes)

    points_query_tensor = o3d.core.Tensor(points_query, dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(points_query_tensor)
    nn_geometry_ids = ans['geometry_ids'].numpy()

    # save mesh
    faces = np.array(mesh_inp.triangles)
    face_ids = nn_geometry_ids[faces]
    geometry_cls, count_cls = stats.mode(face_ids, axis=1,keepdims=False)
    map_color = PathConfig.get_color_map() if should_paint_part else None
    lis_mesh_split = []
    merge_split = None
    for i in range(len(mesh_ids)):
        # save point cloud 
        # mesh_curr_mask =  (nn_geometry_ids == i)
        # mesh_vertices = points_query[mesh_curr_mask]
        # IOUtils.ensure_dir_existence(f'./test/000')
        # GeoUtils.save_points(f'./test/000/{mesh_ids[i][:-4]}.ply', mesh_vertices)

        # save mesh
        mask_remove_curr = (geometry_cls != i)
        mesh_curr = copy(mesh_inp)
        mesh_curr.remove_triangles_by_mask(mask_remove_curr)
        mesh_curr.remove_unreferenced_vertices()

        if should_paint_part:
            try:
                color = map_color[i]
            except (IndexError, KeyError, TypeError):
                color = np.random.rand(3)*0.7+0.3
            mesh_curr.paint_uniform_color(color)

        if len(mesh_curr.vertices)>min_verts_part:
            lis_mesh_split.append(copy(mesh_curr))
        
            if merge_split is None:
                merge_split = copy(mesh_curr)
            else:
                merge_split += copy(mesh_curr)

    debug_ = False
    if debug_:
        print(f'[{i}] Vertices: {len(merge_split.vertices)}')
        if len(merge_split.vertices)>0:
            # IOUtils.ensure_dir_existence('./test/mesh_group_new')
            # color = np.random.rand(3)*0.7+0.3
            # mesh_curr.paint_uniform_color(color)
            write_triangle_mesh(f'./test/reg4/merge_split.ply', merge_split)

    return lis_mesh_split, merge_split
