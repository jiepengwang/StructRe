# heavily borrowed from Atlas and https://github.com/andyzeng/tsdf-fusion-python

import numpy as np
import open3d as o3d

import torch
# from numba import njit, prange
from skimage import measure
import utils.utils_geometry as GeoUtils
from tqdm import tqdm
import time

import utils.utils_image as ImageUtils
import trimesh, pyrender

from confs.path import TSDF_Mode

if TSDF_Mode == 'cpu':
  FUSION_GPU_MODE = 0
  # print('TSDF with CPU mode')
else:
  try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - initializes the CUDA context
    from pycuda.compiler import SourceModule
    FUSION_GPU_MODE = 1
    print('Running fusion in GPU mode.')
  except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0

import mcubes
def extract_geometry(path_mesh, tsdf_vol, threshold, bbox, resolution = 64, should_paint_mesh=False, vec_occu=None):
    # logging.info('threshold: {}'.format(threshold))
    # u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(-tsdf_vol, -threshold)
    b_max_np = bbox[3:]
    b_min_np = bbox[:3]
    # print(f'Vertices number: {len(vertices)}')
    
    if vec_occu is not None:
      # 2.1 method 1
      # vec_occu = np.concatenate([vec_occu1, vec_occu2], axis=0)
      # vec_occu = torch.from_numpy(vec_occu).cuda()
      
      # 2.2 method 2
      # ratio_upsample = resolution//64
      # vertices_quant = vertices//ratio_upsample * ratio_upsample 
      # vertices_quant = torch.from_numpy(vertices_quant).cuda()


      # vec_occu1 = vec_occu
      # ratio = resolution//64
      # vec_occu2 = vec_occu + np.array([[1,1,1]])*ratio
      

      # vec_occu1 = vec_occu
      # ratio = resolution//64
      # vec_occu2 = vec_occu + np.array([[1,1,1]])*ratio
      
      # vec_occu1 = torch.from_numpy(vec_occu1).cuda()
      # vec_occu2 = torch.from_numpy(vec_occu2).cuda()
      # dist_corners = torch.cdist(vec_occu2.float(), vec_occu1.float(), p=2)
      # dist_corners_min = dist_corners.min(dim=1)[0]
      # mask_valid = (dist_corners_min < 1e-6)
      # n_valid = mask_valid.sum()
      # vec_occu1_valid = vec_occu1[mask_valid]
      # vec_occu1_valid_offset = vec_occu1_valid + torch.ones((1,3)).cuda()
      # vec_occu = torch.cat([vec_occu1_valid, vec_occu1_valid_offset], dim=0)
      
      # 2.3 method 3
      ratio_upsample = resolution//64
      vertices_quant = np.round(vertices/ratio_upsample, decimals=0) * ratio_upsample 
      vertices_quant = torch.from_numpy(vertices_quant).cuda()
      vec_occu = torch.from_numpy(vec_occu).cuda()
      
      vertices_quant2 = torch.from_numpy( (vertices//ratio_upsample) * ratio_upsample ).cuda() 
      # vec_occu2 = vec_occu + torch.ones((1,3)).cuda()* ratio_upsample 
      # vec_occu2 = torch.cat([vec_occu, vec_occu2], dim=0)

      # 3. get valid vertices
      lis_mask = []
      lis_n_remove = []
      for vectices_quant_i in [vertices_quant, vertices_quant2]:
        dist = torch.cdist(vectices_quant_i.float(), vec_occu.float(), p=2)
        dist_min = dist.min(dim=1)[0]
        mask_remove = (dist_min > 0).cpu().numpy()
        n_remove = mask_remove.sum()
        
        lis_mask.append(mask_remove)
        lis_n_remove.append(n_remove)
      mask_remove = np.logical_or(lis_mask[0], lis_mask[1])
      n_remove = mask_remove.sum()
      
      # print('done')

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    # mesh = trimesh.Trimesh(vertices, triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices) 
    mesh.triangles = o3d.utility.Vector3iVector(triangles) 
    
    if vec_occu is not None:
        mesh.remove_vertices_by_mask(mask_remove)

    if should_paint_mesh:
      color = np.random.rand(3)*0.7+0.3
      # print(color.shape)
      # print(color)
      mesh.paint_uniform_color(color)

    if path_mesh is not None:
      GeoUtils.write_triangle_mesh(path_mesh, mesh)
      # mesh1.triangle_normals = o3d.utility.Vector3dVector(
      #     np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])

      # mesh.export(path_mesh)
    return mesh

class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size, use_gpu=True):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    # self._trunc_margin = 1000 * self._voxel_size  # truncation on SDF:  default 5'  100, a large value for no truncation
    self._trunc_margin = 0.1  # truncation on SDF:  default 5'  100, a large value for no truncation
    self._color_const = 256 * 256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

    # print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
    #   self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
    #   self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    # )

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

    self.gpu_mode = use_gpu and FUSION_GPU_MODE

    # Copy voxel volumes to GPU
    if self.gpu_mode:
      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)

      # Cuda kernel function (C++)
      self._cuda_src_mod = SourceModule("""
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }""")

      self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

      # Determine block/grid size on GPU
      gpu_dev = cuda.Device(0)
      self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
      self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))

    else:
      # Get voxel grid coordinates
      xv, yv, zv = np.meshgrid(
        range(self._vol_dim[0]),
        range(self._vol_dim[1]),
        range(self._vol_dim[2]),
        indexing='ij'
      )
      self.vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).astype(int).T

  @staticmethod
  # @njit(parallel=True)
  def vox2world(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in prange(vox_coords.shape[0]):
      for j in range(3):
        cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    return cam_pts

  @staticmethod
  # @njit(parallel=True)
  def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix

  @staticmethod
  # @njit(parallel=True)
  def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
    """Integrate the TSDF volume.
    """
    tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    w_new = np.empty_like(w_old, dtype=np.float32)
    for i in prange(len(tsdf_vol)):
      w_new[i] = w_old[i] + obs_weight
      tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
    return tsdf_vol_int, w_new

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """
    im_h, im_w = depth_im.shape

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

    if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
      for gpu_loop_idx in range(self._n_gpu_loops):
        self._cuda_integrate(self._tsdf_vol_gpu,
                            self._weight_vol_gpu,
                            self._color_vol_gpu,
                            cuda.InOut(self._vol_dim.astype(np.float32)),
                            cuda.InOut(self._vol_origin.astype(np.float32)),
                            cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                            cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                            cuda.InOut(np.asarray([
                              gpu_loop_idx,
                              self._voxel_size,
                              im_h,
                              im_w,
                              self._trunc_margin,
                              obs_weight
                            ], np.float32)),
                            cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                            cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                            block=(self._max_gpu_threads_per_block,1,1),
                            grid=(
                              int(self._max_gpu_grid_dim[0]),
                              int(self._max_gpu_grid_dim[1]),
                              int(self._max_gpu_grid_dim[2]),
                            )
        )
    else:  # CPU mode: integrate voxel volume (vectorized implementation)
      # Convert voxel grid coordinates to pixel coordinates
      cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
      cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix(cam_pts, cam_intr)
      pix_x, pix_y = pix[:, 0], pix[:, 1]

      # Eliminate pixels outside view frustum
      valid_pix = np.logical_and(pix_x >= 0,
                  np.logical_and(pix_x < im_w,
                  np.logical_and(pix_y >= 0,
                  np.logical_and(pix_y < im_h,
                  pix_z > 0))))
      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

      # Integrate TSDF
      depth_diff = depth_val - pix_z
      valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      dist = np.minimum(1, depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords[valid_pts, 0]
      valid_vox_y = self.vox_coords[valid_pts, 1]
      valid_vox_z = self.vox_coords[valid_pts, 2]
      w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts]
      tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
      self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      # Integrate color
      old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = np.floor(old_color / self._color_const)
      old_g = np.floor((old_color-old_b*self._color_const)/256)
      old_r = old_color - old_b*self._color_const - old_g*256
      new_color = color_im[pix_y[valid_pts],pix_x[valid_pts]]
      new_b = np.floor(new_color / self._color_const)
      new_g = np.floor((new_color - new_b*self._color_const) /256)
      new_r = new_color - new_b*self._color_const - new_g*256
      new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
      new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
      new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
      self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

  def get_volume(self):
    if self.gpu_mode:
      cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
      cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    return self._tsdf_vol_cpu, self._color_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def get_mesh(self):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors, tsdf_vol


  def get_mesh2(self, path_mesh):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts, faces = mcubes.marching_cubes(-tsdf_vol, 0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(path_mesh)
    return tsdf_vol

def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def fuse_depthmaps_to_tsdf(path_mesh_save, path_intrin, dir_poses, 
                                  arr_depths, arr_colors,
                                  resolution = 0.02, reso_level = 1.0, vec_stem_poses = None, 
                                  path_tsdf_grid = None, box_size_half = None, 
                                  verbose=False):
    # print("Estimating voxel volume bounds...")
    
    if arr_colors is None:
        arr_colors = np.ones(arr_depths.shape+tuple([3]))
    
    vol_bnds = np.zeros((3,2))
    
    cam_intr = GeoUtils.read_cam_matrix(path_intrin)
    if reso_level>1:
          cam_intr = GeoUtils.resize_cam_intrin(cam_intr, resolution_level=reso_level)
    poses = GeoUtils.read_poses(dir_poses)
    if vec_stem_poses is not None:
      poses, _ = GeoUtils.read_poses(dir_poses, lis_stem=vec_stem_poses)
    n_imgs = len(poses)
    assert (n_imgs > 0)
    # print('Init volume')
    for i in range(n_imgs):
      # Read depth image and camera pose
      # depth_im = cv2.imread(f"{dir_data}/frame-{i:06d}.depth.png",-1).astype(float)
      # depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
      depth_im = arr_depths[i].clip(0, 5)
      # depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
      cam_pose = poses[i]

      # Compute camera view frustum and extend convex hull
      view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
      vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
      vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    vol_bnds = vol_bnds* 2
    vol_bnds = vol_bnds - vol_bnds%resolution
    # print(f"Initializing voxel volume...")
    # print(f'Volume box: Min, {vol_bnds[:,0]}; Max, {vol_bnds[:,1]}.')
    # print(f"Volume size: {vol_bnds[:,1] - vol_bnds[:,0]}.")

    tsdf_vol = TSDFVolume(vol_bnds, voxel_size=resolution)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
      # print("Fusing frame %d/%d"%(i+1, n_imgs))

      # Read RGB-D image and camera pose
      color_image = arr_colors[i]  #BGR to RGB
      depth_im = arr_depths[i].clip(0, 5)
      # depth_im /= 1000.
      # depth_im[depth_im == 65.535] = 0
      cam_pose = poses[i]

      # Integrate observation into voxel volume (assume color aligned with depth)
      tsdf_vol.integrate(color_image, depth_im, cam_intr[:3,:3], cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    # if verbose:
    #   print("Average FPS: {:.2f}".format(fps))

    #   # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    #   print(f"Saving mesh to {path_mesh_save}...")
    tsdf_vol_full_grid = tsdf_vol.get_mesh2(path_mesh_save)
    # verts, faces, norms, colors, tsdf_vol_full_grid = tsdf_vol.get_mesh()
    # meshwrite(path_mesh_save, verts, faces, norms, colors)
    
    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    # print("Saving point cloud to pc.ply...")
    # point_cloud = tsdf_vol.get_point_cloud()
    # pcwrite(IOUtils.add_file_name_suffix(path_mesh_save, '_pc'), point_cloud)
    tsdf_grid_cube = None
    if path_tsdf_grid:
          # extract tsdf gird
          grid_vert_min, grid_vert_max = -box_size_half* np.ones(3), box_size_half*np.ones(3)
          idx_min = ((grid_vert_min - vol_bnds[:,0]) // resolution).astype(np.int32)
          # idx_max = ((grid_vert_max - vol_bnds[:,0]) // resolution).astype(np.int32)
          # diff = idx_max-idx_min
          # idx_max = idx_max - 1
          idx_max = idx_min + 64
          tsdf_grid_cube = tsdf_vol_full_grid[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
          # extract_geometry(tsdf_grid_cube, 0, './test/mcubes_mesh.ply')
          # print(tsdf_grid_cube.min(), tsdf_grid_cube.max())
          np.save(path_tsdf_grid, tsdf_grid_cube)
          # print('done')
          
          # visualize grid
          b_visualize_grid = False
          if b_visualize_grid:
            num_sample_axis = int(2.4 // resolution)
            # num_sample_axis = 3
            # X = np.linspace(-1, 1, 3)
            # Y = np.linspace(-1, 5, 7)
            # Z = np.linspace(-1, 3, 5
            X = np.linspace(-box_size_half, box_size_half, num_sample_axis)
            Y = np.linspace(-box_size_half, box_size_half, num_sample_axis)
            Z = np.linspace(-box_size_half, box_size_half, num_sample_axis)
            
            idx_back = (tsdf_grid_cube < -0).nonzero()
            pts_back = np.stack(idx_back, axis=-1)*resolution + grid_vert_min
            sdf_back = tsdf_grid_cube[idx_back[0], idx_back[1], idx_back[2]]
            colors_back = ImageUtils.convert_gray_to_cmap((-sdf_back).reshape(-1,1))
            GeoUtils.save_points('./test/pts_back.ply', pts_back, colors_back.squeeze())
            print(len(idx_back))
            
            idx_front = np.logical_and(tsdf_grid_cube > 0, tsdf_grid_cube <1).nonzero()
            pts_front = np.stack(idx_front, axis=-1)*resolution + grid_vert_min
            sdf_front = tsdf_grid_cube[idx_front[0], idx_front[1], idx_front[2]]
            colors_front = ImageUtils.convert_gray_to_cmap(sdf_front.reshape(-1,1))
            GeoUtils.save_points('./test/pts_front.ply', pts_front, colors_front.squeeze())
            print(len(idx_back))
            
             
            colors =ImageUtils.convert_gray_to_cmap(tsdf_grid_cube.reshape(-1,1)+1, revert = False)
            
            
            # 3,5,3
            yy, xx, zz = np.meshgrid(Y, X, Z)
            # print(xx, '\n\n')
            # print(yy, '\n\n')
            # print(zz, '\n\n')
            pts_grid = np.stack([xx, yy, zz], axis=-1)
            GeoUtils.save_points('./test/grid.ply', pts_grid.reshape(-1,3), colors=colors.squeeze())
            GeoUtils.save_points('./test/grid2.ply', pts_grid[:,:,0].reshape(-1,3), colors=colors.squeeze())
            print('Done')
    
    dict_tsdf_vol = {
              'full_grid': tsdf_vol_full_grid,
              'vol_bnds': vol_bnds
            }
          
    return dict_tsdf_vol

def get_sdf_cube_params(n_sample, box_size_half = 1.15, verbose = False):
    # box_max = np.random.normal(0, 1, size=(3,1))*0.05 + 1.15 # 1.15~1.25
    # box_max = box_max.clip(1.05, 1.25)

    # box_min = -(np.random.normal(0, 1, size=(3,1))*0.05 + 1.15) # 1.15~1.25
    # box_min = box_min.clip(-1.25, -1.05)

    # width = (box_max-box_min)
    # # print(width.max())
    # box_max = box_min + width.mean()
    # params = np.concatenate([box_min, box_max]).squeeze()
    params = np.array([-box_size_half]*3 + [box_size_half]*3)
    # if verbose:
    #   print('BBox paras: ', params)
  
    X = np.linspace(params[0], params[3], num=n_sample).astype(np.float32)
    Y = np.linspace(params[1], params[4], num=n_sample).astype(np.float32)
    Z = np.linspace(params[2], params[5], num=n_sample).astype(np.float32)
    yy, xx, zz = np.meshgrid(Y, X, Z)
    pts_grid = np.stack([xx, yy, zz], axis=-1).reshape(-1,3)
    # if verbose:
    #   print(pts_grid[0], pts_grid[-1])

    return pts_grid, params

def extract_tsdf_grid(dict_tsdf_vol, reso_vox, sdf_paras):
    tsdf_vol_full_grid = dict_tsdf_vol['full_grid']
    vol_bnds = dict_tsdf_vol['vol_bnds']  


    # extract tsdf gird
    sdf_paras = sdf_paras.squeeze()
    grid_vert_min = sdf_paras[:3].squeeze()
    resolution = (sdf_paras[3:] - sdf_paras[:3]) / (reso_vox-1)
    resolution = resolution[0]
    idx_min = ((grid_vert_min - vol_bnds[:,0]) // resolution).astype(np.int32)
    # update sdf paras
    # print(sdf_paras)
    sdf_paras[:3] = vol_bnds[:,0] + resolution*idx_min
    sdf_paras[3:] =  sdf_paras[:3] + resolution* (reso_vox-1)
    # print(sdf_paras)

    # idx_max = ((grid_vert_max - vol_bnds[:,0]) // resolution).astype(np.int32)
    # diff = idx_max-idx_min
    # idx_max = idx_max - 1
    idx_max = idx_min + 64
    tsdf_grid_cube = tsdf_vol_full_grid[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
    # extract_geometry(tsdf_grid_cube, 0, './test/mcubes_mesh.ply')
    # print(tsdf_grid_cube.min(), tsdf_grid_cube.max())
    # np.save(path_tsdf_grid, tsdf_grid_cube)
    # print('done')
    
    # visualize grid
    b_visualize_grid = False
    if b_visualize_grid:
        num_sample_axis = int(2.4 // resolution)
        # num_sample_axis = 3
        # X = np.linspace(-1, 1, 3)
        # Y = np.linspace(-1, 5, 7)
        # Z = np.linspace(-1, 3, 5
        X = np.linspace(-box_size_half, box_size_half, num_sample_axis)
        Y = np.linspace(-box_size_half, box_size_half, num_sample_axis)
        Z = np.linspace(-box_size_half, box_size_half, num_sample_axis)
        
        idx_back = (tsdf_grid_cube < -0).nonzero()
        pts_back = np.stack(idx_back, axis=-1)*resolution + grid_vert_min
        sdf_back = tsdf_grid_cube[idx_back[0], idx_back[1], idx_back[2]]
        colors_back = ImageUtils.convert_gray_to_cmap((-sdf_back).reshape(-1,1))
        GeoUtils.save_points('./test/pts_back.ply', pts_back, colors_back.squeeze())
        print(len(idx_back))
        
        idx_front = np.logical_and(tsdf_grid_cube > 0, tsdf_grid_cube <1).nonzero()
        pts_front = np.stack(idx_front, axis=-1)*resolution + grid_vert_min
        sdf_front = tsdf_grid_cube[idx_front[0], idx_front[1], idx_front[2]]
        colors_front = ImageUtils.convert_gray_to_cmap(sdf_front.reshape(-1,1))
        GeoUtils.save_points('./test/pts_front.ply', pts_front, colors_front.squeeze())
        print(len(idx_back))
        
          
        colors =ImageUtils.convert_gray_to_cmap(tsdf_grid_cube.reshape(-1,1)+1, revert = False)
        
        
        # 3,5,3
        yy, xx, zz = np.meshgrid(Y, X, Z)
        # print(xx, '\n\n')
        # print(yy, '\n\n')
        # print(zz, '\n\n')
        pts_grid = np.stack([xx, yy, zz], axis=-1)
        GeoUtils.save_points('./test/grid.ply', pts_grid.reshape(-1,3), colors=colors.squeeze())
        GeoUtils.save_points('./test/grid2.ply', pts_grid[:,:,0].reshape(-1,3), colors=colors.squeeze())
        print('Done')
        
    return tsdf_grid_cube, sdf_paras
  
class Renderer():
    """Borrowed from Atlas
    OpenGL mesh renderer 
    
    Used to render depthmaps from a mesh for 2d evaluation
    """
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        #self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)#, self.render_flags) 

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R =  np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose@axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
    

def render_depthmaps_pyrender(path_mesh, path_intrin = None, dir_poses= None, dir_rgb = None, 
                intrin = None, poses = None, 
                target_img_size = (640, 480), reso_level = 1.0, 
                path_mask_npz = None,
                dir_verbose = None,
                verbose = False):       
    # gt depth data loader
    assert reso_level == 1.0
    width, height = target_img_size
    if path_mask_npz:
        target_2dmask_mesh = np.load(path_mask_npz)['arr_0']
    
    # mesh renderer
    mesh = trimesh.load(path_mesh, process=False, force = 'mesh')
    
    
    if intrin is None:
      intrin = GeoUtils.read_cam_matrix(path_intrin)

    if poses is None:
      poses = GeoUtils.read_poses(dir_poses)

    num_images = len(poses)
    depths_all = []
    
    # pyrender
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    if verbose:
      print('Render depth maps...')
    iter_imgs = tqdm(range(num_images)) if verbose else range(num_images)
    for i in iter_imgs:
        pose_curr = poses[i]
        _, depth_pred = renderer(height, width, intrin, pose_curr, mesh_opengl)
        if path_mask_npz is not None:
            mask_curr = target_2dmask_mesh[i]
            depth_pred = depth_pred*mask_curr.astype(np.float)
            
        depths_all.append(depth_pred)
        if verbose:
          print(f'Depth map (max, min): {depth_pred.max(), depth_pred.min()}')
    depths_all = np.array(depths_all)

    if dir_verbose is not None:
        num_sample = 100
        if dir_rgb:
          colors, _ = ImageUtils.read_images(dir_rgb, use_rgb_mode=True)
        else:
          colors = np.ones(depths_all.shape+tuple([3]))
        
        for idx in range(num_images):
          depth_curr = depths_all[idx]
          depth_img = (depth_curr / depth_curr.max()) * 255
          ImageUtils.write_image(f'{dir_verbose}/deptmap_{idx}.png', depth_img)

              
        if len(depths_all)> num_sample:
            # logging.info("Sample 200 depth maps to get merged points...")
            idx_imgs = np.random.randint(low=0, high=depths_all.shape[0], size=num_sample)
            depthmaps_fuse = depths_all[idx_imgs]
            points = GeoUtils.fuse_depthmaps_to_points_world(depthmaps_fuse, intrin, np.linalg.inv(poses[idx_imgs]))

            arr_imgs_temp = colors[idx_imgs].reshape(-1,3)
        else:
            points = GeoUtils.fuse_depthmaps_to_points_world(depths_all, intrin, np.linalg.inv(poses))
            arr_imgs_temp = colors.reshape(-1,3)
        idx_pts = np.random.randint(low=0, high=points.shape[0], size=int(1e6))
        pts_sample = points[idx_pts]
        colors_sample = arr_imgs_temp[idx_pts]
        GeoUtils.save_points(f'{dir_verbose}/fused_cloud.ply', pts_sample, colors_sample)
        
        pts_cams = GeoUtils.get_camera_origins(np.linalg.inv(poses))
        GeoUtils.save_points(f'{dir_verbose}/cameras.ply', pts_cams)

    return depths_all, intrin, poses
