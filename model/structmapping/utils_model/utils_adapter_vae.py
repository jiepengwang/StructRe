"""Mesh sampling adapter used by StructMapping geometry inference."""

from copy import copy

import numpy as np
import open3d as o3d

from confs.path import get_viewpoints_sphere

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils


def get_visible_verts_mesh(mesh, cameras):
    mesh_visible = copy(mesh)
    vertices = np.asarray(mesh_visible.vertices)
    visible = GeoUtils.get_visible_points_cameras(vertices, cameras)
    mesh_visible.remove_vertices_by_mask(~visible)
    return mesh_visible


def get_cameras_random_sphere():
    camera_center = 4 * GeoUtils.sample_sphere(
        1,
        use_half_sphere_y=True,
    )
    return [camera_center[0]]


def get_cameras_random_bbox_corner(half_cube_side=3, lis_idx_cam=None):
    radius_view = np.sqrt(half_cube_side**2 * 3)
    bbox_centers = np.asarray(
        [
            [-half_cube_side, half_cube_side, -half_cube_side],
            [half_cube_side, half_cube_side, -half_cube_side],
            [-half_cube_side, half_cube_side, half_cube_side],
            [half_cube_side, half_cube_side, half_cube_side],
            [0, 0, radius_view],
            [0, 0, -radius_view],
            [0, radius_view, 0],
            [0, -radius_view, 0],
            [radius_view, 0, 0],
            [-radius_view, 0, 0],
        ]
    )
    if lis_idx_cam is None:
        n_sample = int(np.random.choice(4)) + 1
        lis_idx_cam = np.random.choice(4, n_sample, replace=False).tolist()
    return bbox_centers[lis_idx_cam]


def get_cameras_random_fps(n_views):
    viewpoints = copy(get_viewpoints_sphere())
    np.random.shuffle(viewpoints)

    if n_views is None:
        n_views = np.random.choice(np.asarray([3, 4, 5]), 1)[0]
    print(f'Current sampled n_views: {n_views}')

    viewpoint_cloud = GeoUtils.create_pointcloud_from_points(viewpoints)
    sampled_cloud = viewpoint_cloud.farthest_point_down_sample(n_views)
    return np.asarray(sampled_cloud.points)


def get_cameras_random_sample(mode_camera_sample):
    if mode_camera_sample == 'random_sphere':
        return get_cameras_random_sphere()
    if mode_camera_sample == 'random_bbox_corner':
        return get_cameras_random_bbox_corner()
    if mode_camera_sample.startswith('partial_'):
        camera_indices = [int(index) for index in mode_camera_sample.split('_', 1)[1]]
        return get_cameras_random_bbox_corner(lis_idx_cam=camera_indices)
    if mode_camera_sample.startswith('fps'):
        n_views = int(mode_camera_sample[-1]) if len(mode_camera_sample) == 4 else None
        return get_cameras_random_fps(n_views)
    raise ValueError(f'Unknown camera sampling mode: {mode_camera_sample}')


def sample_points_on_mesh(
    path_mesh=None,
    mesh=None,
    n_sample=65536,
    dict_conf=None,
):
    config = dict_conf or {}
    mesh = config.get('mesh_inp', mesh)
    if mesh is None:
        mesh = GeoUtils.read_triangle_mesh(path_mesh)

    _, sampled = GeoUtils.sample_points_on_mesh(
        mesh,
        n_sample,
        return_dict_data=True,
    )
    pointcloud = sampled['pointcloud']
    normal = sampled['normal']

    mode_geo = config.get('mode_geo', 'complete')
    cameras = None
    if mode_geo != 'complete':
        cameras = get_cameras_random_sample(mode_geo)
        visible = GeoUtils.get_visible_points_cameras(pointcloud, cameras)
        pointcloud = pointcloud[visible]
        normal = normal[visible]

        path_partial = config.get('path_partial')
        if path_partial:
            mesh_visible = get_visible_verts_mesh(mesh, cameras)
            IOUtils.ensure_pdir_existence(path_partial)
            GeoUtils.write_triangle_mesh(path_partial, mesh_visible)

        path_pointcloud = config.get('path_output_pointcloud')
        if path_pointcloud:
            IOUtils.ensure_pdir_existence(path_pointcloud)
            IOUtils.write_dict_to_h5(
                path_pointcloud,
                {
                    'pointcloud': pointcloud,
                    'normal': normal,
                    'lis_cam_sample': cameras,
                },
            )

    return pointcloud, normal, cameras


def fix_normal_by_view_direction(pointcloud, normal_pred, cameras):
    mask_flip_merge = np.zeros(len(pointcloud), dtype=bool)
    for camera in cameras:
        camera_center = np.asarray(camera)
        _, visible = GeoUtils.hidden_point_removal(
            copy(pointcloud),
            camera_center[None, ...],
        )
        direction_to_camera = camera_center - pointcloud
        mask_flip = np.sum(direction_to_camera * normal_pred, axis=1) < 0
        mask_flip_merge[visible] = mask_flip[visible]

    normal_pred[mask_flip_merge] = -normal_pred[mask_flip_merge]
    return normal_pred


def fix_input_normal(
    pointcloud,
    normal,
    inp_mode_normal,
    lis_cam_sample=None,
):
    if inp_mode_normal in ('gt_normal', None):
        return pointcloud, normal
    if inp_mode_normal == 'no_normal':
        return pointcloud, np.zeros_like(normal)
    if inp_mode_normal == 'estimated_normal':
        point_cloud = GeoUtils.create_pointcloud_from_points(pointcloud)
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,
                max_nn=16,
            )
        )
        normal = fix_normal_by_view_direction(
            pointcloud,
            copy(np.asarray(point_cloud.normals)),
            lis_cam_sample,
        )
        return pointcloud, normal
    raise ValueError(f'Unknown inp_mode_normal: {inp_mode_normal}')


def quantize_pointcloud2grid_mink(pointcloud):
    bbox_size_half = 1.15
    tsdf_resolution = 256
    pointcloud = (
        (pointcloud + bbox_size_half)
        * (tsdf_resolution - 1)
        / (bbox_size_half * 2)
    )
    pointcloud_grid = np.round(pointcloud).astype(int)
    pointcloud_offset = pointcloud - pointcloud_grid

    _, unique_indices = np.unique(
        pointcloud_grid,
        axis=0,
        return_index=True,
    )
    mask_valid_pointcloud = np.zeros(len(pointcloud))
    mask_valid_pointcloud[unique_indices] = 1
    return pointcloud, pointcloud_offset, mask_valid_pointcloud
