"""Stage 1 readers for prepared sparse geometry artifacts."""

from copy import copy

import numpy as np

import utils.utils_io as IOUtils


def load_tsdf_data_h5_to_np(
    sdf_h5_file,
    tsdf_resolution,
    bbox_size_half=1.15,
    ratio_tsdf_grid=1.0,
    should_load_tsdf=True,
    scale_cube=1.0,
):
    """Load one Stage 1 H5 artifact and normalize points to voxel indices."""
    data = IOUtils.read_h5_to_dict(sdf_h5_file)

    if should_load_tsdf:
        if "mask_valid" in data:
            truncation = data["thres_trunc"][0]
            valid_mask = data["mask_valid"]
            sdf = np.full(valid_mask.shape, truncation, dtype=np.float32)
            sdf[valid_mask] = data["tsdf_valid"]
        else:
            sdf = data["tsdf"]
        sdf = sdf.reshape(
            tsdf_resolution,
            tsdf_resolution,
            tsdf_resolution,
        )

    pointcloud = copy(data["pointcloud"])
    if scale_cube != 1.0:
        pointcloud *= scale_cube
        bbox_size_half *= scale_cube
        tsdf_resolution *= scale_cube

    pointcloud = _normalize_points(
        pointcloud,
        bbox_size_half,
        tsdf_resolution,
        ratio_tsdf_grid,
    )
    rounded_pointcloud = np.round(pointcloud).astype(int)

    _, unique_indices = np.unique(
        rounded_pointcloud, axis=0, return_index=True
    )
    valid_point_mask = np.zeros(len(pointcloud), dtype=bool)
    valid_point_mask[unique_indices] = True

    result = {
        "path": sdf_h5_file,
        "pointcloud": pointcloud,
        "normal": data.get("normal"),
        "mask_valid_pointcloud": valid_point_mask,
    }
    if should_load_tsdf:
        result["sdf"] = sdf
    return result


def _normalize_points(
    points,
    bbox_size_half,
    tsdf_resolution,
    ratio_tsdf_grid,
):
    return (
        (points + bbox_size_half)
        * (tsdf_resolution - 1)
        * ratio_tsdf_grid
        / (bbox_size_half * 2)
    )
