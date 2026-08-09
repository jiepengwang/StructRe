import os
import re
from functools import lru_cache

import numpy as np


DIR_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_path(value):
    return os.path.abspath(os.path.expanduser(value))


def _env_path(name, default):
    return _normalize_path(os.environ.get(name, default))


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'on'}


def _env_artifact_id(name, default):
    value = os.environ.get(name, default)
    if (
        not value
        or value in {'.', '..'}
        or os.path.isabs(value)
        or '/' in value
        or '\\' in value
    ):
        raise ValueError(
            f'{name} must be a non-empty directory name, got {value!r}'
        )
    return value


def _default_workspace_root():
    repo_parent = os.path.dirname(DIR_REPO)
    if os.path.basename(repo_parent) == 'code':
        return os.path.dirname(repo_parent)
    return repo_parent


def _fibonacci_sphere(n_points=256, radius=4.0):
    indices = np.arange(n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    return radius * np.column_stack((
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ))

scale_cube_side = 1.0  # to scale cube side

TSDF_Resolution = 256 # or 256
TSDF_Resolution_Part = TSDF_Resolution
TSDF_Resolution_Partial = TSDF_Resolution # or 256

TSDF_Resolution_Sparse_VAE_Encoder = 256
TSDF_Resolution_Minkowski_Grid = TSDF_Resolution_Sparse_VAE_Encoder

TSDF_Truncation_Part = 0.02
Thres_MC = 0.005  # for tsdf

BBox_Size = 2.3
BBox_Size_Half = 1.15
BBox_AABB = np.array([-1.15]*3 + [1.15]*3)
BBox_AABB_Reso256 = BBox_AABB + np.array([0,0,0] + [2.3/((128-1)*2)]*3)

Thres_Min_Vert_Part_Shapenet = 200

# Artifact and checkpoint identifiers
exp_name_sparse_vae = _env_artifact_id(
    'STRUCTRE_STAGE1_ARTIFACT_ID',
    '20230321_1000_sparse_vae_reso256_b2_klreg_smallEncoder_trunc0_02',
)
ckpt_sparse_vae = 'ckpt_0048.pt'

exp_name_part_vae = _env_artifact_id(
    'STRUCTRE_STAGE2_ARTIFACT_ID',
    '20230403_1730_struct_allData_kl_b12_noPadding_useAttenInp',
)
ckpt_part_vae = 'ckpt_0046.pt'

# Prepared artifacts
Folder_Mesh_Aug = 'geometry/mesh_hot_v2'
Folder_Mesh_Aug_Partial = 'geometry/mesh_partial'
Folder_SDF_Augmented_Shape = 'geometry/pointcloud_h5'
Folder_SDF_Part_Sparse_VAE = 'geometry/pointcloud_part_h5_aug_scale'
Folder_SDF_Partial_Sparse_VAE = 'geometry/pointcloud_partial_h5_wn'
Folder_Code_Sparse_VAE = f'code/code_sparse_vae/{exp_name_sparse_vae}'
Folder_Code_Struct = f'code/code_struct/{exp_name_part_vae}'
Folder_Name_Hierarchy_Sparse_VAE = f'hierarchy/hierarchy_pairs2/{exp_name_part_vae}'

TSDF_Mode = 'cpu'

# Storage config
DIR_DATA = _env_path('STRUCTRE_DATA_ROOT', os.path.join(DIR_REPO, 'data'))
DIR_TEAMDRIVE = _env_path('STRUCTRE_WORKSPACE_ROOT', _default_workspace_root())
DIR_EXPS = _env_path('STRUCTRE_EXPERIMENT_ROOT', os.path.join(DIR_REPO, 'exps'))
is_server = _env_flag('STRUCTRE_IS_SERVER')

Path_CKPT_Sparse_VAE = _env_path(
    'STRUCTRE_STAGE1_CHECKPOINT',
    os.path.join(
        DIR_EXPS, 'sparse_vae', exp_name_sparse_vae,
        'checkpoints', ckpt_sparse_vae,
    ),
)
PATH_CKPT_PART_VAE = _env_path(
    'STRUCTRE_STAGE2_CHECKPOINT',
    os.path.join(
        DIR_EXPS, 'part_vae', exp_name_part_vae,
        'checkpoints', ckpt_part_vae,
    ),
)

exp_name_mapping = '20230505_2300_mapping_allScaledData_pointerEncMem_useDirToken_iterDecoding_useAtenInp'
ckpt_mapping = 'ckpt_0005_00700000.pt'
Path_CKPT_Mapping = _env_path(
    'STRUCTRE_STAGE3_CHECKPOINT',
    os.path.join(
        DIR_EXPS, 'structmapping', exp_name_mapping,
        'checkpoints', ckpt_mapping,
    ),
)


def resolve_stage_checkpoint(args, stage):
    checkpoint_by_stage = {
        1: ('path_ckpt_stage1', Path_CKPT_Sparse_VAE),
        2: ('path_ckpt_stage2', PATH_CKPT_PART_VAE),
        3: ('path_ckpt_stage3', Path_CKPT_Mapping),
    }
    if stage not in checkpoint_by_stage:
        raise ValueError(f'Invalid StructRe stage: {stage}')

    arg_name, default = checkpoint_by_stage[stage]
    override = getattr(args, arg_name, None) if args is not None else None
    return _normalize_path(override) if override else default


def resolve_checkpoint_path(path, checkpoint_dir=None):
    if path is None:
        return None

    path = os.path.expanduser(path)
    if os.path.isabs(path) or os.path.dirname(path):
        return os.path.abspath(path)
    if checkpoint_dir is not None:
        return os.path.abspath(os.path.join(checkpoint_dir, path))
    return os.path.abspath(path)


DIR_DATA_SHAPENET_OURS = _env_path(
    'STRUCTRE_SHAPENET_PROCESSED_ROOT',
    os.path.join(DIR_DATA, 'shapenet_ours_v2'),
)
DIR_ROOT_PARTNET = _env_path(
    'STRUCTRE_PARTNET_ROOT',
    os.path.join(DIR_DATA, 'partnet', 'data_hierarchy_v3'),
)

# color map
PATH_COLOR_MAP = _env_path(
    'STRUCTRE_COLOR_MAP',
    os.path.join(DIR_ROOT_PARTNET, 'geometry', 'color_map.npy'),
)
Map_Color = np.random.default_rng(seed=0).random((1000, 3)) * 0.7 + 0.3


@lru_cache(maxsize=None)
def get_color_map(path=None):
    path = _normalize_path(path or PATH_COLOR_MAP)
    if os.path.isfile(path):
        return np.load(path)
    return Map_Color

# 2. Data process config
Categorys_PartNet_Upper = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 
                        'Door', 'Earphone', 'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 
                        'Microwave', 'Mug', 'Refrigerator', 'Scissors', 'StorageFurniture', 'Table', 
                        'TrashCan', 'Vase']
Categorys_PartNet = [elem.lower() for elem in Categorys_PartNet_Upper]


# 4. Model config
Padding_Value_Seq = -2 # padding sequence with various lengthdd
Padding_EOS_Struct = -3  # Padding value of EOS to mark end
Padding_Value_Flag_Level=-1


def parse_shapenet_cats(lis_cat_skip=['cap', 'microwaves', 'cabinet', 'trash_bin', 'flowerpot']):
    '''
    paras:
        lis_cat_skip: cats in ShapeNet, same category but different names in partnet and shapenet
    '''
    info_cats = '''04379243 table 8443 03593526 jar 597 04225987 skateboard 152
02958343 car 7497 02876657 bottle 498 04460130 tower 133
03001627 chair 6778 02871439 bookshelf 466 02942699 camera 113
02691156 airplane 4045 03642806 laptop 460 02801938 basket 113
04256520 sofa 3173 03624134 knife 424 02946921 can 108
04090263 rifle 2373 04468005 train 389 03938244 pillow 96
03636649 lamp 2318 02747177 trash_bin 343 03710193 mailbox 94
04530566 watercraft 1939 03790512 motorbike 337 03207941 dishwasher 93
02828884 bench 1816 03948459 pistol 307 04099429 rocket 85
03691459 loudspeaker 1618 03337140 file_cabinet 298 02773838 bag 83
02933112 cabinet 1572 02818832 bed 254 02843684 birdhouse 73
03211117 display 1095 03928116 piano 239 03261776 earphone 73
04401088 telephone 1052 04330267 stove 218 03759954 microphone 67
02924116 bus 939 03797390 mug 214 04074963 remote 67
02808440 bathtub 857 02880940 bowl 186 03085013 keyboard 65
03467517 guitar 797 04554684 washer 169 02834778 bicycle 59
03325088 faucet 744 04004475 printer 166 02954340 cap 56
03046257 clock 655 03513137 helmet 162
03991062 flowerpot 602 03761084 microwaves 152'''
    lis_info_elem = list(filter(None, re.split(' |\n', info_cats)))

    lis_cats = []
    dict_cat2id = {}
    dict_cat2num = {}
    count = 0

    lis_cat_unique_partnet = []
    for i in range(0, len(lis_info_elem), 3):
        id, cat, num = lis_info_elem[i], lis_info_elem[i+1], lis_info_elem[i+2]
        # print(id, cat, num)
        
        dict_cat2id[cat] = id
        dict_cat2num[cat] = num

        lis_cats.append(cat)

        if cat not in Categorys_PartNet:
            lis_cat_unique_partnet.append(cat)

    lis_cats = sorted(lis_cats)
    lis_cat_unique_partnet = sorted(lis_cat_unique_partnet)

    lis_cat_no_partnet = []
    if lis_cat_skip is not None:
        for cat in lis_cat_unique_partnet:
            if cat in lis_cat_skip:
                continue
            lis_cat_no_partnet.append(cat)


    # print('Categories of partnet:', sorted(Categorys_PartNet), '\n\n')
    # print('Categories of unique in shapenet:', lis_cat_unique_partnet)
    return lis_cats, dict_cat2id, dict_cat2num, lis_cat_no_partnet
lis_cats, dict_cat2id, dict_cat2num, lis_cat_no_partnet = parse_shapenet_cats()
Categorys_ShapeNet = lis_cat_no_partnet


# sphere view points with radius 4
Path_Mesh_SR4 = _env_path(
    'STRUCTRE_VIEWPOINT_MESH',
    os.path.join(DIR_ROOT_PARTNET, 'geometry', 'sphere_r4.ply'),
)
Lis_ViewPoints_Sphere = _fibonacci_sphere()


@lru_cache(maxsize=None)
def get_viewpoints_sphere(path=None):
    path = _normalize_path(path or Path_Mesh_SR4)
    if not os.path.isfile(path):
        return Lis_ViewPoints_Sphere

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            'Open3D is required to load STRUCTRE_VIEWPOINT_MESH'
        ) from exc

    vertices = np.asarray(o3d.io.read_triangle_mesh(path).vertices).copy()
    if vertices.size == 0:
        raise ValueError(f'Viewpoint mesh contains no vertices: {path}')
    return vertices
