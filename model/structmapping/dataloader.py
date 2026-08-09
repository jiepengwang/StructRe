import torch
from torch.utils.data import Dataset

import json
import numpy as np
from termcolor import cprint
from einops import rearrange
from copy import copy
                   
import utils.utils_io as IOUtils
import utils.utils_geometry as GeoUtils

import confs.path as PathConfig
from model.data_contracts import normalize_parent_eos, reverse_child_parent_info



class HierarchyCodeLoadPreprocess(Dataset):
    def __init__(self, args, conf, mode):
        self.use_np_shuffle = conf['dataset.use_np_shuffle']
        self.max_dataset_size = conf['dataset.max_dataset_size']
        self.max_category_size = conf['dataset.max_category_size']
        self.is_training = 'train' in mode


        

        

        self.max_parent_tokens = (
            conf["model.decoder.detr.num_queries"] + 1
        )
        
        with open(f'{PathConfig.DIR_ROOT_PARTNET}/info.json') as f:
            self.info = json.load(f)

        self.all_cats_partnet = self.info['all_cats']
        _, _, _, lis_cat_no_partnet = PathConfig.parse_shapenet_cats()
        self.all_cats_shapenet = lis_cat_no_partnet
        if conf['dataset.category'] == 'all':
            all_cats_partnet = self.all_cats_partnet
            all_cats_shapenet = self.all_cats_shapenet
        elif conf['dataset.category'] == 'all_partnet':
            all_cats_partnet = self.all_cats_partnet
            all_cats_shapenet = []
        elif conf['dataset.category'] == 'all_shapenet':
            all_cats_partnet = []
            all_cats_shapenet = self.all_cats_shapenet
        elif conf['dataset.category'] in self.all_cats_partnet:
            all_cats_partnet = [conf['dataset.category']]
            all_cats_shapenet = []
        elif conf['dataset.category'] in self.all_cats_shapenet:
            all_cats_partnet = []
            all_cats_shapenet = [conf['dataset.category']]
        else:
            raise ValueError(f"Unknown dataset category: {conf['dataset.category']}")

        dict_conf_data = {
            'dir_data_root': PathConfig.DIR_ROOT_PARTNET,
            'mode': mode,
            'folder_data': PathConfig.Folder_Name_Hierarchy_Sparse_VAE,
            'all_cats_to_load': all_cats_partnet,
            'max_category_size': self.max_category_size,
        }

        model_paths = []
        if all_cats_partnet:
            model_paths, count_models = get_samples_data(
                dict_conf_data, args.model_id_start, args.model_id_end
            )
            cprint(
                f'[PartNet] Total loaded samples: {len(model_paths)}. Models: {count_models}',
                'yellow',
            )

        if all_cats_shapenet:
            shapenet_conf = copy(dict_conf_data)
            shapenet_conf['dir_data_root'] = PathConfig.DIR_DATA_SHAPENET_OURS
            shapenet_conf['all_cats_to_load'] = all_cats_shapenet

            shapenet_paths, count_models = get_samples_data(
                shapenet_conf
            )
            cprint(
                f'[ShapeNet] Total loaded samples: {len(shapenet_paths)}. Models: {count_models}',
                'yellow',
            )
            model_paths += shapenet_paths

        self.model_list = model_paths

        if self.use_np_shuffle:
            np.random.default_rng(seed=0).shuffle(self.model_list)

        if len(self.model_list)>self.max_dataset_size:
            cprint(f'Total samples are larger than maximum dataset size', 'red')
        self.model_list = self.model_list[:self.max_dataset_size]
        cprint(f'[*] Total loaded samples: {len(self.model_list)}', 'yellow')


    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, index):

        h5_file = self.model_list[index]

        keys_to_skip=['path_child', 'code_path_child', 'code_path_parent',
                        'node_idx_child', 'node_idx_parent',
                        'link_p2c_dict', 'rotation_angle',
                        'nodeID_link_p2c_dict',
                        'nodeID_link_c2p',
                        'code_enc_child',
                        'code_coords', 'mask_nonempty']

        dict_pair = IOUtils.read_h5_to_dict(h5_file, keys_to_skip=keys_to_skip,
                                                use_tensor=True, keys_skip2tensor = ['link_p2c_dict'],
                                                should_expand_tensor=True)


        if dict_pair["bb_parent"].shape[1] > self.max_parent_tokens:
            raise ValueError(
                f"Hierarchy has more parent tokens than decoder capacity "
                f"({self.max_parent_tokens}): {h5_file}"
            )




        if 'is_patch' in dict_pair:
            # patch code
            if dict_pair["is_patch"].squeeze().item() != 1:
                raise ValueError(f"Invalid patch hierarchy marker: {h5_file}")

            dict_pair['is_patch'] = torch.tensor([[1]])

            # align input
            dict_pair["patch_bb_child"] = dict_pair["bb_child"].clone()
            dict_pair["patch_code_child"] = dict_pair["code_child"].clone()
            dict_pair["patch_code_idx_child"] = (
                dict_pair["code_idx_child"].clone()
            )

            dict_pair["bb_child"] = dict_pair["bb_child"][:, :0]
            dict_pair["code_child"] = dict_pair["code_child"][:, :0]
            dict_pair["code_idx_child"] = dict_pair["code_idx_child"][:, :0]
            dict_pair["link_c2p"] = dict_pair["link_c2p"][:, :0]

        else:
            # part code
            dict_pair['is_patch'] = torch.tensor([[0]])

            if "patch_code_child" not in dict_pair:
                raise ValueError(f"Hierarchy is missing patch code: {h5_file}")


        # align dim
        for key in dict_pair:
            if key in ['code_idx_child', 'code_idx_parent', 'patch_code_idx_child']:
                if dict_pair[key].ndim == 2:
                    dict_pair[key] = rearrange(dict_pair[key], 'b t -> b t 1')

            if key in ['code_child', 'code_parent', 'patch_code_child']:
                if dict_pair[key].ndim == 3:
                    dict_pair[key] = rearrange(dict_pair[key], 'b t c -> b t 1 c')

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

        _, view_name, hierarchy_name = h5_file.split('/')[-3:]
        idx_dir = 0
        can_reverse = (
            self.is_training
            and view_name == 'view_0000'
            and hierarchy_name != 'hierarchy_pair_0000.h5'
        )
        if can_reverse and np.random.rand() > 0.5:
            dict_pair = reverse_child_parent_info(dict_pair)
            idx_dir = 1

        dict_pair['idx_dir_token'] = torch.tensor([[idx_dir]])

        if 'link_patch2part_child' in dict_pair:
            del dict_pair['link_patch2part_child']

        if 'link_patch2part_parent' in dict_pair:
            dict_pair['link_patch2part_parent'] = dict_pair['link_patch2part_parent'].int()

        return dict_pair


def get_samples_data(data_conf, model_id_start=0, model_id_end=-1):
    mode = data_conf['mode']
    categories_to_load = data_conf['all_cats_to_load']
    max_category_size = data_conf['max_category_size']
    data_root = data_conf['dir_data_root']
    hierarchy_folder = data_conf['folder_data']

    filelist_folder = 'mapping_sparse_vae'

    total_models = 0
    model_paths = []
    for category in categories_to_load:
        with open(f'{data_root}/filelists/{category}_{mode}.lst') as filelist:
            model_ids = [line.strip() for line in filelist]

        if model_id_end > 0:
            model_id_end_curr = min(model_id_end, len(model_ids))
            model_ids = model_ids[model_id_start:model_id_end_curr]
            cprint(
                f'[{category}] Total models: {len(model_ids)} '
                f'(start-end: {model_id_start}-{model_id_end_curr})',
                'red',
            )

        n_models = len(model_ids)
        succeed_path = (
            f'{data_root}/filelists/{filelist_folder}/'
            f'{category}_{mode}_succeed.json'
        )
        samples_by_model = IOUtils.read_json_to_dict(succeed_path)
        category_paths = []
        count_models = 0

        for model_id in model_ids:
            if model_id not in samples_by_model:
                continue

            model_folder = f'{data_root}/{hierarchy_folder}/{category}/{model_id}'
            for shape_name in samples_by_model[model_id]:
                _, view_name, _ = shape_name.split('/')
                if view_name == 'view_0002':
                    continue
                category_paths.append(f'{model_folder}/{shape_name}')

            count_models += 1
            if count_models >= max_category_size:
                break

        cprint(
            f'[{category}] Loaded samples: {len(category_paths)}. '
            f'Models: {count_models}/{n_models}',
            'blue',
        )
        total_models += count_models
        model_paths += category_paths

    return model_paths, total_models
