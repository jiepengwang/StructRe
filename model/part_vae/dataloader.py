import torch
from torch.utils.data import Dataset

import json
import os
import numpy as np

from termcolor import cprint
from copy import copy

import utils.utils_io as IOUtils
import confs.path as PathConfig
from model.part_vae.pos_encoding import patch_grid_size

class PartnetCodeLoadPreprocess(Dataset):
    def __init__(self, args, conf, mode):
        self.use_np_shuffle = conf['dataset.use_np_shuffle']

        self.max_dataset_size = int(conf['dataset.max_dataset_size'])
        self.max_category_size = int(conf['dataset.max_category_size'])

        cprint("[PartVAE] Use KL-regularized SparseVAE code.", "yellow")

        self.n_reso_bottleneck = patch_grid_size(conf['model.decoder.seq_len'])
        cprint(f'SparseVAE bottleneck resolution: {self.n_reso_bottleneck}.', 'yellow')

        dataroot = PathConfig.DIR_ROOT_PARTNET
        with open(f'{dataroot}/info.json') as f:
            self.info = json.load(f)

        folder_code = PathConfig.Folder_Code_Sparse_VAE

        cprint("Use sparse sequence.", "yellow")

        self.code_dir = f'{dataroot}/{folder_code}'
        if not os.path.isdir(self.code_dir):
            raise FileNotFoundError(
                f"Stage 1 code artifact directory does not exist: {self.code_dir}"
            )
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
            raise ValueError(
                f"Unsupported dataset category: {conf['dataset.category']}"
            )


        # load partnet data
        mode_data = 'part_vae_sparse_vae'
        dict_conf_data = {
            'dir_data_root': PathConfig.DIR_ROOT_PARTNET,
            'mode': mode,
            'mode_data':  mode_data,
            'folder_data': folder_code,

            'all_cats_to_load': all_cats_partnet,
            'max_category_size': self.max_category_size,

        }
        model_list_augmented_shape, cats_list_augmeted_shape, count_models = get_samples_data(args, dict_conf_data)
        cprint(f'[PartNet] Total loaded samples: {len(model_list_augmented_shape)}. Models: {count_models}', 'yellow')

        if all_cats_shapenet:
            dict_conf_data_shapenet = copy(dict_conf_data)
            dict_conf_data_shapenet['dir_data_root'] = PathConfig.DIR_DATA_SHAPENET_OURS

            dict_conf_data_shapenet['all_cats_to_load'] = all_cats_shapenet

            model_list_augmented_shape2, cats_list_augmeted_shape2 , count_models2 = get_samples_data(args, dict_conf_data_shapenet)
            cprint(f'[ShapeNet] Total loaded samples: {len(model_list_augmented_shape2)}. Models: {count_models2}', 'yellow')
            model_list_augmented_shape += model_list_augmented_shape2
            cats_list_augmeted_shape += cats_list_augmeted_shape2
            
        self.model_list = model_list_augmented_shape
        self.cats_list = cats_list_augmeted_shape

        if self.use_np_shuffle:
            np.random.default_rng(seed=0).shuffle(self.model_list)
            np.random.default_rng(seed=0).shuffle(self.cats_list)

        self.model_list = self.model_list[:self.max_dataset_size]
        cprint(f'[*] Total loaded samples: {len(self.model_list)}', 'yellow')
        
        self.N = len(self.model_list)

    def __getitem__(self, index):

        synset = self.cats_list[index]
        model = self.model_list[index]

        path_data = f"{model}/code.h5"
        try:
            data_dict = IOUtils.read_h5_to_dict(path_data, use_tensor=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SparseVAE code: {path_data}"
            ) from exc

        for key in data_dict:
            data_dict[key] = data_dict[key][None, ...]

        code = data_dict["code"]
        code_coords = data_dict["code_coords"]
        coeff_idx = torch.tensor(
            [1, self.n_reso_bottleneck, self.n_reso_bottleneck**2]
        )
        code_coords_1d = (
            code_coords * coeff_idx[None, None, ...]
        ).sum(dim=2).squeeze(dim=0)

        _, indices_sorted = torch.sort(code_coords_1d)
        data_dict["code"] = code[:, indices_sorted]
        data_dict["code_coords"] = code_coords[:, indices_sorted]
        data_dict.pop("code_enc", None)
        code = data_dict["code"]

        n_code = code.shape[1]

        # Decoder occupancy target on the 8^3 grid.
        n_vox_dense = self.n_reso_bottleneck**3
        mask_nonempty=torch.zeros(n_vox_dense).bool()
        mask_nonempty[code_coords_1d] = True
        data_dict['mask_nonempty'] = mask_nonempty[None, ...]

        mask_valid = torch.ones(1, n_code)
        data_dict["mask_valid"] = mask_valid

        if synset in self.all_cats_shapenet:
            if ('code_sparse_vae_partial' in model) or ('00_level0' in model):
                level_id = 0
            else:
                level_id = 4
        elif synset in self.all_cats_partnet:   
            if 'code_sparse_vae_partial' in model:
                level_id = 0
            else:              
                model_id = model.split('/')[-1]
                name_model_level = model_id.split('_')[1]
                level_id = int(name_model_level[5:])
        else:
            raise ValueError(
                f"Unknown category {synset!r} for Stage 2 sample: {path_data}"
            )
        data_dict['level_id'] = level_id

        data_dict['path'] = path_data
        data_dict['code_coords'] = data_dict['code_coords'].long()

        return data_dict

    def __len__(self):
        return self.N



def get_samples_data(args, dict_conf_data):
    mode = dict_conf_data['mode']
    all_cats_to_load = dict_conf_data['all_cats_to_load']
    max_category_size = dict_conf_data['max_category_size']

    dir_data_root = dict_conf_data['dir_data_root']
    mode_data = dict_conf_data['mode_data']
    folder_data = dict_conf_data['folder_data']

    count_models_all = 0
    model_list_all, cats_list_all = [], []
    for synset in all_cats_to_load:
        with open(f'{dir_data_root}/filelists/{synset}_{mode}.lst') as f:
            model_list_s = []

            path = (
                f'{dir_data_root}/filelists/{mode_data}/'
                f'{synset}_{mode}_succeed.json'
            )
            samples_by_model = IOUtils.read_json_to_dict(path)

            count_models = 0
            vec_model_ids = []
            for l in f.readlines():
                model_id = l.rstrip('\n')
                vec_model_ids.append(model_id)

            if args.model_id_end > 0:
                model_id_end = min(args.model_id_end, len(vec_model_ids))
                vec_model_ids = vec_model_ids[args.model_id_start:model_id_end]
                cprint(
                    f'Total models: {len(vec_model_ids)} '
                    f'(start-end: {args.model_id_start}-{model_id_end})',
                    'red',
                )
            n_models = len(vec_model_ids)
            
            for model_id in vec_model_ids:    
                dir_shape_curr = f'{dir_data_root}/{folder_data}/{synset}/{model_id}'
                if model_id not in samples_by_model:
                    continue

                lis_name_sample = samples_by_model[model_id]
                for shape_name in lis_name_sample:
                    path = f'{dir_shape_curr}/{shape_name}'
                    model_list_s.append(path)

                count_models += 1
                if count_models >= max_category_size:
                    break
            cprint(f'[{synset}] Loaded samples: {len(model_list_s)}. Models: {count_models}/{n_models}', 'blue')
            count_models_all += count_models
            model_list_all += model_list_s
            cats_list_all += [synset] * len(model_list_s)
    return model_list_all, cats_list_all, count_models_all
