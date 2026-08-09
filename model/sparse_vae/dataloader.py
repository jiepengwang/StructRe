import torch
from torch.utils.data import Dataset

import json
import numpy as np
from termcolor import cprint
from copy import copy
      
import utils.utils_io as IOUtils
import model.sparse_vae.data_io as DataloaderUtils
import confs.path as PathConfig

class PartnetLoadPreprocess(Dataset):
    def __init__(self, args, conf, mode):
        self.trunc_thres = conf['dataset.trunc_thres']
        self.use_np_shuffle = conf['dataset.use_np_shuffle']
        self.max_dataset_size = conf['dataset.max_dataset_size']
        self.max_category_size = conf['dataset.max_category_size'] 



        self.scale_cube = conf['dataset.scale_cube']
        print(f'Scale of cube: {self.scale_cube}')

        self.ratio_tsdf_grid = 256.0 / PathConfig.TSDF_Resolution_Part
        cprint(
            f"[Dataloader] Dataset input resolution: 256. "
            f"TSDF resolution: {PathConfig.TSDF_Resolution_Part}. "
            f"Coordinates ratio: {self.ratio_tsdf_grid}",
            "red",
        )
        
        
        dataroot = PathConfig.DIR_ROOT_PARTNET
        with open(f'{dataroot}/info.json') as f:
            self.info = json.load(f)

        mode_shape = conf['dataset.mode_shape']
        if mode_shape == 'shape_aug_rotation':
            folder_sdf = PathConfig.Folder_SDF_Augmented_Shape
            mode_data = 'sparse_vae_aug'
            self.should_load_tsdf = True
            complete_view_only = False
        elif mode_shape == 'shape_parts':
            folder_sdf = PathConfig.Folder_SDF_Part_Sparse_VAE
            mode_data = 'sparse_vae'
            self.should_load_tsdf = False
            complete_view_only = False
        elif mode_shape == 'shape_partial':
            folder_sdf = PathConfig.Folder_SDF_Partial_Sparse_VAE
            mode_data = 'sparse_vae_partial'
            self.should_load_tsdf = False
            complete_view_only = True
        else:
            raise ValueError(f'Unsupported Stage 1 shape mode: {mode_shape}')

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
        else:
            if conf['dataset.category'] in self.all_cats_partnet:
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
        dict_conf_data = {
            'dir_data_root': PathConfig.DIR_ROOT_PARTNET,
            'mode': mode,
            'mode_data':  mode_data,
            'folder_data': folder_sdf,

            'all_cats_to_load': all_cats_partnet,

            'max_category_size': self.max_category_size,
            'complete_view_only': complete_view_only,
        }
        model_list_augmented_shape, cats_list_augmeted_shape, count_models = get_samples_data(args, dict_conf_data)
        cprint(f'[PartNet] Total loaded samples: {len(model_list_augmented_shape)}. Models: {count_models}', 'yellow')

        if all_cats_shapenet:
            dict_conf_data_shapenet = copy(dict_conf_data)
            dict_conf_data_shapenet['dir_data_root'] = PathConfig.DIR_DATA_SHAPENET_OURS

            dict_conf_data_shapenet['all_cats_to_load'] = all_cats_shapenet

            model_list_augmented_shape2, cats_list_augmeted_shape2 , count_models = get_samples_data(args, dict_conf_data_shapenet)
            cprint(f'[ShapeNet] Total loaded samples: {len(model_list_augmented_shape2)}. Models: {count_models}', 'yellow')

            model_list_augmented_shape += model_list_augmented_shape2
            cats_list_augmeted_shape += cats_list_augmeted_shape2


        self.model_list = model_list_augmented_shape
        self.cats_list  = cats_list_augmeted_shape

        if self.use_np_shuffle:
            np.random.default_rng(seed=0).shuffle(self.model_list)
            np.random.default_rng(seed=0).shuffle(self.cats_list)

        self.model_list = self.model_list[:self.max_dataset_size]
        cprint(f'[*] Total loaded samples: {len(self.model_list)}', 'yellow')

        self.N = len(self.model_list)
        cprint(f'Load TSDF file with resolution: {PathConfig.TSDF_Resolution_Part}')


    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, index):

        synset = self.cats_list[index]
        sdf_h5_file = self.model_list[index]
        try:
            dict_data = DataloaderUtils.load_tsdf_data_h5_to_np(
                sdf_h5_file,
                tsdf_resolution=PathConfig.TSDF_Resolution_Part,
                bbox_size_half=PathConfig.BBox_Size_Half,
                ratio_tsdf_grid=self.ratio_tsdf_grid,
                should_load_tsdf=self.should_load_tsdf,
                scale_cube=self.scale_cube,
            )
        except Exception as error:
            raise RuntimeError(
                f'Failed to load Stage1 sample: {sdf_h5_file}'
            ) from error

        
        # convert numpy to tensor
        for key in (
            'sdf',
            'pointcloud',
            'normal',
            'mask_valid_pointcloud',
        ):
            if key in dict_data:
                dict_data[key] = torch.from_numpy(dict_data[key])
        
        if 'sdf' in dict_data:
            dict_data['sdf'] = dict_data['sdf'][None, ...]
        
            thres = self.trunc_thres
            if thres != 0.0:
                dict_data['sdf'] = torch.clamp(dict_data['sdf'], min=-thres, max=thres)
            
        dict_data.update({
            'cat_id': synset,
            'cat_str': synset,
        })

        for key in ['pointcloud', 'normal', 'mask_valid_pointcloud']:
            if key in dict_data:
                dict_data[key] = dict_data[key][None, ...]

        if 'mask_valid_pointcloud' in dict_data:
            dict_data['mask_valid_pointcloud'] = dict_data['mask_valid_pointcloud'].float()

        return dict_data
               

def get_samples_data(args, dict_conf_data):
    mode = dict_conf_data['mode']

    all_cats_to_load = dict_conf_data['all_cats_to_load']
    max_category_size = dict_conf_data['max_category_size']

    dir_data_root = dict_conf_data['dir_data_root']
    mode_data = dict_conf_data['mode_data']
    folder_data = dict_conf_data['folder_data']
    complete_view_only = dict_conf_data['complete_view_only']

    count_models_all = 0
    model_list_all, cats_list_all = [], []
    for synset in all_cats_to_load:
        
        with open(f'{dir_data_root}/filelists/{synset}_{mode}.lst') as f:
            model_list_s = []


            path = f'{dir_data_root}/filelists/{mode_data}/{synset}_{mode}_succeed.json'
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
                    if complete_view_only and '/view_0000.' not in shape_name:
                        continue
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
