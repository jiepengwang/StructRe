import torch
from torch import nn

from termcolor import colored
from pyhocon import ConfigFactory

from model.structmapping.encoder import MappingEncoder
from model.structmapping.decoder import MappingDecoder
from model.structmapping.loss import StructLoss
from model.structmapping.matcher import build_matcher
from model.structmapping.inference import (
    GeometryEncodingMixin,
    HierarchyInferenceMixin,
    ShapeDecodingMixin,
    StructMappingInferenceMixin,
)

from model.part_vae.part_vae import PartVAE, compose_part_vae_state_dict
from model.structmapping.pe import build_position_encoding


import confs.path as PathConfig

import model.structmapping.utils_model.utils_model_io as ModelIOUtils
import model.structmapping.utils_model.utils_model_sampling as ModelSamplingUtils


class StructMapping(
    StructMappingInferenceMixin,
    HierarchyInferenceMixin,
    GeometryEncodingMixin,
    ShapeDecodingMixin,
    nn.Module,
):
    def __init__(self, args, conf):
        super(StructMapping, self).__init__()
        self.conf = conf
        self.args = args

        self.mapping_direction_infer = args.mapping_direction_infer
        self.decode_chunk_size = args.mapping_decode_chunk_size
        if self.decode_chunk_size <= 0:
            raise ValueError("Stage3 decode chunk size must be positive")
        
        
        # 0. parse args

        
        self.sampling_strategy_infer = self.conf['model.sampling_strategy_infer']
        self.filter_strategy_infer = self.conf['model.filter_strategy_infer']
        self.lis_schedule = self.conf['model.lis_schedule_conf_iter'] #[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        self.n_iter = self.conf['model.n_iter']
        self.thres_prob_valid_part_fix = self.conf['train.thres_prob_valid_part']

        self.matcher = build_matcher(conf)
        self.pos_embedding = build_position_encoding(self.conf)
 
        
        self.encoder = MappingEncoder(self.conf)
        self.decoder = MappingDecoder(self.conf)
        self.criterion = StructLoss(self.conf)



        self.freq_reset_counter_query_step = (
            self.conf["train.freq_reset_counter"] * self.args.num_batches
        )
        if self.freq_reset_counter_query_step <= 0:
            raise ValueError("StructMapping query-counter reset interval must be positive")
        
        # 1. Configure the optional Stage 2 geometry model
        self.conf_part_vae = ConfigFactory.parse_file(self.conf['general.config_part_vae'])
        
        if self.conf['model']['should_load_sparse_vae']:
            self.conf_part_vae['model']['should_load_sparse_vae'] = self.conf['model']['should_load_sparse_vae']
        
            self.part_vae = PartVAE(args, self.conf_part_vae)
            self.load_part_vae(
                PathConfig.resolve_stage_checkpoint(args, stage=2)
            )
            self.part_vae.eval()

            self.renderer = self.part_vae.sparse_vae.renderer
        

                
           
    def load_part_vae(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = checkpoint.get("model", checkpoint)
        state_dict = compose_part_vae_state_dict(
            self.part_vae.state_dict(), checkpoint_state
        )
        self.part_vae.load_state_dict(state_dict, strict=True)
        print(colored(
            f"[*] PartVAE: weight successfully loaded from: {checkpoint_path}",
            "blue",
        ))

    def to_cuda(self, dict_data_inp):
        device = next(self.parameters()).device
        for key, value in dict_data_inp.items():
            if key == "path" or not torch.is_tensor(value):
                continue
            dict_data_inp[key] = value.to(device).transpose(0, 1).contiguous()

        dict_data_inp["code_idx_parent"] = dict_data_inp["code_idx_parent"].long()
        dict_data_inp["code_idx_child"] = dict_data_inp["code_idx_child"].long()
        if "patch_code_idx_child" in dict_data_inp:
            dict_data_inp["patch_code_idx_child"] = (
                dict_data_inp["patch_code_idx_child"].long()
            )

        bbox_size = self.conf["model.pe.bbox_size"]
        n_embeddings = self.conf["model.pe.n_emb"]
        if "patch_bb_child" in dict_data_inp:
            dict_data_inp["patch_bb_child"] = (
                ModelIOUtils.voxelize_bbox_to_vertices(
                    dict_data_inp["patch_bb_child"],
                    bbox_size,
                    n_embeddings,
                    mode="vertices",
                )
            )
        dict_data_inp["bb_child"] = ModelIOUtils.voxelize_bbox_to_vertices(
            dict_data_inp["bb_child"],
            bbox_size,
            n_embeddings,
            mode="vertices",
        )
        dict_data_inp["bb_parent"] = ModelIOUtils.voxelize_bbox_to_vertices(
            dict_data_inp["bb_parent"],
            bbox_size,
            n_embeddings,
            mode="vertices",
        )

        return dict_data_inp
        
    def forward(self, dict_data_inp, device, iter_steps):
        logs_all = {}
        self.dict_data_inp = self.to_cuda(dict_data_inp)

        dict_out_enc = self.encoder(dict_data_inp, self.pos_embedding)
        memory, memory_key_padding_mask = dict_out_enc['memory'], dict_out_enc['mask_padding']
        
        input_tf_decoder = self.decoder.set_input(dict_data_inp)
        self.dict_out = self.decoder(input_tf_decoder, memory, memory_key_padding_mask)
        self.dict_out['latent_tf_inp'] = dict_out_enc['latent_tf_inp']

        # matching
        dict_matcher = self.matcher(self.dict_out, dict_data_inp, dict_attach=None)  # T*B
        logs_all.update(dict_matcher['logs_matcher'])
        
        # loss
        loss_all, logs_loss = self.criterion(self.dict_out, dict_data_inp, dict_matcher)
        logs_all.update(logs_loss)
        
        if (
            self.training
            and iter_steps % self.freq_reset_counter_query_step == 0
        ):
            self.criterion.counter_query.zero_()

        # iterative decoding to solve multi-modality problem
        # random sampling
        filter_strategy = 'rand_number'
        dict_attach = ModelSamplingUtils.random_sampling_output(self.dict_out, dict_matcher,
                                                                        filter_strategy=filter_strategy,
                                                                        lis_schedule=self.lis_schedule)
        dict_out_iter = self.decoder(input_tf_decoder,
                                            memory, memory_key_padding_mask,
                                            dict_attach)
        dict_out_iter['latent_tf_inp'] = dict_out_enc['latent_tf_inp']

        # matcher and loss
        dict_matcher_v2 = self.matcher(dict_out_iter, dict_data_inp, dict_attach)
        loss_iter, logs_loss_iter = self.criterion(dict_out_iter, dict_data_inp, dict_matcher_v2)

        loss_all += loss_iter

        logs_all.update({
            'logs_iter': logs_loss_iter
        })


        return loss_all, logs_all

    def inference_one_step(self, dict_data_inp, dict_conf):
        dict_out_enc = self.encoder(dict_data_inp, self.pos_embedding)
        memory = dict_out_enc['memory']
        memory_key_padding_mask = dict_out_enc['mask_padding']

        input_tf_decoder = self.decoder.set_input(dict_data_inp)
        data_out_tmp = self.decoder(
            input_tf_decoder,
            memory,
            memory_key_padding_mask,
        )
        data_out_tmp['latent_tf_inp'] = dict_out_enc['latent_tf_inp']
        data_out_tmp = self.iterative_decode_parts(
            dict_conf,
            dict_data_inp,
            dict_out_enc,
            data_out_tmp,
        )

        thres_valid_part = self.thres_prob_valid_part_fix
        data_out_tmp['dick_link_cp'] = self.decode_hierarchy_pointer(
            dict_data_inp,
            data_out_tmp,
            thres_valid_part,
            mapping_direction_infer=self.mapping_direction_infer,
        )
        return dict_out_enc, data_out_tmp
