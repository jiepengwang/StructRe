import numpy as np
from multiprocessing import Pool, pool

import torch
from torch import nn

from model.part_vae.encoder import StructEncoder
from model.part_vae.decoder import StructDecoder

from model.part_vae.pos_encoding import PositionEmbeddingLearned, patch_grid_size
from model.part_vae.loss import PartVAELoss
from model.sparse_vae.checkpoint import compose_sparse_vae_state_dict
from model.utils.latent_regularization import KLReg

from einops import repeat
from pyhocon import ConfigFactory
from termcolor import colored
from datetime import datetime

import utils.utils_io as IOUtils
import utils.utils_training as TrainUtils

import confs.path as PathConfig

LEGACY_PART_VAE_STATE_PREFIXES = (
    "encoder.pos_embed_func_sin",
    "encoder.layer_norm",
    "encoder.cls_token",
    "decoder.cls_token",
    "decoder.coord_decoder",
    "decoder.dec_linear",
)


def _is_legacy_part_vae_state_key(key):
    return any(
        key == prefix or key.startswith(f"{prefix}.")
        for prefix in LEGACY_PART_VAE_STATE_PREFIXES
    )


def compose_part_vae_state_dict(model_state, checkpoint_state):
    checkpoint_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint_state.items()
    }
    geometry_prefix = "sparse_vae."
    model_geometry = {
        key: value
        for key, value in model_state.items()
        if key.startswith(geometry_prefix)
    }
    checkpoint_geometry = {
        key for key in checkpoint_state if key.startswith(geometry_prefix)
    }
    checkpoint_core = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith(geometry_prefix)
    }
    model_core_keys = {
        key for key in model_state if not key.startswith(geometry_prefix)
    }
    missing = sorted(model_core_keys - checkpoint_core.keys())
    unexpected = sorted(
        key
        for key in checkpoint_core.keys() - model_core_keys
        if not _is_legacy_part_vae_state_key(key)
    )
    if checkpoint_geometry and not model_geometry:
        unexpected.extend(sorted(checkpoint_geometry))
    if missing or unexpected:
        raise RuntimeError(
            "Stage2 checkpoint is incompatible with PartVAE core: "
            f"missing={missing}, unexpected={unexpected}"
        )

    composed_state = dict(model_geometry)
    composed_state.update(
        (key, value)
        for key, value in checkpoint_core.items()
        if key in model_core_keys
    )
    return composed_state


def save_struct_data(tuple_data):
    path, dict_data = tuple_data
    if IOUtils.check_existence(path):
        return

    try:
        IOUtils.ensure_pdir_existence(path)
        IOUtils.write_dict_to_h5(path, dict_data)
    except Exception as exc:
        raise RuntimeError(f"Failed to save structure code: {path}") from exc

def save_struct_data_chunk(lis_chunk):
    n_thread = 8
    with pool.ThreadPool(n_thread) as p_pool:
        p_pool.map(save_struct_data, lis_chunk)    
    
def get_part_level_weight(vec_part_level_id, lis_level_weight):
    part_level_weight = torch.tensor(lis_level_weight)
    max_level = len(lis_level_weight)-1
    vec_part_level_id = vec_part_level_id.clamp(0,max_level)
    vec_part_level_weight = part_level_weight[vec_part_level_id]
    return vec_part_level_weight

class PartVAE(nn.Module):
    def __init__(self, args, conf):
        super(PartVAE, self).__init__()
        self.conf_sparse_vae = ConfigFactory.parse_file(conf['general.config_sparse_vae'])
        self.conf_part_vae = conf


        # 1. Configure the optional Stage 1 geometry model
        should_load_sparse_vae = self.conf_part_vae['model.should_load_sparse_vae']

        if should_load_sparse_vae:
            from model.sparse_vae.sparse_vae import SparseVAE
            self.sparse_vae = SparseVAE(args, self.conf_sparse_vae)
            self.load_sparse_vae(
                PathConfig.resolve_stage_checkpoint(args, stage=1)
            )
            self.sparse_vae.eval()
        else:
            self.sparse_vae = None

        self.pos_embed_func = PositionEmbeddingLearned(conf)

        # 2. model components
        self.encoder = StructEncoder(self.conf_part_vae, self.conf_sparse_vae)
        self.n_code_seg = self.encoder.n_cls
        self.decoder = StructDecoder(self.conf_part_vae, self.conf_sparse_vae)
        self.n_patch_tokens = self.conf_part_vae["model.decoder.seq_len"]
        self.patch_grid_size = patch_grid_size(self.n_patch_tokens)
        if self.conf_part_vae['model.pe.n_emb'] != self.patch_grid_size:
            raise ValueError(
                'Stage 2 positional embedding count must match patch grid: '
                f"{self.conf_part_vae['model.pe.n_emb']} != {self.patch_grid_size}"
            )

        self.regularizer = KLReg()

        self.loss = PartVAELoss(self.conf_part_vae)

        self.n_process = 6
        self.process_pool = None

    
    def forward(self, input_model, device_data, iter_step = None):
        '''
        Args:
            x: N*B
        '''
        logs_all = {}
        
        # 0. input
        data_dict_inp = self.set_input(input_model)

        # 1. encoder and KL regularization
        h, _, _ = self.encoder(data_dict_inp, self.pos_embed_func)
        h_shape = h[:self.n_code_seg] # t b c
        self.latent_code, self.kl_loss, _, _ = self.regularizer(
            h_shape,
            mask_valid=None,
            sample_posterior=self.training,
        )

        # 2. decoder

        data_dict_out = self.decoder(self.latent_code)
        self.data_dict_out = data_dict_out

        # 3. loss
        loss_code_coord, logs_loss = self.loss(data_dict_inp, data_dict_out)
        loss_all = loss_code_coord + \
                    self.conf_part_vae['train.klreg_weight'] * self.kl_loss
        
        phase = 'train' if self.training else 'test'
        logs_all.update(logs_loss)

        # 4. code statistics
        code_shape = self.latent_code.detach()[:self.n_code_seg]
        logs_stat_code = TrainUtils.get_tensor_statistics_fast(code_shape, prefix='logs/0_shape_code_', phase=phase)
        logs_all.update(logs_stat_code)

        logs_all.update({
            f'loss/loss_klreg_{phase}': self.kl_loss.detach().item(),
            f'loss/loss_all_{phase}': loss_all.detach().item(),
            f'logs/seq_length_{phase}': data_dict_inp['code'].shape[0]
        })
        
        return loss_all, logs_all

    def load_sparse_vae(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = checkpoint.get("model", checkpoint)
        state_dict = compose_sparse_vae_state_dict(
            self.sparse_vae.state_dict(), checkpoint_state
        )
        self.sparse_vae.load_state_dict(state_dict, strict=True)
        print(colored(
            f"[*] SparseVAE: weight successfully loaded from: {checkpoint_path}",
            "blue",
        ))

    def set_input(self, input_model):
        for key in input_model:
            if key in ["level_id", "path"]:
                continue
            input_model[key] = input_model[key].cuda().transpose(0, 1).contiguous()

        input_model["part_level_weight"] = None
        if "level_id" in input_model:
            input_model["level_id"] = torch.tensor(input_model["level_id"])
            part_level_weight = get_part_level_weight(
                input_model["level_id"], self.conf_part_vae["train.lis_level_weight"]
            )
            input_model["part_level_weight"] = part_level_weight.cuda()

        return input_model
        
    def _decode_patch_code_output(self, data_dict_out):
        code = data_dict_out["out_code"]
        mask_nonempty = torch.argmax(data_dict_out["out_status"], dim=-1) < 1
        code_coords = TrainUtils.unfold_indices_to_grid(
            self.n_patch_tokens, self.patch_grid_size, use_tensor=True
        ).to(code.device)
        code_coords = repeat(
            code_coords, "t c -> t b c", b=code.shape[1]
        )
        return {
            "code": code,
            "code_coords": code_coords,
            "mask_nonempty": mask_nonempty,
        }

    def validate_geometry(self, dict_data_inp):
        """Decode the latest forward output through the Stage 1 model."""
        self.eval()
        with torch.no_grad():
            for path in dict_data_inp["path"]:
                print(f"Data path: {path.split('/')[-4:]}")

            dict_infer_out = self._decode_patch_code_output(
                self.data_dict_out
            )
            if self.sparse_vae is None:
                raise RuntimeError(
                    "Stage 1 must be loaded before PartVAE mesh validation"
                )
            self.sparse_vae.decode_code_sequence(
                dict_data_inp, dict_infer_out, tsdf_resolution_mc=128
            )
        return {}

    def inference_struct_code(
        self,
        lis_code,
        *,
        memory_inp=None,
        memory_key_padding_mask=None,
        mode_decode_sdf='sparse',
        should_decode_part_code_only=False,
        return_patch_code=False,
    ):
        """Decode continuous KL-regularized structure codes."""
        self.eval()
        self.latent_code = lis_code.transpose(0, 1)
        self.kl_loss = torch.zeros(1, device=lis_code.device)

        with torch.no_grad():
            t1 = datetime.now()

            self.data_dict_out = self.decoder(
                lis_code, memory_key_padding_mask, memory_inp
            )

            dict_infer_out = self._decode_patch_code_output(
                self.data_dict_out
            )

            t2 = datetime.now()
            t_part_vae_decode = (t2-t1).total_seconds()

            if should_decode_part_code_only:
                return dict_infer_out

            if self.sparse_vae is None:
                raise RuntimeError(
                    "Stage 1 must be loaded before decoding structure codes"
                )
            dict_out_decoder = self.sparse_vae.decode_code_sequence(
                None, dict_infer_out, mode_decode_sdf=mode_decode_sdf
            )

            t3 = datetime.now()
            t_sparse_vae_decode = (t3-t2).total_seconds()

            print(f"[Decode part] T_part_vae_decode: {t_part_vae_decode:.04f}. T_sparse_vae_decode: {t_sparse_vae_decode:.04f}")
            if return_patch_code:
                dict_infer2 = {
                    'code': dict_infer_out['code'][dict_infer_out['mask_nonempty']][None, ...],
                    'code_coords': dict_infer_out['code_coords'][dict_infer_out['mask_nonempty']][None, ...],
                }
                return dict_infer2, dict_out_decoder

            return dict_out_decoder

    def extract_code(self, data_dict, folder_code):
        latent_codes = self.latent_code.detach().cpu().numpy()
        vec_data = []

        for idx_shape, path in enumerate(data_dict["path"]):
            cat_curr, model_name, part_name, _ = path.split('/')[-4:]
            if cat_curr in PathConfig.Categorys_PartNet:
                dir_data_root = PathConfig.DIR_ROOT_PARTNET
            elif cat_curr in PathConfig.Categorys_ShapeNet:
                dir_data_root = PathConfig.DIR_DATA_SHAPENET_OURS
            else:
                raise ValueError(f"Unsupported category: {cat_curr}")

            path_struct_code = (
                f"{dir_data_root}/{folder_code}/{cat_curr}/{model_name}/{part_name}"
                f"/code_struct.h5"
            )
            struct_code_idx = np.zeros(len(latent_codes), dtype=np.int64)
            vec_data.append((path_struct_code, {
                "struct_code": latent_codes[:, idx_shape],
                "struct_code_idx": struct_code_idx,
            }))

        if not vec_data:
            return
        n_chunks = self.n_process
        chunk_size = len(vec_data) // n_chunks
        if len(vec_data) % n_chunks > 0:
            chunk_size += 1
        chunks = [
            vec_data[i:i + chunk_size]
            for i in range(0, len(vec_data), chunk_size)
        ]
        if self.process_pool is None:
            self.process_pool = Pool(processes=self.n_process)
        self.process_pool.map(save_struct_data_chunk, chunks)
    
    def infer_part_vae_memory_mesh(self, input_model):
        """Infer structure codes and encoder memory from patch codes."""
        data_dict_inp = self.set_input(input_model)
        h, memory_key_padding_mask, out_tf_enc = self.encoder(
            data_dict_inp, self.pos_embed_func
        )
        h_shape = h[:self.n_code_seg]
        self.latent_code, self.kl_loss, _, _ = self.regularizer(
            h_shape,
            mask_valid=None,
            sample_posterior=self.training,
        )
        return {
            "code": self.latent_code,
            "memory_key_padding_mask": memory_key_padding_mask,
            "memory_inp": out_tf_enc[self.n_code_seg:],
        }
