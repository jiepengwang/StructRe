import torch
from torch import nn

from model.structmapping.decoder import MLP

from model.utils.utils_transformer import TransformerEncoderLayer, TransformerEncoder





class MappingEncoder(nn.Module):
    '''Borrowed from ViT
    '''
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        # transformer embedding
        mparam = conf['model.encoder']
        d_tf = mparam['embed_dim']
        nhead = mparam['nhead']
        num_encoder_layers = mparam['nlayers_enc']
        dim_feedforward = mparam['d_hid']
        dropout = mparam['dropout']
        normalize_before = mparam['normalize_before']
        activation = mparam['activation']

        n_bbox_coord = 6
                
        d_code_sparse_vae = self.conf['model.dim_emb_patch_code']
        input_dim1 = d_code_sparse_vae + conf['model.pe.dim_emb']*n_bbox_coord
        self.linear_pre_tf_patch = MLP(input_dim =input_dim1,
                                            hidden_dim=self.conf['model.encoder.mlp_pre_patch.d_hidden'],
                                            output_dim=d_tf,
                                            num_layers=self.conf['model.encoder.mlp_pre_patch.n_layers'])
                
        input_dim2 = self.conf['model.dim_emb_code'] + conf['model.pe.dim_emb']*(n_bbox_coord+1)
                
        self.linear_pre_tf_struct_pe = MLP(input_dim = input_dim2,
                                                hidden_dim=self.conf['model.encoder.mlp_pre_struct_pe.d_hidden'],
                                                output_dim=d_tf,
                                                num_layers=self.conf['model.encoder.mlp_pre_struct_pe.n_layers']) # convert to code dimension

        
        # transoformer
        encoder_layer = TransformerEncoderLayer(d_tf, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_tf) if normalize_before else None
        self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.emb_direction = nn.Embedding(2, d_tf)
        
    @staticmethod
    def get_key_padding_mask(code_idx_child):
        mask = (code_idx_child < 0)
        mask = mask.transpose(0,1)  # T*B->B*T
        return mask

    def forward(self, dict_inp, pos_embedding_func):

        rotation_angle = dict_inp['rotation_angle']

        # 1. patch code
        bb_child_patch = dict_inp['patch_bb_child']
        code_child_patch = dict_inp['patch_code_child'].flatten(-2, -1) # B*T*n*C
        code_idx_child_patch = dict_inp['patch_code_idx_child']

        bb_child_patch_emb = pos_embedding_func(bb_child_patch, rot_angle=None, use_emb_rot=False)
        inp_tf_patch = torch.cat([bb_child_patch_emb, code_child_patch], dim=2)
        inp_tf_patch = self.linear_pre_tf_patch(inp_tf_patch)

        # 2. part code
        bb_child_part = dict_inp['bb_child']     # B*T*C
        code_child_part = dict_inp['code_child']
        code_idx_child_part = dict_inp['code_idx_child']  # B*T

        use_emb_rot = True
        bb_child_struct_emb = pos_embedding_func(bb_child_part, rot_angle=rotation_angle, use_emb_rot=use_emb_rot)
        code_child_struct_emb = code_child_part
        code_child_struct_emb = code_child_struct_emb.flatten(-2, -1) # B*T*n*C
                
        inp_tf_struct = torch.cat([bb_child_struct_emb, code_child_struct_emb], dim=2)
        inp_tf_struct = self.linear_pre_tf_struct_pe(inp_tf_struct)

        # 3. merge part and patch
        inp_tf = torch.cat([inp_tf_struct, inp_tf_patch], dim=0) # merge along dim T
        code_idx_child = torch.cat([code_idx_child_part, code_idx_child_patch], dim=0)
                
        
        idx_dir_token = dict_inp['idx_dir_token'].long()
        bs = idx_dir_token.shape[1]

        emb_dir = self.emb_direction(idx_dir_token)
        code_idx_dir_token = torch.zeros(
            (1, bs, 1),
            dtype=code_idx_child.dtype,
            device=code_idx_child.device,
        )

        inp_tf = torch.cat([inp_tf, emb_dir], dim=0)
        code_idx_child = torch.cat([code_idx_child, code_idx_dir_token], dim=0)

        mask_padding = self.get_key_padding_mask(code_idx_child)
        if mask_padding.ndim == 3:
            if mask_padding.shape[-1] != 1:
                raise ValueError(
                    "StructMapping expects one hierarchy index per token, got "
                    f"shape {tuple(code_idx_child.shape)}"
                )
            mask_padding = mask_padding[..., 0]
        elif mask_padding.ndim != 2:
            raise ValueError(
                f"Unsupported hierarchy index shape: {tuple(code_idx_child.shape)}"
            )
            
        x = self.transformer(inp_tf, src_key_padding_mask = mask_padding)
        dict_out_enc = {
            'memory': x,
            'mask_padding': mask_padding
        }
        dict_out_enc['latent_tf_inp'] = x

        return dict_out_enc
