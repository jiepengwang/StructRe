import torch
from torch import nn

from einops import repeat

from model.utils.utils_transformer import TransformerEncoderLayer, TransformerEncoder


class StructEncoder(nn.Module):
    """Transformer encoder for sparse Stage 1 patch-code sequences."""
    def __init__(self, conf_part_vae, conf_sparse_vae):
        super().__init__()
        # transformer, borrowed from AutoSDF
        sparse_latent_dim = conf_sparse_vae['model.latent.dim']

        # transformer embedding
        mparam = conf_part_vae['model.encoder']
        d_tf = mparam['embed_dim']
        nhead = mparam['nhead']
        num_encoder_layers = mparam['nlayers_enc']
        dim_feedforward = mparam['d_hid']
        dropout = mparam['dropout']
        normalize_before = mparam['normalize_before']
        activation = mparam['activation']

        d_inp = conf_part_vae["model.pe.dim_emb"] * 3 + sparse_latent_dim
        self.fuse_linear = nn.Linear(d_inp, d_tf)
        self.n_cls = 1
        self.embedding_start = nn.Embedding(self.n_cls, d_tf)

        # transformer
        encoder_layer = TransformerEncoderLayer(d_tf, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_tf) if normalize_before else None
        self.encoder_tf = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.quant_linear = nn.Linear(
            d_tf, 2 * conf_part_vae["model.latent.dim"]
        )

        self._init_weights()

    def _init_weights(self):
        """initialize the weights of params."""
        self.fuse_linear.bias.data.normal_(0, 0.02)
        self.fuse_linear.weight.data.normal_(0, 0.02)

        self.quant_linear.bias.data.normal_(0, 0.02)
        self.quant_linear.weight.data.normal_(0, 0.02)

    def forward(self, data_dict, pos_embedding_func):
        _, bs = data_dict['code'].shape[:2]
        # cls embedding
        emb_cls = self.embedding_start.weight
        emb_cls = repeat(emb_cls, 't d-> t b d', b=bs)

        code = data_dict['code']
        code_coords = data_dict['code_coords']

        # coordinate embedding
        emb_coord = pos_embedding_func(code_coords)

        inp_code = torch.cat([code, emb_coord], dim=2)
        inp_code = self.fuse_linear(inp_code)

        inp = torch.cat([emb_cls, inp_code], dim=0)

        mask_valid = data_dict["mask_valid"]
        idx_cls = torch.ones(
            (self.n_cls, bs),
            dtype=mask_valid.dtype,
            device=mask_valid.device,
        )
        mask_valid_all = torch.cat(
            [idx_cls, mask_valid], dim=0
        ).transpose(0, 1)
        src_key_padding_mask = mask_valid_all <= 0

        out_tf_enc = self.encoder_tf(inp, src_key_padding_mask=src_key_padding_mask)
        x = self.quant_linear(out_tf_enc)

        return x, src_key_padding_mask, out_tf_enc
