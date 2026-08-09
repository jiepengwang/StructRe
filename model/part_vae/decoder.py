# Code for Transformer module
# heavily borrowed from AutoSDF RandTransformer

import torch
import torch.nn as nn

from einops import repeat

from model.utils.utils_transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from model.utils.utils_mlp import MLP

class StructDecoder(nn.Module):
    def __init__(self, conf_part_vae, conf_sparse_vae):
        super().__init__()

        # SparseVAE
        sparse_latent_dim = conf_sparse_vae['model.latent.dim']

        # transformer embedding
        mparam = conf_part_vae['model.decoder']
        d_tf = mparam['embed_dim']
        nhead = mparam['nhead']
        num_decoder_layers = mparam['nlayers_enc']
        dim_feedforward = mparam['d_hid']
        dropout = mparam['dropout']
        normalize_before = mparam['normalize_before']
        activation = mparam['activation']

        self.query_embed = nn.Embedding(conf_part_vae['model.decoder.seq_len'], d_tf)

        # status branch
        self.status_decoder = MLP(
            input_dim=d_tf,
            hidden_dim=conf_part_vae["model.decoder.status_decoder.d_hidden"],
            output_dim=2,
            num_layers=conf_part_vae["model.decoder.status_decoder.n_layers"],
        )

        # code branch
        self.code_decoder = MLP(
            input_dim=d_tf,
            hidden_dim=conf_part_vae["model.decoder.code_decoder.d_hidden"],
            output_dim=sparse_latent_dim,
            num_layers=conf_part_vae["model.decoder.code_decoder.n_layers"],
        )

        # transformer
        decoder_layer = TransformerDecoderLayer(
            d_tf, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_tf)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self.post_quant_linear = nn.Linear(
            conf_part_vae["model.latent.dim"], d_tf
        )

        self._init_weights()

    def _init_weights(self):
        """initialize the weights of params."""
        self.post_quant_linear.bias.data.normal_(0, 0.02)
        self.post_quant_linear.weight.data.normal_(0, 0.02)

    def forward(
        self, memory, memory_key_padding_mask=None, memory_inp=None,
    ):
        _, bs = memory.shape[:2]

        inp_query = self.query_embed.weight
        inp_query = repeat(inp_query, 't pos_d -> t bs pos_d', bs=bs)

        memory = self.post_quant_linear(memory)
        if memory_inp is not None:
            memory = torch.cat([memory, memory_inp])
        out_tf = self.decoder(
            inp_query, memory, memory_key_padding_mask=memory_key_padding_mask
        )[0]

        out_code = self.code_decoder(out_tf)
        out_status = self.status_decoder(out_tf)

        return {"out_code": out_code, "out_status": out_status}
