# Code for Transformer module
# heavily borrowed from AutoSDF RandTransformer

import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

from model.utils.utils_transformer import TransformerDecoderLayer, TransformerDecoder


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MappingDecoder(nn.Module):
    def __init__(self, conf):
        super(MappingDecoder, self).__init__()

        self.conf = conf

        # transformer embedding
        mparam = conf['model.decoder']
        d_tf = mparam['embed_dim']
        nhead = mparam['nhead']
        num_decoder_layers = mparam['nlayers_enc']
        dim_feedforward = mparam['d_hid']
        dropout = mparam['dropout']
        normalize_before = mparam['normalize_before']
        activation = mparam['activation']
        

        
        # 2. transformer
        decoder_layer = TransformerDecoderLayer(d_tf, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_tf)
        self.transformer = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        num_queries_detr = self.conf['model.decoder.detr.num_queries']

        self.query_emb_detr = nn.Embedding(num_queries_detr, d_tf)
            
        
        # 3. output branch: code, bbox, stauts
        self.init_cls_bbox_network_general()

            
        self._init_weights()
   
    
    def init_cls_bbox_network_general(self):
        d_tf = self.conf['model.decoder.embed_dim'] 
        d_struct_code= self.conf['model.dim_emb_code']

        # 1. status branch
        d_status = 2
        self.status_emb = MLP(input_dim=d_tf, hidden_dim=self.conf['model.decoder.status.d_hidden'], 
                                                output_dim=d_status, 
                                                num_layers=self.conf['model.decoder.status.n_layers']) 
        
        
        # 2. struct code branch
        conf_mlp_cls={
            'input_dim': d_tf,
            'hidden_dim': self.conf['model.decoder.cls.d_hidden'],
            'output_dim': d_struct_code,
            'num_layers': self.conf['model.decoder.cls.n_layers']
        }
        self.class_embed = MLP(**conf_mlp_cls)

        # 3. bbox branch           
        d_out_bbox = self.conf['model.decoder.bbox.d_output']
        self.bbox_embed = MLP(input_dim=d_tf, 
                                hidden_dim=self.conf['model.decoder.bbox.d_hidden'], 
                                output_dim=d_out_bbox, 
                                num_layers=self.conf['model.decoder.bbox.n_layers'])
        
    def _init_weights(self):
        """initialize the weights of params."""


        for layer in self.bbox_embed.layers:
            layer.bias.data.normal_(0, 0.02)
            layer.weight.data.normal_(0, 0.02)           
        

    
    def set_input(self, data_dict_inp):
        bb_parent = data_dict_inp['bb_parent']     # B*T*C  
        B = bb_parent.shape[1]

        inp_query = self.query_emb_detr.weight

        inp_tf = repeat(inp_query, 't c -> t b c', b = B)
            
        return inp_tf

    def forward(self, inp_tf, memory, memory_key_padding_mask=None,
                        dict_attach=None):
        '''
        Args:
            inp: 1+T
        '''

        if dict_attach is not None:
            inp_tf = inp_tf + dict_attach["code_attach"].detach()
                
        outp = self.transformer(
            inp_tf,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )[0]

        out_class = self.class_embed(outp)
        out_class = repeat(out_class, 't b c -> t b n c', n=1) # t b 1 c
            
        out_bbox = self.bbox_embed(outp)
        out_status = self.status_emb(outp)
        
        # normalize cos and sin
        if out_bbox.shape[-1] == 8:
            bb_rot_pred_cos = out_bbox[..., 6:7].detach()
            bb_rot_pred_sin = out_bbox[..., 7:].detach()
            norm_rot = (bb_rot_pred_cos.square() + bb_rot_pred_sin.square()).sqrt()
            out_bbox[..., 6:] = out_bbox[..., 6:]/(norm_rot+1e-6)
           
        dict_out = {
            'out_class': out_class,
            'out_bbox': out_bbox,
            'out_status': out_status, # empty or not
        }

        dict_out['latent_tf_out'] = outp

        dict_out['input_tf_decoder'] = inp_tf
        return dict_out
