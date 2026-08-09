# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import torch
from torch import nn
from einops import repeat
import numpy as np




class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, conf):
        super().__init__()
        self.n_emd = conf['model.pe.n_emb']
        self.n_emd_half = self.n_emd // 2
        self.dim_emb = conf['model.pe.dim_emb']
        self.vox_size = conf['model.pe.bbox_size'] / float(self.n_emd-1)

        self.n_emb_rot = conf['model.pe.n_emb_rot']
        self.angle_interval = 2*np.pi/self.n_emb_rot
        
        self.emb_center = nn.Embedding(self.n_emd, self.dim_emb)
        self.emb_lwh = nn.Embedding(self.n_emd, self.dim_emb)
        self.emb_rot = nn.Embedding(self.n_emb_rot, self.dim_emb)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.emb_center.weight)
        nn.init.uniform_(self.emb_lwh.weight)
        nn.init.uniform_(self.emb_rot.weight)

    def forward(self, inp, rot_angle, use_emb_rot=False):
        '''
        paras:
            inp: t*b*c
        '''
        
        # center embedding
        inp_center = inp[...,:3]
        inp_center_vox = inp_center/self.vox_size + self.n_emd_half
        inp_center_vox = inp_center_vox.round().long().clamp(0, self.n_emd-1)
        inp_emb_center = self.emb_center(inp_center_vox)
                
        # lwh embedding
        inp_lwh = inp[...,3:]
        inp_lwh_vox = (inp_lwh/self.vox_size).round().long().clamp(0, self.n_emd-1)
        inp_emb_lwh = self.emb_lwh(inp_lwh_vox)
            
        if not use_emb_rot:
            inp_emb_all = torch.cat([inp_emb_center, inp_emb_lwh], dim=2)
        else:
            # rotation embedding
            rot_angle = rot_angle[..., None] + np.pi
            rot_angle_vox = rot_angle/self.angle_interval
            rot_angle_vox = rot_angle_vox.round().long().clamp(0, self.n_emb_rot-1)
            inp_emb_angle = self.emb_rot(rot_angle_vox)

            n_code = len(inp_emb_center)
            inp_emb_angle = repeat(inp_emb_angle, '1 b n c -> t b n c', t=n_code)

            inp_emb_all = torch.cat([inp_emb_center, inp_emb_lwh, inp_emb_angle], dim=2)

        inp_emb_flatten = inp_emb_all.flatten(-2,-1)
        return inp_emb_flatten


def build_position_encoding(conf):
    return PositionEmbeddingLearned(conf)
