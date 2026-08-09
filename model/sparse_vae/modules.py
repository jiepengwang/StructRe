""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from termcolor import cprint

from model.utils.utils_mink import ResnetConv3Mink
from model.utils.latent_regularization import KLReg



class ResidualSparseEncoder3D(nn.Module):
    def __init__(self, conf):
        super().__init__()
        enc_ch = list(conf['model.channels_encoder'])

        self.n_layers_resblock_enc = conf['model.n_layers_resblock_enc']
        self.k_pre_conv_enc = conf['model.k_pre_conv_enc']

        self.n_blocks_downsample = conf['model.n_blocks_downsample']

        enc_ch[self.n_blocks_downsample] *= 2
        enc_ch[self.n_blocks_downsample - 1] = 256

        # Encoder
        
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(
                3,
                enc_ch[0],
                kernel_size=3,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.lis_blocks_downsample = []
        self.lis_blocks_conv = []
        for i in range(self.n_blocks_downsample):
            d_inp = enc_ch[i]
            d_out = enc_ch[i+1]
            
            # 1/(2^i) -> 1/(2^(i+1))
            block_downsample_i = nn.Sequential(
                ME.MinkowskiConvolution(d_inp, d_out, kernel_size=2, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(d_out),
                ME.MinkowskiReLU(),
            )  
            self.lis_blocks_downsample.append(block_downsample_i)
            if i < self.n_blocks_downsample - 1:
                self.lis_blocks_conv.append(
                    ResnetConv3Mink(
                        d_out,
                        d_out,
                        d_out,
                        dimension=3,
                        n_layers_block=self.n_layers_resblock_enc,
                    )
                )

        self.lis_blocks_downsample = nn.ModuleList(self.lis_blocks_downsample)
        self.lis_blocks_conv = nn.ModuleList(self.lis_blocks_conv)

        d_out_final = enc_ch[self.n_blocks_downsample]
        self.conv_out = ME.MinkowskiConvolution(d_out_final, d_out_final, kernel_size=self.k_pre_conv_enc, dimension=3)
        cprint(f'[SparseEncoder3D] Dimension of final output: {d_out_final}', 'yellow')


        self.post_conv_reg = ME.MinkowskiConvolution(
            d_out_final, d_out_final // 2, kernel_size=1, dimension=3
        )
        for parameter in self.post_conv_reg.parameters():
            parameter.requires_grad = False
        self.regularizer = KLReg()

    def forward(self, data_input):
        x = data_input
        x = self.enc_block_s1(x)
        for i in range(self.n_blocks_downsample):
            x = self.lis_blocks_downsample[i](x)
            if i < self.n_blocks_downsample - 1:
                x = self.lis_blocks_conv[i](x)


        out = self.conv_out(x)
        
        feat_reg, loss_reg, _, _ = self.regularizer(
            out.F.clone(), sample_posterior=self.training
        )
        out = self.post_conv_reg(out)
        out.F[:] = feat_reg[:]
        return out, loss_reg          


class ResidualSparseDecoder3D(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        dec_ch = conf['model.channels_decoder']


        # Decoder
        self.n_blocks_downsample = conf['model.n_blocks_downsample']
        self.n_blocks_upsample = conf['model.n_blocks_upsample']
        
        # build blocks
        self.lis_blocks = []
        self.lis_cls = []
        for i in range(self.n_blocks_upsample):
            d_inp = dec_ch[self.n_blocks_downsample-i]
            d_out = dec_ch[self.n_blocks_downsample-i-1]
            block_i = nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(
                    d_inp, d_out, kernel_size=4, stride=2, dimension=3
                ),
                ME.MinkowskiBatchNorm(d_out),
                ME.MinkowskiReLU(),
                ResnetConv3Mink(d_out, d_out, d_out, dimension=3),
            )
            self.lis_blocks.append(block_i)

            cls_i = ME.MinkowskiConvolution(d_out, 1, kernel_size=1, bias=True, dimension=3)
            self.lis_cls.append(cls_i)

        self.lis_blocks = nn.ModuleList(self.lis_blocks)
        self.lis_cls = nn.ModuleList(self.lis_cls)

        d_out_final = dec_ch[self.n_blocks_downsample-self.n_blocks_upsample]
        self.conv_out = ME.MinkowskiConvolution(d_out_final, d_out_final, kernel_size=3, dimension=3)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for curr_in in kernel_map.values():
                target[curr_in[0].long()] = 1
        return target
        
    def forward(self, out_encoder, target_key):
        x = out_encoder
        lis_out_cls, lis_tgt_cls = [], []
        for i in range(self.n_blocks_upsample):
            # upsample and classfication
            x = self.lis_blocks[i](x)
            x_cls = self.lis_cls[i](x)
            tgt_cls = self.get_target(x, target_key)

            lis_out_cls.append(x_cls)
            lis_tgt_cls.append(tgt_cls)

            # pruning
            thres_prune = 0 if self.training else -0.5
            
            keep_i = (x_cls.F > thres_prune).squeeze()
            if self.training:
                keep_i += tgt_cls
            x = self.pruning(x, keep_i)

        x = self.conv_out(x)

        return lis_out_cls, lis_tgt_cls, x
