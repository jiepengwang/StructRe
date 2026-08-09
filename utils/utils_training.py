import torch
import numpy as np


def get_tensor_statistics(inp_tensor, header = '', decimals = 3, tag='train'):
    inp_max = round(inp_tensor.max().item(), decimals)
    inp_min = round(inp_tensor.min().item(), decimals)
    inp_mean_o = inp_tensor.mean().item()
    inp_mean = round(inp_tensor.mean().item(), decimals)
    inp_mean_abs = round(inp_tensor.abs().mean().item(), decimals)
    inp_sum = round(inp_tensor.sum().item(), decimals)

    num_le_mean = (inp_tensor<inp_mean_o).sum()
    num_zero = (inp_tensor==0).sum()

    msg_stat = f'{header} (min, max): {inp_min, inp_max}. (abs_mean, mean, sum): {inp_mean_abs, inp_mean, inp_sum}. <mean: {num_le_mean}/{inp_tensor.numel()}. num_zero: {num_zero}'
    dict_info = {
        f'{tag}_min': inp_min,
        f'{tag}_max': inp_max,
        f'{tag}_mean': inp_mean,
        f'{tag}_mean_abs': inp_mean_abs,
    }
    return msg_stat, dict_info


def get_tensor_statistics_fast(inp_tensor, prefix='', phase='train'):
    inp_max = inp_tensor.max().item()
    inp_min = inp_tensor.min().item()
    inp_mean = inp_tensor.mean().item()
    inp_mean_abs = inp_tensor.abs().mean().item()

    dict_info = {
        f'{prefix}min_{phase}': inp_min,
        f'{prefix}max_{phase}': inp_max,
        f'{prefix}mean_{phase}': inp_mean,
        f'{prefix}mean_abs_{phase}': inp_mean_abs,
    }
    return dict_info


def unfold_indices_to_grid(n_indices, n_reso, use_tensor=True):
    indices= np.arange(n_indices)
    
    Z = indices//(n_reso**2)
    id_box_xy = indices%(n_reso**2)
    
    Y = id_box_xy//n_reso
    X = id_box_xy%n_reso
    
    coords = np.stack([X, Y, Z],axis=-1)
    if use_tensor:
        coords = torch.from_numpy(coords).long()
    return coords
