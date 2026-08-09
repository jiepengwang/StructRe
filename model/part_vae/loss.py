import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat


class PartVAELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

        # loss
        self.func_loss_code = F.mse_loss

        self.recon_weight = conf['train.recon_weight']
        self.status_weight  = conf['train.status_weight']

        empty_weight = torch.ones(2)
        empty_weight[-1] = conf['train.empty_weight']
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, data_dict_inp, data_dict_out):
        phase = 'train' if self.training else 'test'
        logs = {}
        
        code_gt = data_dict_inp['code'].detach()
        
        code_pred = data_dict_out['out_code'] # T*B*C

        T, B = code_pred.shape[:2]
        part_level_weight = data_dict_inp['part_level_weight']

        # status loss
        status_cls = data_dict_out['out_status']  # T*B*C
        mask_nonempty = data_dict_inp['mask_nonempty']
        target_status = torch.ones_like(mask_nonempty).long()
        target_status[mask_nonempty] = 0

        status_cls_flatten = rearrange(status_cls, 't b c -> (t b) c')
        loss_status = F.cross_entropy(
            status_cls_flatten,
            target_status.reshape(-1),
            self.empty_weight,
            reduction='none',
        )
        logs[f'loss/loss_status_{phase}'] = loss_status.detach().mean().item()

        loss_status = loss_status.reshape(T, B)
        loss_status = loss_status * part_level_weight[None, ...]
        loss_status = loss_status.mean()

        status_cls_max = torch.argmax(status_cls, dim=-1)
        mask_status_right = status_cls_max == target_status
        accu_status = mask_status_right.sum() / mask_status_right.numel()
        logs[f'logs/accu_status_{phase}'] = accu_status.detach().item()
                                
        # code loss
        mask_nonempty = data_dict_inp['mask_nonempty']
        code_pred_nonempty = (code_pred.transpose(0,1)[mask_nonempty.transpose(0,1)])

        mask_valid = data_dict_inp['mask_valid']>0
        code_gt_nonempty = (code_gt.transpose(0,1)[mask_valid.transpose(0,1)])
        code_gt_nonempty = code_gt_nonempty.contiguous()

        err_code_valid = F.l1_loss(code_pred_nonempty.detach(), code_gt_nonempty)
        loss_code_valid = self.func_loss_code(code_pred_nonempty, code_gt_nonempty, reduction='none')
        logs.update({
            f'logs/err_code_{phase}': err_code_valid.item(),
            f'loss/loss_code_{phase}': loss_code_valid.detach().mean().item(),
        })

        part_level_weight_nonempty = repeat(part_level_weight, 'b -> t b', t=T)
        part_level_weight_nonempty = (part_level_weight_nonempty.transpose(0,1)[mask_nonempty.transpose(0,1)])

        loss_code_valid = loss_code_valid.mean(dim=-1)
        loss_code_valid = loss_code_valid * part_level_weight_nonempty

        loss_code = loss_code_valid.mean()
    
        loss_all = self.status_weight*loss_status + \
                    self.recon_weight*loss_code

        return loss_all, logs
