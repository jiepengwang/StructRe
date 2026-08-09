'''Heavily borrowed from detr
'''
import torch
from torch import nn
import torch.nn.functional as F

from termcolor import cprint
from einops import rearrange, repeat
from model.structmapping.pointer import Pointer

import model.structmapping.utils_model.utils_model_io as ModelIOUtils


class StructLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.weight_pointer  = self.conf['train.loss.weight_pointer']

        self.n_query = self.conf['model.decoder.detr.num_queries']
        self.register_buffer(
            "counter_query",
            torch.zeros(self.n_query),
            persistent=False,
        )
        self.counter_min = self.conf['train.counter_min']

        self.thres_prob_valid_part = self.conf['train.thres_prob_valid_part']

        self.pointer = Pointer(self.conf)

        self.eos_coef = self.conf['model.decoder.detr.eos_coef']
        print(f'Empty weight for DETR optimization: {self.eos_coef}')
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.weight_bbox = self.conf['train.loss.weight_bbox']
        self.weight_cls = self.conf['train.loss.weight_cls']
        self.weight_status = self.conf['train.loss.weight_status']
        

        self.reg_loss_type = self.conf['train.loss.reg_loss_type']
        if self.reg_loss_type == "l2":
            self.reg_loss = F.mse_loss
            cprint("Use L2 loss", "yellow")
        elif self.reg_loss_type == "l1":
            self.reg_loss = F.l1_loss
            cprint("Use L1 loss", "yellow")
        else:
            raise ValueError(
                f"Unsupported StructMapping regression loss: {self.reg_loss_type}"
            )
            
    def forward(self, dict_out, dict_inp, dict_matcher):
        loss_all, logs_loss = self.forward_detr_v2(dict_out, dict_inp, dict_matcher)
        return loss_all, logs_loss
    
    def forward_detr_v2(self, dict_out, dict_inp, dict_matcher):
        logs = {}
        phase = 'train' if self.training else 'test'                 

        best_match_pred, best_match_tgt = dict_matcher['src_all_idx'], dict_matcher['tgt_all_idx']

        if self.training:
            # count query
            _, bs = dict_out['out_status'].shape[:2]
            stat_query = torch.zeros(
                (self.n_query, bs), device=dict_out["out_status"].device
            )
            stat_query[best_match_pred[0], best_match_pred[1]] = 1.0
            stat_query_count = stat_query.sum(dim=-1)

            self.counter_query += stat_query_count
            self.counter_query.clamp_(0, 8192)
            n_dead = (self.counter_query<self.counter_min).sum()
            logs.update({
                f'logs/query_n_dead_{phase}': n_dead.item()         
            })
        
        # 1. status loss
        status_cls = dict_out['out_status']  # T*B*C
        target_status = torch.full(
            status_cls.shape[:2],
            1,
            dtype=torch.int64,
            device=status_cls.device,
        )
        target_status[best_match_pred] = 0

        status_cls_flatten = rearrange(status_cls, 't b c -> (t b) c')
        if self.eos_coef == 1.0:
            loss_status = F.cross_entropy(
                status_cls_flatten,
                target_status.reshape(-1),
                reduction='none',
            )
        else:
            loss_status = F.cross_entropy(
                status_cls_flatten,
                target_status.reshape(-1),
                self.empty_weight,
                reduction='none',
            )
        loss_status = loss_status.mean()

        status_cls_match = status_cls.detach()[best_match_pred]
        status_cls_match_max = status_cls_match.softmax(-1)[..., 0]
        status_pred = status_cls.detach().softmax(-1)[..., 0]
        thres_valid = self.thres_prob_valid_part
        mask_status_right = status_cls_match_max > thres_valid
        n_nonempty_matcher = mask_status_right.sum().item()
        n_gt = mask_status_right.numel()
        accu_recall = n_nonempty_matcher / max(n_gt, 1)

        mask_status_all = status_pred > thres_valid
        n_nonempty_all = mask_status_all.sum().item()
        accu_prec = n_nonempty_matcher / (n_nonempty_all + 1e-6)
        fscore = (
            2 * accu_prec * accu_recall
            / (accu_prec + accu_recall + 1e-6)
        )
        logs.update({
            f'logs/accu_precision_{phase}': accu_prec,
            f'logs/accu_recall_{phase}': accu_recall,
            f'logs/accu_fscore_{phase}': fscore,
            f'loss/loss_status_{phase}': loss_status.detach().item(),
        })
        # 2. code loss
        cls_pred = dict_out['out_class']
        cls_pred_match = cls_pred[best_match_pred]
        cls_tgt = dict_inp['code_parent']
        cls_tgt_match = cls_tgt[best_match_tgt]

        loss_cls = self.reg_loss(cls_pred_match, cls_tgt_match, reduction='mean')

        code_pred_valid = cls_pred_match.detach()
        err_code_mean = F.l1_loss(code_pred_valid, cls_tgt_match)
        logs.update({
            f'logs/1_code_gt_abs_mean_{phase}': cls_tgt_match.abs().mean().item(),
            f'logs/1_code_pred_abs_mean_{phase}': code_pred_valid.abs().mean().item(),
            f'logs/1_err_code_mean_{phase}': err_code_mean.item(),
        })

        # 3. bbox loss
        bb_pred = dict_out['out_bbox']
        bb_pred_match = bb_pred[best_match_pred]

        bb_tgt = dict_inp['bb_parent']
        n_tgt = len(bb_tgt)
        bb_rot_tgt = repeat(dict_inp['bb_rot'], '1 b c -> n b c', n=n_tgt)
        bb_tgt = torch.cat([bb_tgt, bb_rot_tgt], dim=2)
        bb_tgt_match = bb_tgt[best_match_tgt]

        loss_bbox = self.reg_loss(bb_pred_match, bb_tgt_match, reduction='none')
        loss_bb_center = loss_bbox.detach()[..., :3]
        loss_bb_lwh = loss_bbox.detach()[..., 3:6]
        loss_bb_rot = loss_bbox.detach()[..., 6:]

        angle_pred = ModelIOUtils.reverse_cossin_to_angle(
            bb_pred_match.detach()[..., 6:]
        )
        angle_gt = ModelIOUtils.reverse_cossin_to_angle(bb_tgt_match[..., 6:])
        err_angle = F.l1_loss(angle_pred, angle_gt)

        err_bbox = F.l1_loss(
            bb_pred_match.detach(), bb_tgt_match, reduction='none'
        )
        err_bb_center = err_bbox[..., :3]
        err_bb_lwh = err_bbox[..., 3:6]
        logs.update({
            f'loss/loss_bbox_center_{phase}': loss_bb_center.mean().item(),
            f'loss/loss_bbox_lwh_{phase}': loss_bb_lwh.mean().item(),
            f'loss/loss_bbox_rot_{phase}': loss_bb_rot.mean().item(),
            f'logs/0_err_bbox_center_{phase}': err_bb_center.mean().item(),
            f'logs/0_err_bbox_lwh_{phase}': err_bb_lwh.mean().item(),
            f'logs/0_err_bbox_rot_{phase}': err_angle.item(),
        })
        loss_bbox = loss_bbox.mean()

        # pointer loss
        loss_pointer, dict_logs_pointer = self.pointer(
            dict_inp, dict_out, best_match_pred, best_match_tgt
        )
        logs.update(dict_logs_pointer)

        # loss_all
        loss_all = self.weight_bbox * loss_bbox + \
                    self.weight_cls * loss_cls + \
                    self.weight_status * loss_status + \
                    self.weight_pointer * loss_pointer

                        
        logs.update({
            f'loss/loss_all_{phase}': loss_all.detach().item(),
            f'loss/loss_code_{phase}': loss_cls.detach().item(),
            f'loss/loss_bbox_all_{phase}':loss_bbox.detach().item()
        })

        return loss_all, logs
