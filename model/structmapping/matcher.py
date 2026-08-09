# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import numpy as np
from einops import repeat



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, conf):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the structure-code regression error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.conf = conf
        self.cost_status = self.conf['train.loss.weight_status']
        self.cost_class = self.conf['train.loss.weight_cls']
        self.cost_bbox = self.conf['train.loss.weight_bbox']
        if (
            self.cost_class == 0
            and self.cost_bbox == 0
            and self.cost_status == 0
        ):
            raise ValueError(
                "StructMapping matcher requires at least one non-zero cost; "
                f"got status={self.cost_status}, class={self.cost_class}, "
                f"bbox={self.cost_bbox}"
            )


        self.reg_loss_type = self.conf["train.loss.reg_loss_type"]
        if self.reg_loss_type == "l2":
            self.pnorm = 2
        elif self.reg_loss_type == "l1":
            self.pnorm = 1
        else:
            raise ValueError(
                f"Unsupported matcher regression loss: {self.reg_loss_type}"
            )
            
    @torch.no_grad()
    def forward(self, outputs, targets, dict_attach=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "out_class": Tensor of regression structure codes
                 "out_bbox": Tensor of predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        output_class = outputs["out_class"]
        if output_class.ndim != 4:
            raise ValueError(
                "StructMapping matcher expects out_class shape [T, B, N, C], "
                f"got {tuple(output_class.shape)}"
            )
        num_queries, bs = output_class.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_class = outputs["out_class"].detach().flatten(0, 1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["out_bbox"].detach().flatten(0, 1)  # [batch_size * num_queries, 4]

        code_idx_parent = targets['code_idx_parent']  # 20
        code_parent = targets['code_parent']  # 20
        bb_parent = targets['bb_parent']  # 20*4
        if 'bb_rot' in targets:
            n_tgt = len(bb_parent)
            bb_rot_tgt = targets['bb_rot']
            bb_rot_tgt = repeat(bb_rot_tgt, '1 b c -> n b c', n=n_tgt)
            bb_parent = torch.cat([bb_parent, bb_rot_tgt], dim=2)    

        mask_valid_tgt = torch.logical_and(
            code_idx_parent > -1,
            code_idx_parent < self.conf["model.num_emb_struct"],
        )
        if mask_valid_tgt.ndim == 3:
            if mask_valid_tgt.shape[-1] != 1:
                raise ValueError(
                    "StructMapping matcher expects one target index per token, got "
                    f"shape {tuple(code_idx_parent.shape)}"
                )
            mask_valid_tgt = mask_valid_tgt[..., 0]
        elif mask_valid_tgt.ndim != 2:
            raise ValueError(
                f"Unsupported target index shape: {tuple(code_idx_parent.shape)}"
            )
        
        tgt_bbox, tgt_code = [], []
        for i in range(bs):
            mask_valid_i = mask_valid_tgt[:, i]

            tgt_code_i = code_parent[:, i][mask_valid_i]  # 20
            tgt_bbox_i = bb_parent[:, i][mask_valid_i] # 20*4
            
            tgt_code.append(tgt_code_i)
            tgt_bbox.append(tgt_bbox_i)

        tgt_code = torch.cat(tgt_code, dim=0)
        tgt_bbox = torch.cat(tgt_bbox, dim=0)
        if len(tgt_code) == 0:
            raise ValueError(
                "StructMapping sample has no valid targets: {}".format(
                    targets.get("path")
                )
            )
        
        if outputs["out_status"] is None:
            raise ValueError("StructMapping matcher requires out_status")
        num_valid = len(tgt_code)
        prob_status = outputs["out_status"].detach().flatten(0, 1).softmax(-1)
        cost_status = -torch.log(prob_status[:, :1] + 1e-6)
        cost_status = cost_status.expand(-1, num_valid)
        # bbox cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=self.pnorm)
        d_bbox = out_bbox.shape[1]
        if self.pnorm == 2:
            cost_bbox = cost_bbox.square()
        cost_bbox = cost_bbox / d_bbox
        # code cost
        cost_class = torch.cdist(
            out_class.flatten(1, 2),
            tgt_code.flatten(1, 2),
            p=self.pnorm,
        )
        d_code = out_class.shape[-1]
        if self.pnorm == 2:
            cost_class = cost_class.square()
        cost_class = cost_class / d_code

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_status * cost_status
        C = C.view(num_queries, bs, -1).cpu()

        if dict_attach is not None:
            # update cost matrix
            idx_attach_bat = dict_attach['idx_attach_bat']
            idx_attach_seq = dict_attach['idx_attach_seq']
            idx_attach_seq_tgt = dict_attach['idx_attach_seq_tgt']
            C[idx_attach_seq, idx_attach_bat, idx_attach_seq_tgt] = -1


        sizes = mask_valid_tgt.sum(dim=0).tolist()
        C1 = C.split(sizes, -1)

        indices = [linear_sum_assignment(c[:, i]) for i, c in enumerate(C1)]


        src_all, tgt_all, bat_all = [], [], []
        tgt_merge_all = []
        n_tgt_curr = 0
        for idx_b in range(len(indices)):
            src, tgt = indices[idx_b]
            bat = np.ones_like(src, dtype=int)*idx_b

            src_all.append(src)
            tgt_all.append(tgt)
            bat_all.append(bat)

            idx_merge_tgt = np.ones_like(src, dtype=int)*n_tgt_curr
            tgt_merge_all.append(idx_merge_tgt+tgt)
            
            n_tgt_curr += sizes[idx_b]
            if sizes[idx_b] != len(bat):
                raise RuntimeError(
                    f"Failed to match all targets for: {targets['path']}"
                )

        device = out_class.device
        src_all = torch.as_tensor(np.concatenate(src_all), device=device)
        tgt_all = torch.as_tensor(np.concatenate(tgt_all), device=device)
        bat_all = torch.as_tensor(np.concatenate(bat_all), device=device)
        tgt_merge_all = torch.as_tensor(
            np.concatenate(tgt_merge_all), device=device
        )

        src_all_idx = (src_all, bat_all)
        tgt_all_idx = (tgt_all, bat_all)
            
        logs_matcher = {
            'logs/matcher_cost_cls': cost_class.mean().detach().item(),
            'logs/matcher_cost_cls_min': cost_class.min().detach().item(),
            
            'logs/matcher_cost_bbox': cost_bbox.mean().detach().item(),
            'logs/matcher_cost_bbox_min': cost_bbox.min().detach().item(),
            
            'logs/matcher_cost_status': cost_status.mean().detach().item() if torch.is_tensor(cost_status) else cost_status,
            'logs/matcher_cost_status_min': cost_status.min().detach().item() if torch.is_tensor(cost_status) else cost_status
        }

        dict_matcher = {
            'src_all_idx': src_all_idx,
            'tgt_all_idx': tgt_all_idx,
            'logs_matcher': logs_matcher,
            'tgt_merge_all': tgt_merge_all
        }
        return dict_matcher

def build_matcher(conf):
    return HungarianMatcher(conf)
