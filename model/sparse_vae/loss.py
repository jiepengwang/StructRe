import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit_cls = nn.BCEWithLogitsLoss()

    def forward(self, sdf_pred, sdf_gt, out_cls, targets):
        loss_sdf = F.l1_loss(sdf_pred, sdf_gt)

        num_layers = len(out_cls)
        loss_occu = 0
        for out_cl, target in zip(out_cls, targets):
            cl_pred = out_cl.F.squeeze()
            cl_target = target.to(device=cl_pred.device, dtype=cl_pred.dtype)
            loss_occu += self.crit_cls(cl_pred, cl_target) / num_layers

        return loss_occu, loss_sdf
