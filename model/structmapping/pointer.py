import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.utils_mlp import MLP

from einops import repeat


class Pointer(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf

        

        d_latent = self.conf['model.decoder.embed_dim']
        conf_mlp_cls={
            'input_dim': d_latent, 
            'hidden_dim': self.conf['model.pointer.d_hidden'], 
            'output_dim': self.conf['model.pointer.d_output'],
            'num_layers': self.conf['model.pointer.n_layers']
        }
        self.emb_latent = MLP(**conf_mlp_cls)

    def calculate_pointer_loss_part2part(self, dict_inp, latent_tf_inp, latent_tf_out, mask_invalid_tf_inp):

        T, B, _ = latent_tf_inp.shape[:3]
        device = latent_tf_inp.device
        N2 = len(dict_inp['bb_parent'][:-1]) # remove last token

        # 1. mask of vaild link pair, for adjacent matrix
        link_c2p = dict_inp['link_c2p'].long()
        link_c2p_c = link_c2p[..., 0] # T*B, c
        link_c2p_p = link_c2p[..., 1] # T*B, p
        _, B_link, _ = link_c2p.shape
        
        idx_b = torch.arange(B_link, device=device).reshape(1, B_link)
        idx_b = idx_b.expand_as(link_c2p_p)

        mask_valid_link_c2p = (link_c2p_p>=0)
        link_c2p_p_valid = link_c2p_p[mask_valid_link_c2p]
        link_c2p_c_valid = link_c2p_c[mask_valid_link_c2p]
        link_c2p_b_valid = idx_b[mask_valid_link_c2p]


        # 2. mask of valid input location
        mask_valid_match = torch.zeros(B, N2, device=device)  # T*B*N
        mask_valid_match[link_c2p_b_valid, link_c2p_p_valid] = 1.0
        mask_valid_match = repeat(mask_valid_match, 'b n -> t b n', t=T).clone()
        
        mask_valid_match[mask_invalid_tf_inp] = 0.0  # set invalid locations to 0 along T
        mask_valid_match = mask_valid_match.bool() # T*B*N

        # 3. weight for each pair, self.training
        empty_weight = 0.2
        mat_weight_pair = torch.ones(T, B, N2, device=device)*empty_weight
        if B == 1 and len(link_c2p_c_valid) != T:
            print('[Pointer] no valid link. [Loss calculation]')
            return latent_tf_inp.sum() * 0.0, {}
        
        mat_weight_pair[link_c2p_c_valid, link_c2p_b_valid, link_c2p_p_valid] = 1.0
        mat_weight_pair[mask_valid_match==False] = 0.0 # T*B*N

        # 4. gt for each pair
        mat_prob_gt_pair = torch.zeros(T, B, N2, device=device)
        mat_prob_gt_pair[link_c2p_c_valid, link_c2p_b_valid, link_c2p_p_valid] = 1.0


        # 5. calculate attention
        latent_tf_inp_emb = self.emb_latent(latent_tf_inp)
        latent_tf_out_reordered_emb = self.emb_latent(latent_tf_out)

        latent_tf_inp_emb = latent_tf_inp_emb.permute(1,0,2) # T*B*D -> B*T*D
        latent_tf_out_reordered_emb = latent_tf_out_reordered_emb.permute(1,2,0)  # N*B*D -> B*D*N
        
        prob_link = torch.matmul(latent_tf_inp_emb, latent_tf_out_reordered_emb) # B*T*N
        prob_link = torch.sigmoid(prob_link).transpose(0,1) # B*T*N -> T*B*N
        
        # 6. calculate loss at valid locations
        prob_pred_valid = prob_link[mask_valid_match]
        weight_pair_valid = mat_weight_pair[mask_valid_match]
        prob_gt_valid = mat_prob_gt_pair[mask_valid_match]

        if prob_pred_valid.numel() == 0:
            return prob_link.sum() * 0.0, {}
        loss_pointer = F.binary_cross_entropy(prob_pred_valid, prob_gt_valid, reduction='none')
        loss_pointer = (loss_pointer * weight_pair_valid).mean()

        # 7. calculate accuracy
        dict_logs_pointer = self.get_logs_pointer(
            loss_pointer.detach(),
            prob_pred_valid.detach(),
            prob_gt_valid,
            mode_pointer='1_part2part',
            prob_pred_all=prob_link.detach(),
        )
        phase = 'train' if self.training else 'test' 
        dict_logs_pointer.update({
            f'loss/loss_pointer_{phase}': loss_pointer.detach().item(),
        })            
        return loss_pointer, dict_logs_pointer

    def get_logs_pointer(self, loss_pointer, prob_pred_valid, prob_gt_valid, mode_pointer, prob_pred_all):
        mask_pair_pred = prob_pred_valid.detach()>=0.5
        mask_pair_gt = prob_gt_valid>=0.5
        
        mask_pair_pred_all = prob_pred_all>=0.5
        mask_pred_status_all = torch.logical_and(
            mask_pair_pred == mask_pair_gt,
            mask_pair_pred,
        )
        

        # F-score
        n_all_pred = mask_pair_pred_all.sum()
            
        n_valid_gt = mask_pair_gt.sum()
        
        n_overlap = mask_pred_status_all.sum()
        
        prec = n_overlap / (n_all_pred + 1e-5)
        recall = n_overlap / (n_valid_gt + 1e-5)
        f_score = 2*prec*recall / (prec+recall+1e-5)
        
        phase = 'train' if self.training else 'test'  
        dict_logs_pointer = {
            f'pointer/{mode_pointer}_loss_{phase}': loss_pointer.detach().item(),
            f'pointer/{mode_pointer}_precision_{phase}': prec.item(),
            f'pointer/{mode_pointer}_recall_{phase}': recall.item(),
            f'pointer/{mode_pointer}_fscore_{phase}': f_score.item(),

        }
        
        return dict_logs_pointer
    

    def forward(self, dict_inp, dict_out, best_match_pred, best_match_tgt):
        
        # input info
        latent_tf_inp_all = dict_out['latent_tf_inp']  # T*B*D
        T_seq_part = len(dict_inp['bb_child'])
        latent_tf_inp = latent_tf_inp_all[:T_seq_part] # :T_seq_part, patch code
        if T_seq_part==0 or ('link_c2p' not in dict_inp): # or (not self.training):
            return latent_tf_inp_all.sum() * 0.0, {}

        mask_invalid_tf_inp = dict_inp['code_idx_child'].squeeze(-1)<0

        _, batch_size, latent_dim = latent_tf_inp.shape
        n_parent = len(dict_inp["bb_parent"][:-1])

        target_sequence, target_batch = best_match_tgt
        prediction_sequence, prediction_batch = best_match_pred
        sort_key = target_batch * max(n_parent, 1) + target_sequence
        match_order = torch.argsort(sort_key)

        target_sequence = target_sequence[match_order]
        target_batch = target_batch[match_order]
        prediction_sequence = prediction_sequence[match_order]
        prediction_batch = prediction_batch[match_order]
        latent_tf_out_valid = dict_out["latent_tf_out"][
            prediction_sequence, prediction_batch
        ]

        latent_tf_out_reordered = torch.zeros(
            n_parent,
            batch_size,
            latent_dim,
            device=latent_tf_inp.device,
            dtype=latent_tf_inp.dtype,
        )
        latent_tf_out_reordered[target_sequence, target_batch] = (
            latent_tf_out_valid
        )

        loss_pointer, dict_logs_pointer = self.calculate_pointer_loss_part2part(dict_inp, latent_tf_inp, 
                                                                                                latent_tf_out_reordered,
                                                                                                mask_invalid_tf_inp)
        

        return loss_pointer, dict_logs_pointer
