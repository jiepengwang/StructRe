import numpy as np
import torch


@torch.no_grad()
def random_sampling_output(
    dict_out,
    dict_matcher,
    filter_strategy='rand_number',
    step_iter=0,
    lis_schedule=None,
):
    best_match_pred = dict_matcher['src_all_idx']
    best_match_tgt = dict_matcher['tgt_all_idx']
    tgt_merge_all = dict_matcher['tgt_merge_all']

    out_prob = dict_out['out_status'].detach().softmax(-1)[...,0]
    out_prob_match = out_prob[best_match_pred[0], best_match_pred[1]]

    prob_rand = torch.rand_like(out_prob_match)

    if filter_strategy == 'conf_filter':
        mask_attach = out_prob_match > prob_rand
    elif filter_strategy == 'rand_filter':
        mask_attach = prob_rand > 0.5
    elif filter_strategy == 'conf_schedule':
        thres_prob = lis_schedule[step_iter]
        mask_attach = out_prob_match > thres_prob
    elif filter_strategy == 'rand_number':
        n_match = len(out_prob_match)
        n_mask = np.random.choice(n_match, 1, replace=False)[0] + 1
        arr_idx_s = np.random.choice(n_match, n_mask, replace=False)

        mask_attach = torch.zeros_like(out_prob_match).bool()
        mask_attach[arr_idx_s] = True
    elif filter_strategy == 'use_all':
        mask_attach = torch.ones_like(out_prob_match).bool()
    else:
        raise ValueError(f'Unknown filter strategy: {filter_strategy}')

    idx_seq = best_match_pred[0][mask_attach]
    idx_bat = best_match_pred[1][mask_attach]
    idx_seq_tgt = tgt_merge_all[mask_attach]

    code_pred = dict_out['latent_tf_out'].detach()
    code_attach = torch.zeros_like(code_pred)
    code_attach_valid = code_pred[idx_seq, idx_bat]
    code_attach[idx_seq, idx_bat] += code_attach_valid

    dict_attach = {
        'best_match_pred_v0': best_match_pred,
        'best_match_tgt_v0': best_match_tgt,
        'code_attach': code_attach,
        'idx_attach_seq': idx_seq,
        'idx_attach_seq_tgt': idx_seq_tgt,
        'idx_attach_bat': idx_bat,
        'mask_attach': mask_attach,
    }

    return dict_attach


def random_sampling_output_infer(
    dict_out,
    sampling_strategy='rand_prob',
    step_iter=0,
    mask_prev=None,
    filter_strategy='rand_prob',
    lis_schedule=None,
):
    out_prob = dict_out['out_status'].detach().softmax(-1)[...,0]
    if sampling_strategy == 'rand_prob':
        prob_rand = torch.rand_like(out_prob)
        mask_sample = prob_rand > 0.5
        mask_filter = out_prob > 0.2

        mask_attach = torch.logical_and(mask_sample, mask_filter)

    elif sampling_strategy == 'topk_schedule':
        prob_rand = torch.rand_like(out_prob)
        thres_prob_schedule = lis_schedule[step_iter]

        if filter_strategy == 'conf_filter':
            mask_sample = out_prob > prob_rand
        elif filter_strategy in [
            'conf_filter_inside',
            'conf_filter_inside_accum',
        ]:
            mask_sample = out_prob > (
                prob_rand * (1 - thres_prob_schedule)
                + thres_prob_schedule
            )
        elif filter_strategy == 'rand_filter':
            mask_sample = prob_rand > 0.5
        elif filter_strategy == 'conf_schedule':
            mask_sample = out_prob > thres_prob_schedule
        else:
            raise ValueError(f'Unknown filter strategy: {filter_strategy}')

        mask_filter = out_prob > thres_prob_schedule
        mask_attach = torch.logical_and(mask_sample, mask_filter)

        if mask_prev is not None:
            mask_prev = torch.logical_or(mask_attach, mask_prev)
            mask_attach = mask_prev

        if (
            filter_strategy in ['conf_filter_inside_accum', 'rand_filter']
            and mask_prev is None
        ):
            mask_prev = mask_attach
    else:
        raise ValueError(f'Unknown sampling strategy: {sampling_strategy}')

    code_pred = dict_out['latent_tf_out'].detach()
    code_attach = torch.zeros_like(code_pred)

    code_select = code_pred[mask_attach]
    code_attach[mask_attach] += code_select

    dict_attach = {
        'code_attach': code_attach,
        'mask_attach': mask_attach,
    }

    return dict_attach, mask_prev, {}
