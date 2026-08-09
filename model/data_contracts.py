"""Hierarchy direction-reversal contract shared by model stages."""

from copy import deepcopy

import torch


def normalize_parent_eos(pair, sample_path=None):
    """Collapse duplicated negative parent sentinels to the final EOS token."""
    parent_indices = pair["code_idx_parent"]
    invalid_mask = parent_indices < 0
    n_invalid = int(invalid_mask.sum().item())
    if n_invalid <= 1:
        return pair

    valid_shape = (
        parent_indices.ndim == 3
        and parent_indices.shape[0] == 1
        and parent_indices.shape[2] == 1
    )
    if n_invalid != 2 or not valid_shape:
        location = f" for {sample_path}" if sample_path is not None else ""
        raise ValueError(
            f"Invalid duplicate parent EOS contract{location}: expected exactly "
            "two negative tokens in shape 1 x T x 1, got "
            f"count={n_invalid}, shape={tuple(parent_indices.shape)}"
        )

    keep_mask = (parent_indices >= 0)[0, :, 0]
    keep_mask[-1] = True
    for key, value in pair.items():
        if "_parent" in key:
            pair[key] = value[:, keep_mask]
    return pair


def reverse_child_parent_info(pair):
    """Swap child and parent tensors while preserving the parent EOS token."""
    reversed_pair = deepcopy(pair)

    for key in ("bb", "code", "code_idx"):
        reversed_pair[f"{key}_child"] = pair[f"{key}_parent"][:, :-1]
        eos = pair[f"{key}_parent"][:, -1:]
        reversed_pair[f"{key}_parent"] = torch.cat(
            [pair[f"{key}_child"], eos], dim=1
        )

    if "link_c2p" in pair:
        link_c2p = pair["link_c2p"]
        reversed_pair["link_c2p"] = torch.stack(
            [link_c2p[..., 1], link_c2p[..., 0]], dim=-1
        )

    if "link_patch2part_child" in pair:
        reversed_pair["link_patch2part_child"] = pair[
            "link_patch2part_parent"
        ]
        reversed_pair["link_patch2part_parent"] = pair[
            "link_patch2part_child"
        ]

    return reversed_pair
