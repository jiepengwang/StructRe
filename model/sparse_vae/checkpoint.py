"""Strict Stage 1 checkpoint composition with legacy-state filtering."""

LEGACY_SPARSE_VAE_STATE_PREFIXES = (
    "layer_norm",
    "sdf_decoder.fc_feat",
    "sdf_decoder.fc_coord",
)


def _is_legacy_sparse_vae_state_key(key):
    return any(
        key == prefix or key.startswith(f"{prefix}.")
        for prefix in LEGACY_SPARSE_VAE_STATE_PREFIXES
    )


def compose_sparse_vae_state_dict(model_state, checkpoint_state):
    checkpoint_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint_state.items()
    }
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(
        key
        for key in checkpoint_keys - model_keys
        if not _is_legacy_sparse_vae_state_key(key)
    )
    if missing or unexpected:
        raise RuntimeError(
            "Stage1 checkpoint is incompatible with SparseVAE: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        key: checkpoint_state[key]
        for key in model_state
        if key in checkpoint_state
    }
