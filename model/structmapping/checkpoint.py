"""Strict Stage 3 checkpoint composition for optional nested dependencies."""

NESTED_PART_VAE_STATE_PREFIX = "part_vae."
LEGACY_STRUCTMAPPING_STATE_KEYS = frozenset({
    "decoder.start_emb.weight",
    "decoder.linear_pre_tf.weight",
    "decoder.linear_pre_tf.bias",
})


def compose_structmapping_state_dict(model_state, checkpoint_state):
    checkpoint_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint_state.items()
    }
    stage3_keys = {
        key
        for key in model_state
        if not key.startswith(NESTED_PART_VAE_STATE_PREFIX)
    }
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(stage3_keys - checkpoint_keys)
    unexpected = sorted(
        key
        for key in checkpoint_keys - stage3_keys
        if key not in LEGACY_STRUCTMAPPING_STATE_KEYS
    )
    if missing or unexpected:
        raise RuntimeError(
            "Stage3 checkpoint is incompatible with StructMapping: "
            f"missing={missing}, unexpected={unexpected}"
        )

    composed = dict(model_state)
    composed.update(
        {key: checkpoint_state[key] for key in stage3_keys}
    )
    return composed
