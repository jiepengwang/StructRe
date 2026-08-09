"""Runtime workflow for the 10 canonical rewriting inference tasks."""

import math

from runtime.inference.registry import get_inference_task
from runtime.inference.task_dispatch import run_task_mode
from runtime.runner import load_runtime, prepare_inference_runner


def validate_rewriting_arguments(args, name_task):
    if not name_task:
        raise ValueError("--mapping_test_name_task is required")
    task = get_inference_task(name_task)
    if task.surface != 'task_mode':
        raise ValueError(
            f'{task.name} is a dataloader task and cannot be used as a '
            'rewriting task'
        )
    missing = [
        option
        for attribute, option in task.required_options
        if getattr(args, attribute, None) in (None, '')
    ]
    if missing:
        raise ValueError(
            f'{task.name} requires {", ".join(missing)}'
        )
    if task.name == 'mesh_cycle':
        cycle_noise = args.struct_add_partcode_noise
        if not math.isfinite(cycle_noise) or cycle_noise < 0:
            raise ValueError(
                '--struct_add_partcode_noise must be finite and non-negative'
            )
    return task


def run_rewriting(args, runner=None, phase=None):
    """Run one canonical rewriting task with an optional prepared Runner."""
    task = validate_rewriting_arguments(
        args,
        args.mapping_test_name_task,
    )
    runtime = load_runtime()
    if runner is None:
        runner, _ = prepare_inference_runner(args)
    if phase is None:
        phase = f'infer_{args.extract_mode}{args.tag_phase_infer}'

    return run_task_mode(
        task=task,
        runner=runner,
        args=args,
        phase=phase,
        path_config=runtime['PathConfig'],
        progress=runtime['tqdm'],
    )
