"""Runtime dispatch for canonical task-mode inference tasks."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

# Historical generation sweeps explored perturbation magnitudes through 2.4.
_MAX_GENERATION_NOISE_LEVEL = 2.4


@dataclass(frozen=True)
class TaskItem:
    """One top-level invocation produced by a task preset."""

    output_subdir: str
    argument_updates: tuple = ()


@dataclass(frozen=True)
class TaskHandler:
    """Declarative item expansion for one canonical task."""

    build_items: object
    show_progress: bool = False


def _single_item(_args, _path_config):
    return (TaskItem('sample0'),)


def _completion_items(_args, _path_config):
    return (TaskItem('sample0', (('struct_add_partcode_noise', 0),)),)


def _generation_noise_items(args, _path_config):
    noise_levels = getattr(args, 'mapping_noise_levels', (1.6,))
    if not noise_levels:
        raise ValueError('--mapping_noise_levels requires at least one value')
    if args.mapping_n_iter_rand < 1:
        raise ValueError('--mapping_n_iter_rand must be at least 1')

    items = []
    seen_levels = set()
    for raw_level in noise_levels:
        noise_level = float(raw_level)
        if not math.isfinite(noise_level):
            raise ValueError(
                '--mapping_noise_levels values must be finite'
            )
        if not 0.0 <= noise_level <= _MAX_GENERATION_NOISE_LEVEL:
            raise ValueError(
                '--mapping_noise_levels values must be between 0.0 and '
                f'{_MAX_GENERATION_NOISE_LEVEL}'
            )
        rounded_level = round(noise_level, 2)
        if rounded_level == 0:
            rounded_level = 0.0
        if not math.isclose(
            noise_level,
            rounded_level,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                '--mapping_noise_levels values support at most two decimals'
            )
        if rounded_level in seen_levels:
            raise ValueError(
                f'Duplicate --mapping_noise_levels value: {rounded_level}'
            )
        seen_levels.add(rounded_level)

        n_iter_rand = (
            1 if rounded_level == 0.0 else args.mapping_n_iter_rand
        )
        noise_tag = f'noise{rounded_level:.02f}'.replace('.', '_')
        items.append(TaskItem(
            f'{noise_tag}/sample0',
            (
                ('mapping_n_iter_rand', n_iter_rand),
                ('struct_add_partcode_noise', rounded_level),
            ),
        ))
    return tuple(items)


TASK_MODE_HANDLERS = {
    'generation_noise_sweep': TaskHandler(
        _generation_noise_items,
    ),
    'completion': TaskHandler(
        _completion_items,
    ),
    'reconstruction': TaskHandler(
        _single_item,
    ),
    'reconstruction_shapenet_batch': TaskHandler(
        _single_item,
    ),
    'reconstruction_partnet_batch': TaskHandler(
        _single_item,
    ),
    'mesh_cycle': TaskHandler(
        _single_item,
    ),
    'mesh_reconstruct': TaskHandler(
        _single_item,
    ),
    'mesh_encode': TaskHandler(
        _single_item,
    ),
    'shape_code_generate': TaskHandler(
        _single_item,
    ),
    'shape_code_interpolate': TaskHandler(
        _single_item,
    ),
}


_MISSING = object()


def _direction_updates(task):
    directions = {"0": 0, "1": 1, "0 -> 1": 0}
    try:
        direction = directions[task.direction]
    except KeyError as error:
        raise ValueError(
            f"Unsupported direction for task {task.name}: {task.direction}"
        ) from error
    return (("mapping_direction_infer", direction),)


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _model_arguments(model):
    return getattr(_unwrap_model(model), 'args', None)


@contextmanager
def _temporary_argument_updates(args, model, updates):
    model_instance = _unwrap_model(model)
    targets = [args]
    model_args = _model_arguments(model)
    if model_args is not None and model_args is not args:
        targets.append(model_args)

    previous_values = []
    for name, value in updates:
        update_targets = list(targets)
        if name == "mapping_direction_infer":
            update_targets.append(model_instance)
        for target in update_targets:
            previous_values.append((
                target,
                name,
                getattr(target, name, _MISSING),
            ))
            setattr(target, name, value)

    try:
        yield
    finally:
        for target, name, previous_value in reversed(previous_values):
            if previous_value is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, previous_value)


def run_task_mode(task, runner, args, phase, path_config, progress):
    """Run one canonical task-mode task through its declarative handler."""
    if task.surface != 'task_mode':
        raise ValueError(f'{task.name} is not a task-mode inference task')

    try:
        handler = TASK_MODE_HANDLERS[task.name]
    except KeyError as error:
        raise ValueError(
            f'No runtime handler registered for task {task.name!r}'
        ) from error

    items = handler.build_items(args, path_config)
    item_iter = progress(items) if handler.show_progress else items
    model = _unwrap_model(runner.model)
    results = []
    task_updates = _direction_updates(task)
    for item in item_iter:
        with _temporary_argument_updates(
            args,
            runner.model,
            task_updates + item.argument_updates,
        ):
            results.append(model.run_inference_task(
                task.name,
                dir_exp=f'{runner.exp_dir}/{phase}/{item.output_subdir}',
                args=args,
            ))
    return results
