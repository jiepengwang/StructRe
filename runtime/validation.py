"""Dataloader-backed validation for the three active model stages."""

from copy import deepcopy

from runtime.code_inference import CODE_TASK_BY_ARCH, run_code_inference
from runtime.rewriting import run_rewriting, validate_rewriting_arguments
from runtime.runner import load_runtime, prepare_inference_runner


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _validate_dataloader_inference_direction(model_arch, direction):
    if model_arch != 'structmapping':
        return direction
    if direction is None:
        return 0
    if direction != 0:
        raise ValueError(
            'Bottom-up StructMapping inference requires a --mapping_test_mesh '
            'task that provides initial shape information; the dataloader '
            'validation path supports direction 0.'
        )
    return direction


def _validate_custom_mesh_task_arguments(args, name_task):
    return validate_rewriting_arguments(args, name_task).name


def _select_loader(runner, extract_mode):
    if extract_mode == 'train':
        return runner.train_loader
    if extract_mode == 'test':
        return runner.test_loader
    raise ValueError(
        f'Unsupported extract_mode {extract_mode!r}; '
        "expected 'train' or 'test'"
    )


def _validate_repeat_count(model_arch, repeat_count):
    if repeat_count < 1:
        raise ValueError('--mapping_n_iter_rand must be at least 1')
    if repeat_count > 1 and model_arch not in {'part_vae', 'structmapping'}:
        raise ValueError(
            'Repeated dataloader validation is supported only for '
            'part_vae and structmapping'
        )


def _print_metrics(logs):
    for key, value in logs.items():
        if key.startswith('loss/') or 'accu_' in key:
            print(f'{key}: {value}')


def _validate_sparse_vae_batch(
    *, runner, model, data_dict, phase, repeat_index, runtime, device, args,
):
    del repeat_index, device, args
    with runtime['torch'].no_grad():
        model.validate_sdf(
            data_dict,
            phase=phase,
            exp_dir=runner.exp_dir,
        )


def _validate_part_vae_batch(
    *, runner, model, data_dict, phase, repeat_index, runtime, device, args,
):
    del phase, repeat_index, args
    if model.sparse_vae is None:
        raise RuntimeError(
            'PartVAE mesh validation requires Stage 1; pass '
            '--should_load_sparse_vae and --path_ckpt_stage1'
        )
    with runtime['torch'].no_grad():
        _, logs = runner.model(data_dict, device, runner.iter_steps)
        _print_metrics(logs)
        model.validate_geometry(data_dict)


def _validate_structmapping_batch(
    *, runner, model, data_dict, phase, repeat_index, runtime, device, args,
):
    if args.mapping_validate_hierarchy:
        part_vae = getattr(model, 'part_vae', None)
        if part_vae is None or getattr(part_vae, 'sparse_vae', None) is None:
            raise RuntimeError(
                'StructMapping hierarchy validation requires Stage 2 and '
                'Stage 1; pass --should_load_sparse_vae, --path_ckpt_stage2, and '
                '--path_ckpt_stage1'
            )

    with runtime['torch'].no_grad():
        _, logs = runner.model(data_dict, device, runner.iter_steps)
        _print_metrics(logs)

        if not args.mapping_validate_hierarchy:
            return
        model.validate_hierarchy(
            f'{runner.exp_dir}/{phase}/sample{repeat_index}',
            dict_data_inp_hier=model.select_hierarchy_sample(
                model.dict_data_inp
            ),
            should_validate_hierarchy=True,
            dict_info_shape_inp=None,
            should_render_mesh=True,
        )


_VALIDATION_HANDLERS = {
    'sparse_vae': _validate_sparse_vae_batch,
    'part_vae': _validate_part_vae_batch,
    'structmapping': _validate_structmapping_batch,
}


def validate(args):
    """Run one dataloader validation workflow or a legacy delegated task."""
    if not args.mapping_test_mesh and args.mapping_direction_infer is None:
        args.mapping_direction_infer = 0

    runtime = load_runtime()
    runner, device = prepare_inference_runner(args)
    model_arch = runner.conf['model.arch']
    phase = f'infer_{args.extract_mode}{args.tag_phase_infer}'

    if args.mapping_test_mesh:
        run_rewriting(args, runner=runner, phase=phase)
        return
    if model_arch in CODE_TASK_BY_ARCH and not args.sparse_vae_should_valid_mesh:
        run_code_inference(args, runner=runner)
        return

    try:
        handler = _VALIDATION_HANDLERS[model_arch]
    except KeyError as error:
        raise ValueError(
            f'Unsupported validation model.arch: {model_arch!r}'
        ) from error

    model = _unwrap_model(runner.model)
    direction = _validate_dataloader_inference_direction(
        model_arch,
        args.mapping_direction_infer,
    )
    if model_arch == 'structmapping':
        args.mapping_direction_infer = direction
        model.mapping_direction_infer = direction
        if model.args is not args:
            model.args.mapping_direction_infer = direction

    repeat_count = args.mapping_n_iter_rand
    _validate_repeat_count(model_arch, repeat_count)
    loader = _select_loader(runner, args.extract_mode)

    runner.model.eval()
    runtime['cprint'](
        f'[{model_arch}] Validate {args.extract_mode} data',
        'yellow',
    )
    print(f'Number of batches: {len(loader)}')
    batches = loader if args.distributed else runtime['tqdm'](loader)

    for data_dict_inp in batches:
        for repeat_index in range(repeat_count):
            if repeat_count > 1:
                print(
                    f'Validation sample {repeat_index + 1}/{repeat_count}'
                )
            handler(
                runner=runner,
                model=model,
                data_dict=deepcopy(data_dict_inp),
                phase=phase,
                repeat_index=repeat_index,
                runtime=runtime,
                device=device,
                args=args,
            )
            runtime['torch'].cuda.empty_cache()

    print('Done')
