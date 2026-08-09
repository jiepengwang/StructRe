"""Dataloader workflows for Stage 1 and Stage 2 code extraction."""

from runtime.runner import load_runtime, prepare_inference_runner


CODE_TASK_BY_ARCH = {
    'sparse_vae': 'extract_patch_codes',
    'part_vae': 'extract_part_codes',
}


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _select_loader(runner, extract_mode):
    if extract_mode == 'train':
        return runner.train_loader
    if extract_mode == 'test':
        return runner.test_loader
    raise ValueError(
        f'Code extraction requires --extract_mode train or test, got '
        f'{extract_mode!r}'
    )


def run_code_inference(args, runner=None):
    """Extract Stage 1 patch codes or Stage 2 part codes in one pass."""
    runtime = load_runtime()
    if runner is None:
        runner, device = prepare_inference_runner(args)
    else:
        device = args.gpu

    model_arch = runner.conf['model.arch']
    try:
        task_name = CODE_TASK_BY_ARCH[model_arch]
    except KeyError as error:
        raise ValueError(
            'Code extraction supports model.arch sparse_vae or part_vae, '
            f'got {model_arch!r}'
        ) from error

    loader = _select_loader(runner, args.extract_mode)
    model = _unwrap_model(runner.model)
    if model_arch == 'sparse_vae':
        runner.conf['train']['should_extract_code'] = True
    else:
        folder_code = runtime['Folder_Code_Struct']
        runtime['cprint'](
            f'Save extracted code to: {folder_code}',
            'yellow',
        )

    runtime['cprint'](
        f'[{model_arch}] {task_name} from {args.extract_mode} data',
        'yellow',
    )
    print(f'Number of batches: {len(loader)}')
    for data_dict in runtime['tqdm'](loader):
        with runtime['torch'].no_grad():
            _, logs_model = runner.model(
                data_dict,
                device,
                runner.iter_steps,
            )

        if model_arch == 'sparse_vae':
            model.save_code(data_dict, logs_model)
        else:
            model.extract_code(data_dict, folder_code=folder_code)
        runtime['torch'].cuda.empty_cache()

    print('Done')
