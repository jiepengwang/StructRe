"""Unified experiment entry point for StructRe workflows."""

from runtime.arguments import parse_arguments
from runtime.cli import handle_inspection_arguments
from runtime.code_inference import run_code_inference
from runtime.launch import launch_training, launch_validation
from runtime.pipelines import PIPELINES, PIPELINE_RUNTIME, format_pipeline
from runtime.rewriting import run_rewriting, validate_rewriting_arguments
from runtime.runner import Runner, load_runtime
from runtime.training import get_runner_ddp, get_runner_deepspeed, start
from runtime.validation import (
    _validate_custom_mesh_task_arguments,
    _validate_dataloader_inference_direction,
    validate,
)


def main(argv=None):
    """Route the selected workflow to its shared runtime implementation."""
    args = parse_arguments(argv)
    if handle_inspection_arguments(args, pipeline_formatter=format_pipeline):
        return 0

    workflow_mode = args.mode
    if workflow_mode == 'train':
        args.mode = 'train'
        args.use_no_dataloader = False
        args.mapping_test_mesh = False
        args.sparse_vae_should_valid_mesh = False
        args.mapping_validate_hierarchy = False
        launch_training(args, start)
    elif workflow_mode == 'validate':
        args.mode = 'test'
        args.use_no_dataloader = False
        args.mapping_test_mesh = False
        if args.mapping_direction_infer is None:
            args.mapping_direction_infer = 0
        args.sparse_vae_should_valid_mesh = True
        args.mapping_validate_hierarchy = True
        launch_validation(args, validate)
    elif workflow_mode == 'infer_code':
        args.mode = 'test'
        args.use_no_dataloader = False
        args.mapping_test_mesh = False
        args.sparse_vae_should_valid_mesh = False
        args.mapping_validate_hierarchy = False
        launch_validation(args, run_code_inference)
    elif workflow_mode == 'infer_rewriting':
        args.mode = 'test'
        args.mapping_test_mesh = True
        validate_rewriting_arguments(args, args.mapping_test_name_task)
        launch_validation(args, run_rewriting)
    elif workflow_mode == 'test':
        if args.mapping_test_mesh:
            name_task = args.mapping_test_name_task or 'generation'
            args.mapping_test_name_task = name_task
            _validate_custom_mesh_task_arguments(args, name_task)
        launch_validation(args, validate)
    else:
        raise NotImplementedError(f'Unsupported mode: {workflow_mode}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
