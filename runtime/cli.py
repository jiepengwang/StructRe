"""Runtime-free inspection routing shared by experiment entry points."""

from runtime.inference.registry import (
    format_inference_task,
    format_inference_task_inventory,
)
from runtime.pipelines import format_pipeline


def handle_inspection_arguments(args, pipeline_formatter=None):
    """Print one requested capability description and report if handled."""
    if args.describe_pipeline:
        if pipeline_formatter is None:
            pipeline_formatter = format_pipeline
        print(pipeline_formatter(args.describe_pipeline))
        return True
    if args.list_inference_tasks:
        print(format_inference_task_inventory())
        return True
    if args.describe_inference_task:
        print(format_inference_task(args.describe_inference_task))
        return True
    return False
