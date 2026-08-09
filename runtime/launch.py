"""Shared process launch helpers for experiment workflows."""

import os

from runtime.runner import load_runtime


def prepare_launch_arguments(args):
    """Populate process and device fields expected by legacy runtime code."""
    print(f'Found cpus: {os.cpu_count()} cpus.')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if args.visible_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus

    runtime = load_runtime()
    np = runtime['np']
    torch = runtime['torch']

    args.world_size = 1
    args.rank = 0
    if args.distributed:
        port = np.random.randint(15000, 15025)
        args.dist_url = f'tcp://127.0.0.1:{port}'
        args.dist_backend = 'nccl'
        args.gpu = None

    num_gpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.num_gpus_per_node = num_gpus_per_node
    args.batch_size_total = num_gpus_per_node * args.batch_size
    return runtime, num_gpus_per_node


def launch_validation(args, validation_fn):
    """Launch one validation or inference workflow through DistUtils."""
    runtime, num_gpus_per_node = prepare_launch_arguments(args)
    runtime['mp'].set_start_method('spawn')
    runtime['DistUtils'].launch(
        fn=validation_fn,
        n_gpu_per_machine=num_gpus_per_node,
        n_machine=1,
        machine_rank=0,
        dist_url=args.dist_url2,
        args=(args,),
    )


def launch_training(args, training_fn):
    """Launch a single-process or multiprocessing training workflow."""
    runtime, num_gpus_per_node = prepare_launch_arguments(args)
    if args.distributed:
        runtime['mp'].set_start_method('forkserver')
        args.world_size = num_gpus_per_node * args.world_size
        runtime['mp'].spawn(
            training_fn,
            nprocs=num_gpus_per_node,
            args=(num_gpus_per_node, args),
        )
    else:
        args.gpu = 0
        training_fn(args.gpu, num_gpus_per_node, args)
