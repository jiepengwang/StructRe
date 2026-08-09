"""Training runner construction and process entry workflow."""

import os

from runtime.runner import Runner, load_runtime


def _load_runtime():
    globals().update(load_runtime())


def _load_deepspeed():
    try:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
    except ImportError as exc:
        raise RuntimeError(
            '--deepspeed requires the optional deepspeed package'
        ) from exc
    return deepspeed, get_accelerator


def get_runner_ddp(ngpus_per_node, args):
    _load_runtime()
    if args.distributed:
        args.multi_gpu = True
        args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        args.workers = int(
            (args.num_workers + ngpus_per_node - 1) / ngpus_per_node
        )
        print(
            '(gpu id, rank, batch_size, workers): ',
            args.gpu,
            args.rank,
            args.batch_size,
            args.workers,
        )
        torch.cuda.set_device(args.gpu)

        runner = Runner(args)
        runner.model = nn.SyncBatchNorm.convert_sync_batchnorm(runner.model)
        runner.model = runner.model.cuda(args.gpu)
        runner.model = torch.nn.parallel.DistributedDataParallel(
            runner.model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
        )
    elif args.gpu is None:
        args.multigpu = True
        runner = Runner(args)
        runner.model = runner.model.cuda()
        runner.model = torch.nn.DataParallel(runner.model)
    else:
        runner = Runner(args)
        runner.model = runner.model.cuda()
    return runner


def get_runner_deepspeed(ngpus_per_node, args):
    _load_runtime()
    return Runner(args)


def start(gpu, ngpus_per_node, args):
    _load_runtime()
    print('Start')

    if args.deepspeed:
        gpu = args.local_rank

    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    args.multigpu = False
    if args.deepspeed:
        deepspeed, get_accelerator = _load_deepspeed()
        runner = get_runner_deepspeed(ngpus_per_node, args)
        deepspeed.init_distributed(dist_backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        get_accelerator().set_device(local_rank)
    else:
        runner = get_runner_ddp(ngpus_per_node, args)

    print(f'created model on gpu {gpu}')
    runner.train(args.gpu)
    print('Done')
