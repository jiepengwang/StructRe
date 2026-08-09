"""Model lifecycle and training runner with lazy ML runtime imports."""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from shutil import copyfile


_RUNTIME_EXPORT_NAMES = (
    "np", "colored", "cprint",
    "ensure_dir_existence", "get_gpu_usage",
    "DistUtils", "IOUtils", "ImageUtils",
    "ConfigFactory", "tqdm", "create_model",
    "ShapeDataLoader", "torch", "nn", "optim", "dist", "mp",
    "SummaryWriter", "DIR_EXPS",
    "is_server",
    "Folder_Code_Struct",
    "PathConfig",
)
_RUNTIME_LOADED = False


def _runtime_exports():
    return {name: globals()[name] for name in _RUNTIME_EXPORT_NAMES}


def load_runtime():
    global _RUNTIME_LOADED
    global np, colored, cprint
    global ensure_dir_existence, get_gpu_usage
    global DistUtils, IOUtils, ImageUtils
    global ConfigFactory, tqdm, create_model, ShapeDataLoader
    global torch, nn, optim, dist, mp, SummaryWriter
    global DIR_EXPS, is_server
    global Folder_Code_Struct
    global PathConfig

    if _RUNTIME_LOADED:
        return _runtime_exports()

    os.environ['LANG'] = 'en_US'
    os.environ['NUMEXPR_MAX_THREADS'] = '8'

    import numpy as np
    from termcolor import colored, cprint
    from utils.utils_io import ensure_dir_existence, get_gpu_usage
    import utils.distributed as DistUtils
    import utils.utils_io as IOUtils
    import utils.utils_image as ImageUtils
    from pyhocon import ConfigFactory
    from tqdm import tqdm
    from model.model_config import create_model
    from model.dataloader_config import ShapeDataLoader
    import torch
    from torch import nn, optim
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.utils.tensorboard import SummaryWriter
    from confs.path import (
        DIR_EXPS,
        Folder_Code_Struct,
        is_server,
    )
    import confs.path as PathConfig

    _RUNTIME_LOADED = True
    return _runtime_exports()


def _merge_checkpoint_model_state(
    model_state, checkpoint_state, model_arch=None,
):
    normalized_checkpoint = {
        key.removeprefix("module."): value
        for key, value in checkpoint_state.items()
    }
    model_keys = set(model_state)
    checkpoint_keys = set(normalized_checkpoint)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)

    if model_arch == "part_vae":
        from model.part_vae.part_vae import compose_part_vae_state_dict

        merged_state = compose_part_vae_state_dict(
            model_state, normalized_checkpoint
        )
    elif model_arch == "sparse_vae":
        from model.sparse_vae.checkpoint import (
            compose_sparse_vae_state_dict,
        )

        merged_state = compose_sparse_vae_state_dict(
            model_state, normalized_checkpoint
        )
    elif model_arch == "structmapping":
        from model.structmapping.checkpoint import (
            compose_structmapping_state_dict,
        )

        merged_state = compose_structmapping_state_dict(
            model_state, normalized_checkpoint
        )
    elif missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint model keys do not match the current model: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    else:
        merged_state = normalized_checkpoint
    return merged_state, missing_keys, unexpected_keys


class Runner:
    def __init__(self, args):
        load_runtime()
        self._configure_runtime(args)

        if not args.use_no_dataloader:
            self.initialize()

        # model
        self.model = create_model(self.args, self.conf)
        self.arch_model = self.conf['model.arch']

        self.model.exp_dir = self.exp_dir

        self.iter_steps = 0
        self.epoch = 0
        self.is_server = is_server

        # optmizer
        self.set_optimizer(use_OneCycleLR=False, steps_per_epoch=-1)

        self.is_contiue = args.is_continue
        if self.is_contiue:
            self.load_pretrained_model()


    def _configure_runtime(self, args):
        self.args = args
        self.conf = ConfigFactory.parse_file(args.conf)

        if args.mode in ['test']:
            self.conf['dataset']['drop_last'] = False
            self.conf['dataset']['shuffle'] = False
            self.conf['dataset']['use_np_shuffle'] = False
            self.conf['dataset']['should_shuffle_test_data'] = False

        if args.should_load_sparse_vae:
            self.conf['model']['should_load_sparse_vae'] = True

        self.freq_report = self.conf['train.freq_report']

        self.model_arch = self.conf['model.arch']

        if args.exp_name:
            self.conf['general']['exp_name'] = args.exp_name
        self.exp_dir = f"{DIR_EXPS}/{self.conf['model.arch']}/{self.conf['general.exp_name']}"

        if args.category:
            self.conf['dataset']['category'] = args.category

        if args.lr is not None:
            self.conf['train']['lr'] = args.lr
            cprint(f'Learning rate: {args.lr}.', 'yellow')

        # dataloader
        if args.max_dataset_size:
            self.conf['dataset']['max_dataset_size'] = args.max_dataset_size

        if args.max_category_size:
            self.conf['dataset']['max_category_size'] = args.max_category_size

        if self.model_arch == 'sparse_vae' and args.sparse_vae_mode_shape:
            self.conf['dataset']['mode_shape'] = args.sparse_vae_mode_shape

        if self.model_arch in ['structmapping', 'completion']:
            if args.mapping_n_iter_decoding:
                self.conf['model']['n_iter'] = args.mapping_n_iter_decoding
                print(f'[Inner loop] Iterative decoding steps: {args.mapping_n_iter_decoding}')

            self.arch_decoder = 'detr'

        if self.args.mode == 'train':
            if ((not self.args.distributed) or self.args.rank == 0):
                self.file_backup()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.args.device = device
        print(f'Runner device: {self.args.device}')

        # save args
        dir_args_record = os.path.join(self.exp_dir, 'recording_args')
        path_args =  f'{dir_args_record}/{IOUtils.get_time_str("Y_M_D-H_M_S")}_{args.tag_phase_infer[1:]}_{args.extract_mode}.conf'
        IOUtils.ensure_dir_existence(dir_args_record)
        IOUtils.ensure_pdir_existence(path_args)
        with open(path_args, 'w') as f:
            args_dump = deepcopy(args)
            if 'device' in args_dump.__dict__:
                args_dump.device = None
            args_dict = vars(args_dump)
            json.dump(args_dict, f, indent=2)

    def load_pretrained_model(self):
        # Load checkpoint
        requested_checkpoint = PathConfig.resolve_checkpoint_path(
            self.args.path_ckpt,
            os.path.join(self.exp_dir, 'checkpoints'),
        )
        if requested_checkpoint is not None:
            if not os.path.isfile(requested_checkpoint):
                raise FileNotFoundError(
                    f'Checkpoint does not exist: {requested_checkpoint}'
                )
            cprint(
                f'Load specified ckpt: {requested_checkpoint}',
                'yellow',
            )
            self.load_checkpoint(requested_checkpoint)
            return

        if self.args.use_pretrained_mapping_ckpt:
            cprint('Use pretrained mapping model', 'yellow')
            self.load_checkpoint(
                PathConfig.resolve_stage_checkpoint(self.args, stage=3),
                only_load_weights=True,
            )
            return

        latest_model_name = None
        if not os.path.exists(os.path.join(self.exp_dir, 'checkpoints')):
            print(f'No checkpoint found in {self.exp_dir}')
            return

        model_list_raw = os.listdir(os.path.join(self.exp_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            model_list.append(model_name)
        model_list.sort()
        if not (len(model_list)>0):
            return

        latest_model_name = model_list[-1]
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            fpath = os.path.join(self.exp_dir, 'checkpoints', latest_model_name)
            self.load_checkpoint(fpath)


    def file_backup(self):
        if self.args.distributed and (not self.args.rank == 0):
            return
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] in ['.py', 'aml', 'onf']:
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.args.conf, os.path.join(self.exp_dir, 'recording', 'config.conf'))

    def check_input_sequence(self, data_dict):
        if self.model_arch != 'structmapping' or self.arch_decoder != 'detr':
            return data_dict

        mask_parent_valid = (
            data_dict['code_idx_parent'].squeeze(dim=-1) >= 0
        )
        n_valid_all = mask_parent_valid.sum()
        if n_valid_all > self.max_seq_parent:
            self.max_seq_parent = n_valid_all
        arr_n_valid = mask_parent_valid.sum(dim=-1)

        if n_valid_all > self.max_parent_batch:
            idx_max = np.argmax(arr_n_valid)
            for key in data_dict:
                data_dict[key] = data_dict[key][idx_max:(idx_max + 1)]
            print(
                f'[{self.iter_steps}] Total valid parent: '
                f'{n_valid_all}/{self.max_seq_parent}. '
                'Change batch size to: 1'
            )

        mask_child_valid = (
            data_dict['patch_code_idx_child'].squeeze(dim=-1) >= 0
        )
        n_valid_child_all = mask_child_valid.sum()
        if n_valid_child_all > self.max_seq_child:
            self.max_seq_child = n_valid_child_all
        arr_n_child_valid = mask_child_valid.sum(dim=-1)

        child_exceeds_limit = (
            n_valid_child_all > self.max_child_batch
            or (arr_n_child_valid > 1500).any()
        )
        if child_exceeds_limit:
            idx_max = np.argmax(arr_n_child_valid)
            for key in data_dict:
                data_dict[key] = data_dict[key][idx_max:(idx_max + 1)]
            print(
                f'[{self.iter_steps}] Total valid child: '
                f'{n_valid_child_all}/{self.max_seq_child}. '
                'Change batch size to: 1. '
                f'[{arr_n_child_valid}] '
                f'[{data_dict["path"][0].split("/")[-5:]}]'
            )

        return data_dict

    def train(self, device):
        # optmizer

        # start training
        should_write = ((not self.args.distributed) or self.args.rank == 0)
        if should_write:

            dir_log = f"{self.exp_dir}/logs"
            os.makedirs(dir_log, exist_ok=True)
            self.writer = SummaryWriter(log_dir=dir_log)

            if self.conf['model']['arch']=='structmapping':
                dir_log2 = f"{self.exp_dir}/logs_iter"
                os.makedirs(dir_log2, exist_ok=True)
                self.writer_iter = SummaryWriter(log_dir=dir_log2)

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.train()
        epoch_start = self.epoch
        self.model_arch = self.conf['model.arch']
        self.batch_size = self.args.batch_size

        self.max_seq_parent = 0; self.max_seq_child = 0
        self.max_parent_batch = 400; self.max_child_batch = 1000 * self.args.batch_size

        iterator_epochs = range(epoch_start, self.conf['train.num_epochs']) if (self.is_server and (not should_write)) else tqdm(range(epoch_start, self.conf['train.num_epochs']))


        for epoch in iterator_epochs:
            if should_write and (not self.is_server):
                t_loader = tqdm(self.train_loader, desc=f"Epoch: {epoch + 1}/{self.conf['train.num_epochs']}.")
            else:
                t_loader = self.train_loader

            if self.args.distributed and self.arch_model!='sparse_vae':
                self.train_dataset.train_sampler.set_epoch(epoch)

            num_batches = len(t_loader)
            t_start = datetime.now()
            for data_dict in t_loader:
                self.model.train()

                # check batch size
                data_dict = self.check_input_sequence(data_dict)

                if should_write and (self.iter_steps % self.args.num_batches == 0) and self.args.distributed:
                    print(f'\n\n*******[{epoch}] Process data: ')
                    count_p = 0
                    for j in range(len(data_dict["path"])):
                        path = data_dict["path"][j].split('/')
                        print(f'{path[-4:]}')
                        count_p += 1
                        if count_p>1:
                            break

                loss, logs_model = self.model(data_dict, device, self.iter_steps) # B*C*DHW


                if self.args.deepspeed:
                    self.model.backward(loss)
                    self.model.step()
                else:
                    # back-propagate
                    self.optimizer.zero_grad()

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # update logs and save checkpoints
                if self.iter_steps % 100 ==0:
                    torch.cuda.empty_cache()

                self.iter_steps +=1
                if should_write:
                    logs = logs_model

                    if (self.iter_steps % self.conf['train.freq_valid_mesh']==0 or self.iter_steps==1) and self.conf['train.freq_valid_mesh']>0:
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            self.validate(data_dict, logs)
                        torch.cuda.empty_cache()

                    if self.iter_steps % self.freq_report == 0:
                        loss_ = float(loss.data.cpu().numpy())
                        if not self.is_server:

                            t_loader.set_description(f"Epoch: {epoch + 1}/{self.conf.train.num_epochs}. Loss: {loss_:.4f}.")
                            t_loader.refresh()
                        else:
                            # merge logs
                            msg_all = ''
                            for key in logs:
                                msg_all += f'{key}: {logs[key]}.\n'
                            print(f"\nEpoch: {epoch+1}/{self.conf.train.num_epochs}. Loop: {self.iter_steps%num_batches}/{num_batches}. Loss: {loss_:.4f}")
                            print(msg_all)

                        try:
                            logs.update({
                                'logs/epoch': epoch+1,
                                'logs/lr': self.optimizer.param_groups[0]['lr'],
                                'gpu/gpu_mem': get_gpu_usage().memoryUsed,
                                'gpu/gpu_load': get_gpu_usage().load
                            })
                        except (IndexError, OSError, RuntimeError) as error:
                            print(f'Failed to get GPU usage: {error}')

                        self.write_summary(logs, self.iter_steps)

                    if (self.iter_steps % self.conf['train.freq_save_step'] == 0):
                        path_checkpoints = f'{self.exp_dir}/checkpoints/ckpt_{(epoch+1):04d}_{self.iter_steps:08d}.pt'
                        self.save_checkpoint(path_checkpoints)


            self.scheduler.step()

            if should_write and ((epoch+1) % self.conf['train.freq_save'] == 0):
                path_checkpoints = f'{self.exp_dir}/checkpoints/ckpt_{(epoch+1):04d}_{self.iter_steps:08d}.pt'
                self.save_checkpoint(path_checkpoints)

                t_epoch = (datetime.now()-t_start).total_seconds()/60.0
                self.write_summary({'logs/time_epoch': t_epoch}, epoch)

            self.epoch = epoch + 1


        # empty cache
        torch.cuda.empty_cache()

    def write_summary(self, logs_summary, iter_steps):
        if 'logs_iter' in logs_summary:
            logs_iter = logs_summary['logs_iter']
            for key in logs_iter:
                self.writer_iter.add_scalar(key, logs_iter[key], iter_steps)

            del logs_summary['logs_iter']

        for key in logs_summary:
            self.writer.add_scalar(key, logs_summary[key], iter_steps)


    def validate(self, data_dict, logs):
        if self.conf['model.arch'] == 'sparse_vae' and self.epoch>2:
            self.validate_sparse_vae()
        elif self.conf['model.arch'] == 'part_vae':
            self.validate_part_vae(data_dict, logs)
        elif self.conf['model.arch'] == 'structmapping':
            self.validate_structmapping()
        else:
            return

    def validate_sparse_vae(self):
        self.model.eval()
        with torch.no_grad():
            model_infer = self.model.module if hasattr(self.model, 'module') else self.model
            try:
                data_dict_infer = next(self.iter_test_loader) # infer test data
            except StopIteration:
                self.iter_test_loader = iter(self.test_loader)
                data_dict_infer = next(self.iter_test_loader)

            _, logs_model = model_infer(data_dict_infer, 'cuda', self.iter_steps) # B*C*DHW

            path_dist = f'{self.exp_dir}/images/code_dist/test/code_{self.epoch:04d}_{self.iter_steps:08d}.png'
            IOUtils.ensure_pdir_existence(path_dist)
            code_valid_p = model_infer.feat_quant.cpu().numpy()
            ImageUtils.draw_histo_arr(path_dist, code_valid_p, n_his=80, thres_clip=None)


            self.write_summary(logs_model, self.iter_steps)

    def validate_part_vae(self, data_dict, logs):
        self.model.eval()
        model_infer = self.model.module if hasattr(self.model, 'module') else self.model
        try:
            for mode in ['train', 'test']:
                if mode == 'train':
                    data_dict_infer = data_dict  # infer training data
                elif mode == 'test':
                    try:
                        data_dict_infer = next(self.iter_test_loader) # infer test data
                    except StopIteration:
                        self.iter_test_loader = iter(self.test_loader)
                        data_dict_infer = next(self.iter_test_loader)

                    with torch.no_grad():
                        _, logs_model_infer = model_infer(data_dict_infer, 'cuda', self.iter_steps) # B*C*DHW
                        logs.update(logs_model_infer)

                torch.cuda.empty_cache()

                path_dist = f'{self.exp_dir}/images/code_dist/{mode}/code_{self.epoch:04d}_{self.iter_steps:08d}.png'
                IOUtils.ensure_pdir_existence(path_dist)
                code_valid_p = model_infer.zq.detach().cpu().numpy()
                ImageUtils.draw_histo_arr(path_dist, code_valid_p, n_his=80, thres_clip=None)

        except Exception as error:
            print(f'Validation failed; continuing training: {error}')
            torch.cuda.empty_cache()

        return None

    def validate_structmapping(self):
        self.model.eval()
        with torch.no_grad():
            model_infer = self.model.module if hasattr(self.model, 'module') else self.model

            try:
                data_dict_infer = next(self.iter_test_loader) # infer test data
            except StopIteration:
                try:
                    self.iter_test_loader = iter(self.test_loader)
                    data_dict_infer = next(self.iter_test_loader)
                except StopIteration:
                    return

            _, logs2 = model_infer(data_dict_infer, None, None)
            self.write_summary(logs2, self.iter_steps)

    def save_checkpoint(self, path_checkpoints):
        IOUtils.ensure_pdir_existence(path_checkpoints)

        self.model.eval()
        torch.save({"model": self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': self.epoch,
                    "iter_step": self.iter_steps,
                    'lr': self.optimizer.param_groups[0]['lr']}, path_checkpoints)
        print('model saved / path: {}'.format(path_checkpoints))
        self.model.train()

    # load model
    def load_checkpoint(self, fpath, only_load_weights=False):
        ckpt = torch.load(fpath, map_location="cpu")
        load_dict_model, missing_keys, unexpected_keys = (
            _merge_checkpoint_model_state(
                self.model.state_dict(), ckpt["model"],
                self.conf["model.arch"],
            )
        )
        self.model.load_state_dict(load_dict_model, strict=True)
        print(colored(
            f'[*] {self.conf["model.arch"]}: weight successfully loaded '
            f"from: {fpath}",
            "blue",
        ))

        structure_changed = bool(missing_keys or unexpected_keys)
        if only_load_weights or structure_changed:
            if structure_changed:
                cprint(
                    "Checkpoint model structure differs from the current model; "
                    f"loaded weights only (missing={len(missing_keys)}, "
                    f"unexpected={len(unexpected_keys)}).",
                    "yellow",
                )
            else:
                cprint("Only load model weights.", "yellow")
            return

        if self.args.mode == "train":
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if "lr" in ckpt:
                lr = ckpt["lr"]
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"] + 1
            self.iter_steps = self.epoch * self.args.num_batches
            print(
                f"Continue from Epoch: {self.epoch}, "
                f"Current step: {self.iter_steps}"
            )
            print("\n")

    def initialize(self):
        '''Initialize dataloader, optimizer, et al.
        '''
        print('Initialize dataloader...')
        self.train_dataset = ShapeDataLoader(self.args, self.conf, mode='train')
        self.train_loader = self.train_dataset.data

        self.test_loader = ShapeDataLoader(self.args, self.conf, mode='test').data
        self.args.num_batches = len(self.train_loader)

        def get_data_generator(loader):
            while True:
                for data in loader:
                    yield data

        self.iter_test_loader = get_data_generator(self.test_loader)


        # cudnn setting
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        print(f'Finish dataloader initialization.\n')

    def set_optimizer(self, use_OneCycleLR = False, steps_per_epoch = -1):
        if self.args.mode == 'train':
            # initialize optimizers
            paras_train = [p for p in self.model.parameters() if p.requires_grad == True]
            print('Model parameters: {}.\n'.format(sum([p.data.nelement() for p in paras_train])))

            self.optimizer = optim.Adam(paras_train, lr=self.conf.train.lr)
            if use_OneCycleLR:
                # learning rate scheduler
                self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                        max_lr=self.conf.train.lr,
                                                        epochs=self.conf.train.num_epochs,
                                                        steps_per_epoch= steps_per_epoch,
                                                        div_factor=self.conf.train.div_factor,
                                                        final_div_factor=self.conf.train.final_div_factor)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.conf.train.lr_decay, 0.9)


            self.scaler = torch.cuda.amp.GradScaler()


            if self.args.deepspeed:
                try:
                    import deepspeed
                except ImportError as exc:
                    raise RuntimeError(
                        '--deepspeed requires the optional deepspeed package'
                    ) from exc
                ds_config = {
                    "train_micro_batch_size_per_gpu": self.args.batch_size,
                    "optimizer": {
                        "type": "Adam",
                        "params": {
                            "lr": 1e-4
                        }
                    },
                }


                self.model, self.optimizer, _, self.scheduler  = deepspeed.initialize(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.scheduler,
                    args=self.args,
                    dist_init_required=False,
                    config=ds_config)


def prepare_inference_runner(args):
    """Build and place an inference Runner on its configured CUDA device."""
    load_runtime()
    runner = Runner(args)
    args.gpu = 0
    runner.model = runner.model.cuda()
    runner.model.eval()

    if args.distributed:
        nn.SyncBatchNorm.convert_sync_batchnorm(runner.model)
        runner.model = nn.parallel.DistributedDataParallel(
            runner.model,
            device_ids=[DistUtils.get_local_rank()],
            output_device=DistUtils.get_local_rank(),
        )

    return runner, args.gpu
