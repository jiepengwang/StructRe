import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader
from termcolor import cprint

import confs.path as PathConfig


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tuple(tensor.shape)
    extended_shape = tuple(extended_shape)
    if not 1 <= tensor.ndim <= 4:
        raise ValueError(
            f"Expected a 1-4D tensor for padding, got shape {tensor_shape}"
        )
    if len(extended_shape) != tensor.ndim:
        raise ValueError(
            "Padding target rank must match the tensor rank: "
            f"shape={tensor_shape}, target={extended_shape}"
        )
    if any(target < current for current, target in zip(tensor_shape, extended_shape)):
        raise ValueError(
            "Padding target cannot be smaller than the tensor: "
            f"shape={tensor_shape}, target={extended_shape}"
        )

    extended_tensor = torch.full(
        extended_shape,
        fill,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    tensor_slices = tuple(slice(0, size) for size in tensor_shape)
    extended_tensor[tensor_slices] = tensor
    return extended_tensor


def padded_stack(tensors, padding=0):
    if not tensors:
        raise ValueError("Cannot pad an empty tensor collection")

    tensor_shapes = [tuple(tensor.shape) for tensor in tensors]
    dim_count = tensors[0].ndim
    if not 1 <= dim_count <= 4:
        raise ValueError(
            f"Expected 1-4D tensors for padding, got shapes {tensor_shapes}"
        )
    if any(tensor.ndim != dim_count for tensor in tensors):
        raise ValueError(
            f"All tensors must have the same rank, got shapes {tensor_shapes}"
        )

    max_shape = tuple(
        max(tensor.shape[dimension] for tensor in tensors)
        for dimension in range(dim_count)
    )
    padded_tensors = [
        extend_tensor(tensor, max_shape, fill=padding)
        for tensor in tensors
    ]
    return torch.cat(padded_tensors)


def collate_fn_padding(batch):
    '''https://github.com/lavis-nlp/spert/tree/master/spert
    '''
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if key in ['path', 'level_id', 'cat_id', 'cat_str', 
                    'path_child', 'code_path_child', 'code_path_parent']:
            padded_batch[key] = samples
            continue
            
        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack(samples, padding=PathConfig.Padding_Value_Seq)

    return padded_batch


class ShapeDataLoader(object):
    def __init__(self, args, conf, mode):
        dataset_name = conf.dataset.dataset_name
        if dataset_name == 'partnet':
            from model.sparse_vae.dataloader import PartnetLoadPreprocess

            dataset_class = PartnetLoadPreprocess
        elif dataset_name == 'partnetcode':
            from model.part_vae.dataloader import PartnetCodeLoadPreprocess

            dataset_class = PartnetCodeLoadPreprocess
        elif dataset_name == 'hierarchy_code':
            from model.structmapping.dataloader import HierarchyCodeLoadPreprocess

            dataset_class = HierarchyCodeLoadPreprocess
        else:
            raise ValueError(
                f"Unsupported dataset name {dataset_name!r}; expected "
                "'partnet', 'partnetcode', or 'hierarchy_code'"
            )

        self.t_samples = dataset_class(args, conf, mode)
        collate_fn = collate_fn_padding
        
        # train, train_big
        if 'train' in mode:
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples, shuffle=True)
                self.data = DataLoader(self.t_samples, args.batch_size,
                                            shuffle=(self.train_sampler is None),
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            drop_last=True,
                                            collate_fn = collate_fn, 
                                            sampler=self.train_sampler,
                                            persistent_workers=True)

            else:
                self.train_sampler = None
                
                self.data = DataLoader(self.t_samples,
                                            batch_size=args.batch_size,
                                            shuffle=conf['dataset.shuffle'],
                                            drop_last=conf['dataset.drop_last'],
                                            collate_fn = collate_fn, 
                                            num_workers=int(args.num_workers),
                                            persistent_workers=True)
        else:
            should_shuffle_test_data = conf['dataset.should_shuffle_test_data']
            if args.distributed:
                should_shuffle_test_data = True
            self.data = DataLoader(self.t_samples, 
                                    batch_size=args.batch_size,
                                   shuffle=should_shuffle_test_data,
                                   num_workers=int(args.num_workers),
                                   collate_fn = collate_fn, 
                                   pin_memory=False)
        
        self.n_batches = len(self.data)
        cprint(f'Batch size: {args.batch_size}. Batch number: {len(self.data)}', 'yellow')
