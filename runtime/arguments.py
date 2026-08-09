"""Runtime-free command-line arguments shared by StructRe entry points."""

import argparse
import os
import sys

from runtime.inference.registry import INFERENCE_TASK_ACCEPTED_NAMES


PIPELINE_NAMES = ('stage1', 'stage2', 'stage3')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument(
        '--describe_pipeline',
        choices=PIPELINE_NAMES,
        help='Print a model pipeline without loading runtime dependencies',
    )
    inspection.add_argument(
        '--list_inference_tasks',
        action='store_true',
        help='List all inference tasks as stable JSON without loading runtime dependencies',
    )
    inspection.add_argument(
        '--describe_inference_task',
        choices=INFERENCE_TASK_ACCEPTED_NAMES,
        help='Describe an inference task without loading runtime dependencies',
    )
    parser.add_argument('--conf', type=str, default='./confs/part_vae.conf', help='Config path')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--category', default=None, type=str, help='Category of training data')
    parser.add_argument('--exp_name', default=None, type=str, help='Experiment name')
    parser.add_argument('--tag_phase_infer', default='', type=str, help='Tag for inference')

    parser.add_argument('--lr', default=None, type=float, help='Tag for inference')

    parser.add_argument('--is_continue', action='store_true', help='Whether loading trained weights (checkpoints)')
    parser.add_argument('--use_no_dataloader', action='store_true', help='Whether loading trained weights (checkpoints)')
    parser.add_argument('--path_ckpt', type=str, default=None, help='Checkpoint path which will be loaded')
    parser.add_argument(
        '--path_ckpt_stage1',
        type=str,
        default=None,
        help='Stage 1 Sparse VAE checkpoint used by downstream models',
    )
    parser.add_argument(
        '--path_ckpt_stage2',
        type=str,
        default=None,
        help='Stage 2 Part VAE checkpoint used by downstream models',
    )
    parser.add_argument(
        '--path_ckpt_stage3',
        type=str,
        default=None,
        help='Stage 3 rewriting checkpoint used for pretrained initialization',
    )

    parser.add_argument('--distributed', action="store_true", help="Use DDP if set")
    parser.add_argument('--visible_gpus', default=None, type=str, help = 'used gpu. Ex: 0; 0,1,2')
    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--num_batches", default=16, type=int, help="Number of thread for data saving")

    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training of deepspeed')
    parser.add_argument('--deepspeed',  action='store_true', default=False, help='whether use deepspeed')

    # dataset
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per gpu")
    parser.add_argument('--model_id_start', default=0, type=int)
    parser.add_argument('--model_id_end', default=-1, type=int)
    parser.add_argument("--max_dataset_size", default=None, type=int, help="Batch size per gpu")
    parser.add_argument("--max_category_size", default=None, type=int, help="Batch size per gpu")

    # 1. Stage 1
    parser.add_argument('--sparse_vae_dataset_name', default=None, type=str, help='Dataset name to preprocess')
    parser.add_argument('--sparse_vae_mode_shape', default=None, type=str, help='Stage 1 data mode: shape_aug_rotation, shape_parts, shape_partial')


    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url2", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--extract_mode", default='train', type=str, help='Dataloader split used for code extraction' )
    parser.add_argument("--sparse_vae_should_valid_mesh", default=False, action='store_true',  help="Validate Stage 1 mesh output instead of extracting codes")
    parser.add_argument("--should_load_sparse_vae", default=False, action='store_true',  help="Load the Stage 1 geometry decoder for Stage 2/3 inference")

    # 2. struct
    parser.add_argument("--struct_add_partcode_noise", default=None, type=float, help="Whether to add noise on part code")


    # 3. mapping
    parser.add_argument("--mapping_validate_hierarchy", action='store_true', default=False, help="Whether augmenting bbox data")

    parser.add_argument("--mapping_inp_mode_normal", default='gt_normal', type=str, help="Whether using normal as input")

    parser.add_argument("--mapping_use_patch_condition_last_step", action='store_true', default=False, help="Mode to attach output code")


    parser.add_argument("--mapping_direction_infer", default=None, type=int, help="Mapping: filter strategy in iterative decoding")

    parser.add_argument("--mapping_ratio_mask_memory", default=0.5, type=float, help="ratio to mask memory")
    parser.add_argument(
        "--mapping_decode_chunk_size",
        default=8,
        type=int,
        help="Maximum number of part codes decoded per Stage 2 geometry batch",
    )
    parser.add_argument(
        "--mode_thres_prob_valid_part",
        default="fixed",
        type=str,
        help=(
            "Use the configured part-validity threshold (fixed) "
            "or an explicit numeric value"
        ),
    )

    parser.add_argument("--mapping_test_mesh", default=None, action='store_true', help="Mapping: whether to test a mesh")
    parser.add_argument(
        "--mapping_input_mesh_dir",
        default=None,
        type=str,
        help="Directory of PLY meshes for custom-mesh inference tasks",
    )
    parser.add_argument(
        "--mapping_input_manifest",
        default=None,
        type=str,
        help="JSONL manifest for file-backed rewriting tasks",
    )
    parser.add_argument(
        "--mapping_shape_code_path",
        default=None,
        type=str,
        help="Shared shape-code HDF5 output/input for custom-mesh tasks",
    )
    parser.add_argument(
        "--mapping_shape_code_source",
        default=None,
        type=str,
        help="Source HDF5 key for shape-code interpolation",
    )
    parser.add_argument(
        "--mapping_shape_code_target",
        default=None,
        type=str,
        help="Target HDF5 key for shape-code interpolation",
    )
    parser.add_argument(
        "--mapping_shape_code_steps",
        default=6,
        type=int,
        help="Number of source-to-target interpolation steps",
    )
    parser.add_argument("--mapping_update_struct_code", default=None, action='store_true', help="Mapping: whether to test a mesh")

    parser.add_argument("--mapping_n_iter_rand", default=1, type=int, help="Mapping: iteration times")
    parser.add_argument(
        "--mapping_noise_levels",
        nargs='+',
        type=float,
        default=(1.6,),
        metavar='LEVEL',
        help=(
            "Legacy hierarchy part-code noise levels for generation_noise_sweep "
            "(range: 0.0-2.4; default: 1.6)"
        ),
    )
    parser.add_argument("--mapping_n_iter_decoding", default=None, type=int, help="Mapping: iteration times of inner loop")
    parser.add_argument("--mapping_n_hier_infer", default=None, type=int, help="Mapping: how many hierarchies to infer")

    parser.add_argument("--mapping_test_name_task", default=None, type=str, help="Mapping: whether use gt code")


    parser.add_argument('--use_pretrained_mapping_ckpt', action='store_true', help='Whether using pretrained mapping ckpt')

    # mapping_idx_seg
    parser.add_argument("--mapping_idx_seg", default=None, type=int, help="Mapping: idx of seg")
    args = parser.parse_args(argv)
    return args
