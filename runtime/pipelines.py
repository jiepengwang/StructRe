"""Runtime-free descriptions of the three-stage training pipelines."""

PIPELINE_RUNTIME = (
    'complete StructRe ML runtime, CUDA support, and prepared stage data'
)


PIPELINES = {
    'stage1': (
        {
            'step': 'train_sparse_vae',
            'command': (
                '--conf ./confs/sparse_vae.conf --mode train '
                '--exp_name $STRUCTRE_STAGE1_ARTIFACT_ID '
                '--category <category>'
            ),
            'requires': (
                'Stage 1 complete-shape HDF5 and sparse_vae_aug manifests'
            ),
            'input': 'STRUCTRE_PARTNET_ROOT/geometry/pointcloud_h5',
            'output': (
                'STRUCTRE_EXPERIMENT_ROOT/sparse_vae/'
                '$STRUCTRE_STAGE1_ARTIFACT_ID/checkpoints'
            ),
        },
        {
            'step': 'extract_part_codes',
            'command': (
                '--conf ./confs/sparse_vae.conf --mode infer_code --is_continue '
                '--path_ckpt $STRUCTRE_STAGE1_CHECKPOINT '
                '--category <category> --sparse_vae_dataset_name partnet '
                '--sparse_vae_mode_shape shape_parts '
                '--extract_mode <train|test>'
            ),
            'requires': (
                'Stage 1 checkpoint, part HDF5, and sparse_vae manifests'
            ),
            'input': (
                'STRUCTRE_PARTNET_ROOT/geometry/'
                'pointcloud_part_h5_aug_scale'
            ),
            'output': (
                'STRUCTRE_PARTNET_ROOT/code/code_sparse_vae/'
                '$STRUCTRE_STAGE1_ARTIFACT_ID'
            ),
        },
    ),
    'stage2': (
        {
            'step': 'train_part_vae',
            'command': (
                '--conf ./confs/part_vae.conf --mode train '
                '--exp_name $STRUCTRE_STAGE2_ARTIFACT_ID '
                '--category <category>'
            ),
            'requires': (
                'Stage 1 part codes and part_vae_sparse_vae manifests'
            ),
            'input': (
                'STRUCTRE_PARTNET_ROOT/code/code_sparse_vae/'
                '$STRUCTRE_STAGE1_ARTIFACT_ID'
            ),
            'output': (
                'STRUCTRE_EXPERIMENT_ROOT/part_vae/'
                '$STRUCTRE_STAGE2_ARTIFACT_ID/checkpoints'
            ),
        },
        {
            'step': 'extract_struct_codes',
            'command': (
                '--conf ./confs/part_vae.conf --mode infer_code --is_continue '
                '--path_ckpt $STRUCTRE_STAGE2_CHECKPOINT '
                '--category <category> --extract_mode <train|test>'
            ),
            'requires': 'Stage 2 checkpoint and Stage 1 part codes',
            'input': (
                'STRUCTRE_PARTNET_ROOT/code/code_sparse_vae/'
                '$STRUCTRE_STAGE1_ARTIFACT_ID'
            ),
            'output': (
                'STRUCTRE_PARTNET_ROOT/code/code_struct/'
                '$STRUCTRE_STAGE2_ARTIFACT_ID'
            ),
        },
    ),
    'stage3': (
        {
            'step': 'train_structmapping',
            'command': (
                '--conf ./confs/structmapping.conf --mode train '
                '--exp_name <stage3-experiment> --category <category> '
            ),
            'requires': (
                'hierarchy-pair HDF5 and mapping_sparse_vae manifests'
            ),
            'input': (
                'STRUCTRE_PARTNET_ROOT/hierarchy/hierarchy_pairs2/'
                '$STRUCTRE_STAGE2_ARTIFACT_ID'
            ),
            'output': (
                'STRUCTRE_EXPERIMENT_ROOT/structmapping/'
                '<stage3-experiment>/checkpoints'
            ),
        },
        {
            'step': 'validate_hierarchy',
            'command': (
                '--conf ./confs/structmapping.conf --mode validate '
                '--exp_name <stage3-experiment> '
                '--is_continue --path_ckpt $STRUCTRE_STAGE3_CHECKPOINT '
                '--path_ckpt_stage1 $STRUCTRE_STAGE1_CHECKPOINT '
                '--path_ckpt_stage2 $STRUCTRE_STAGE2_CHECKPOINT '
                '--category <category> --extract_mode test '
                '--should_load_sparse_vae'
            ),
            'requires': (
                'Stage 1, Stage 2, and Stage 3 checkpoints plus '
                'hierarchy-pair test data'
            ),
            'input': (
                'STRUCTRE_PARTNET_ROOT/hierarchy/hierarchy_pairs2/'
                '$STRUCTRE_STAGE2_ARTIFACT_ID'
            ),
            'output': (
                'STRUCTRE_EXPERIMENT_ROOT/structmapping/'
                '<stage3-experiment>/infer_test*'
            ),
        },
        {
            'step': 'extract_custom_mesh_codes',
            'command': (
                '--conf ./confs/structmapping.conf --mode infer_rewriting '
                '--exp_name <stage3-experiment> '
                '--is_continue --path_ckpt $STRUCTRE_STAGE3_CHECKPOINT '
                '--path_ckpt_stage1 $STRUCTRE_STAGE1_CHECKPOINT '
                '--path_ckpt_stage2 $STRUCTRE_STAGE2_CHECKPOINT '
                '--category <category> --use_no_dataloader '
                '--should_load_sparse_vae '
                '--mapping_test_name_task mesh_encode '
                '--mapping_input_mesh_dir <ply-directory> '
                '--mapping_shape_code_path <shape-codes.h5> '
            ),
            'requires': 'Stage 1, Stage 2, and Stage 3 checkpoints',
            'input': '<ply-directory>',
            'output': '<shape-codes.h5>',
        },
        {
            'step': 'generate_from_shape_codes',
            'command': (
                '--conf ./confs/structmapping.conf --mode infer_rewriting '
                '--exp_name <stage3-experiment> '
                '--is_continue --path_ckpt $STRUCTRE_STAGE3_CHECKPOINT '
                '--path_ckpt_stage1 $STRUCTRE_STAGE1_CHECKPOINT '
                '--path_ckpt_stage2 $STRUCTRE_STAGE2_CHECKPOINT '
                '--category <category> --use_no_dataloader '
                '--should_load_sparse_vae '
                '--mapping_test_name_task test_mesh_uncond_gen '
                '--mapping_shape_code_path <shape-codes.h5> '
            ),
            'requires': 'Stage 1, Stage 2, and Stage 3 checkpoints',
            'input': '<shape-codes.h5>',
            'output': (
                'STRUCTRE_EXPERIMENT_ROOT/structmapping/'
                '<stage3-experiment>/infer_test*'
            ),
        },
    ),
}


def format_pipeline(name):
    lines = [f'Pipeline: {name}', f'Runtime: {PIPELINE_RUNTIME}']
    for index, step in enumerate(PIPELINES[name], start=1):
        lines.extend((
            f'{index}. {step["step"]}',
            f'   args: {step["command"]}',
            f'   requires: {step["requires"]}',
            f'   input: {step["input"]}',
            f'   output: {step["output"]}',
        ))
    return '\n'.join(lines)
