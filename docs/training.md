# Training And Inference

StructRe provides one entry point with explicit modes for training, dataloader-backed validation, code extraction, and rewriting inference. Install the platform-specific PyTorch, MinkowskiEngine, and PyTorch3D builds first, then install the shared Python runtime:

```bash
python -m pip install -r requirements/runtime.txt
```

See `docs/environment.md` for the reference CUDA 12.8 software stack. CUDA extensions must be rebuilt for other PyTorch, CUDA, compiler, or GPU combinations.

Inspect each model stage without loading the ML or geometry runtime:

```bash
python exp_runner.py --describe_pipeline stage1
python exp_runner.py --describe_pipeline stage2
python exp_runner.py --describe_pipeline stage3
```

## Shared Configuration

Choose stable artifact IDs before training. Use the same value as the experiment name so checkpoints and extracted codes remain visibly paired:

```bash
export STRUCTRE_PARTNET_ROOT=/datasets/partnet/data_hierarchy_v3
export STRUCTRE_EXPERIMENT_ROOT=/outputs/structre
export STRUCTRE_STAGE1_ARTIFACT_ID=stage1-public-v1
export STRUCTRE_STAGE2_ARTIFACT_ID=stage2-public-v1
export STRUCTRE_STAGE1_CHECKPOINT=/checkpoints/structre/stage1.pt
export STRUCTRE_STAGE2_CHECKPOINT=/checkpoints/structre/stage2.pt
export STRUCTRE_STAGE3_CHECKPOINT=/checkpoints/structre/stage3.pt
```

`STRUCTRE_STAGE1_CHECKPOINT`, `STRUCTRE_STAGE2_CHECKPOINT`, and `STRUCTRE_STAGE3_CHECKPOINT` may point directly to downloaded or locally trained weights. The equivalent `--path_ckpt_stage1`, `--path_ckpt_stage2`, and `--path_ckpt_stage3` arguments override nested model dependencies. `--path_ckpt` always selects the active model's checkpoint.

## Stage 1: SparseVAE

Training consumes prepared complete-shape artifacts; the raw PartNet/ShapeNet adapters and producers are not distributed. Set `STRUCTRE_PARTNET_ROOT` to a root containing:

```text
info.json
filelists/<category>_<train-or-test>.lst
filelists/sparse_vae_aug/<category>_<train-or-test>_succeed.json
geometry/pointcloud_h5/<category>/<model-id>/<sample>.h5
```

Each Stage 1 HDF5 contains `pointcloud: [N, 3]`, `normal: [N, 3]`, and either dense `tsdf` or sparse `mask_valid` + `tsdf_valid` + `thres_trunc` fields for a 256-cubed grid. The succeed manifest maps each model ID to sample paths relative to its category/model directory. With both train and test manifests present:

```bash
python exp_runner.py --mode train --conf ./confs/sparse_vae.conf --exp_name "$STRUCTRE_STAGE1_ARTIFACT_ID" --category chair
```

Stage 2 consumes codes extracted from prepared hierarchy-part HDF5. In addition to the base `<category>_<split>.lst`, provide `filelists/sparse_vae/<category>_<split>_succeed.json` and samples beneath `geometry/pointcloud_part_h5_aug_scale/<category>/<model-id>/`. Each selected part HDF5 contains `pointcloud` and `normal`; Stage 1 code extraction does not read TSDF for `shape_parts`. Run both splits with the trained Stage 1 checkpoint:

```bash
python exp_runner.py --mode infer_code --conf ./confs/sparse_vae.conf \
    --is_continue --path_ckpt "$STRUCTRE_STAGE1_CHECKPOINT" \
    --category chair --sparse_vae_dataset_name partnet \
    --sparse_vae_mode_shape shape_parts --extract_mode train
```

Repeat with `--extract_mode test`. Before Stage 2 training, the prepared root must include `filelists/part_vae_sparse_vae/<category>_<split>_succeed.json`. Each manifest maps model IDs to relative sample directories beneath `<partnet>/code/code_sparse_vae/$STRUCTRE_STAGE1_ARTIFACT_ID/<category>/<model-id>/`; every selected sample directory contains `code.h5` with `code` and integer `code_coords` arrays. For PartNet, the sample-directory basename must encode its hierarchy level in the second underscore-delimited component, `level<integer>` (for example, `00_level1`), because the loader derives `level_id` from that component.

## Stage 2: PartVAE

```bash
python exp_runner.py --mode train --conf ./confs/part_vae.conf \
    --exp_name "$STRUCTRE_STAGE2_ARTIFACT_ID" --category chair

python exp_runner.py --mode infer_code --conf ./confs/part_vae.conf \
    --is_continue --path_ckpt "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category chair --extract_mode train
```

Repeat extraction with `--extract_mode test`. The producer and consumer share these paths:

```text
<partnet>/code/code_sparse_vae/$STRUCTRE_STAGE1_ARTIFACT_ID
  -> PartVAE
  -> <partnet>/code/code_struct/$STRUCTRE_STAGE2_ARTIFACT_ID
```

## Stage 3: StructMapping

Stage 3 reads hierarchy-pair HDF5 built from Stage 1 patch codes and Stage 2 structural codes. Raw-data and hierarchy-pair producers are not distributed; place prepared pairs under `<partnet>/hierarchy/hierarchy_pairs2/$STRUCTRE_STAGE2_ARTIFACT_ID/` with `mapping_sparse_vae` succeed manifests before training.

The matching `filelists/mapping_sparse_vae/<category>_<split>_succeed.json` maps model IDs to `<shape>/<view>/<hierarchy>.h5` sample paths. Every pair provides child and parent `bb`, `code`, and `code_idx` tensors plus `link_c2p`. A patch-level pair must include a scalar or single-element `is_patch` field equal to `1`; the loader promotes its base child tensors to patch tensors. A part-level pair must omit `is_patch` and additionally provide `patch_bb_child`, `patch_code_child`, and `patch_code_idx_child`. `code_idx_parent` ends with one negative EOS token, and the number of parent tokens must not exceed `model.decoder.detr.num_queries + 1`.

```bash
python exp_runner.py --mode train --conf ./confs/structmapping.conf \
    --exp_name stage3-public-v1 --category chair
```

Validate the trained rewriting model against hierarchy-pair test data:

`exp_runner.py --mode validate` automatically enables hierarchy decoding in direction 0.

```bash
python exp_runner.py --mode validate --conf ./confs/structmapping.conf \
    --exp_name stage3-public-v1 \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category chair --extract_mode test \
    --should_load_sparse_vae
```

## Custom Mesh Structured Reconstruction

`mesh_reconstruct` processes every `.ply` directly inside one directory in
sorted order. Each input mesh is sampled and encoded by Stage 1, rewritten
top-down by Stage 3, and decoded through Stage 2 and Stage 1 into per-level
structured meshes plus `tree_pred.pkl`. The input mesh is also the geometry
source; no PartNet filelist or ground-truth mesh is read.

Set `STRUCTRE_CUSTOM_MESH_DIR` to the input directory. `--category` must be a
category supported by the loaded checkpoint. Inputs should follow the training
normalization convention, approximately `[-1, 1]` per axis.

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-mesh-reconstruct \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category microwave --extract_mode test \
    --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task mesh_reconstruct \
    --mapping_input_mesh_dir "$STRUCTRE_CUSTOM_MESH_DIR" \
    --mapping_n_hier_infer 2
```

Outputs are written below
`exps/structmapping/<exp-name>/infer_test/sample0/mesh_merge/`. Each mesh gets a
separate directory containing `tree_pred.pkl` and merged `level_<N>.ply`
artifacts. `--model_id_start` and `--model_id_end` select a deterministic slice
of the sorted input list. The legacy task name `test_mesh` remains an alias.

Canonical rewriting tasks own their mapping direction. `--mapping_direction_infer` is
only needed by the dataloader hierarchy-validation entry.

Use `--mapping_test_name_task mesh_cycle` with the same explicit mesh directory
and category to run direction-0 reconstruction followed by direction-1 rewriting.
`--struct_add_partcode_noise` is required and accepts `0.0` for a no-noise
baseline. `--mapping_n_iter_rand N` is the only repeat control and produces one
cycle result per selected mesh in each of `N` sample directories. For example,
replace the task flag in the command above and add
`--struct_add_partcode_noise 0.4 --mapping_n_iter_rand 10`. The legacy alias is
`test_mesh_cycle_infer`.

## Completion From Partial Point Clouds

`completion` reads one explicit JSONL manifest and does not consult a PartNet
filelist or derive input paths from a global category. Set
`STRUCTRE_COMPLETION_MANIFEST` to that file:

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-completion \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --extract_mode test --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task completion \
    --mapping_input_manifest "$STRUCTRE_COMPLETION_MANIFEST" \
    --mapping_n_hier_infer 2
```

Each non-empty JSONL line is one object:

```json
{"id":"sample-001","category":"chair","partial_pointcloud":"partial/001.h5","gt_mesh":"optional/001.ply"}
```

`id`, `category`, and `partial_pointcloud` are required. `category` is
case-insensitive, canonicalized to lowercase, and must match a category supported
by `confs/path.py`. `gt_mesh` is optional and does not affect prediction.
Relative paths are resolved against the manifest directory; absolute paths are
also accepted. Each partial HDF5 must contain non-empty, numeric, finite
`pointcloud: [N, 3]`, `normal: [N, 3]`, and `lis_cam_sample: [V, 3]` arrays.
Point coordinates must use the Stage 1 normalization domain `[-1.15, 1.15]`;
camera samples must use the same coordinate frame and are required by the
current estimated-normal orientation step.
Duplicate category/id pairs and missing or malformed inputs are rejected before
inference. `--model_id_start` and `--model_id_end` select manifest rows, while
`--mapping_n_iter_rand` repeats each selected row.

Outputs use the normal StructMapping experiment directory and retain the input
`category/id` in their hierarchy artifact paths.

## Structured Reconstruction From Mesh Manifests

The three dataset reconstruction tasks read the same explicit JSONL mesh
manifest. They do not enumerate internal category lists, read dataset filelists,
or derive mesh paths from a configured data root. Each non-empty line is one
object:

```json
{"id":"sample-001","category":"chair","mesh":"meshes/001.ply"}
```

`id`, `category`, and `mesh` are required. Relative mesh paths resolve against
the manifest directory. IDs must be unique within each category, categories are
case-insensitive, and meshes must be existing `.ply` files. The canonical
`reconstruction` task accepts any category supported by `confs/path.py`;
`reconstruction_shapenet_batch` and `reconstruction_partnet_batch` use the same
loop but reject rows outside their respective category sets.

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-reconstruction \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --extract_mode test --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task reconstruction \
    --mapping_input_manifest "$STRUCTRE_RECONSTRUCTION_MANIFEST" \
    --mapping_n_hier_infer 2
```

`--model_id_start` and `--model_id_end` select manifest rows.
`--mapping_n_iter_rand` repeats every selected row; repeated runs use separate
`sampleNNN` directories. Outputs preserve the manifest `category/id` under the
normal StructMapping experiment directory. Legacy aliases for the two batch
tasks remain accepted.

## Dataset Generation Noise Sweep

`generation_noise_sweep` creates samples for downstream generation evaluation by
using the hierarchy part-code perturbation at each requested magnitude.
The perturbation affects initial patch conditioning and every nonterminal rewrite
output; it is not a root-only perturbation. The task does not calculate metrics;
evaluate the generated mesh/point-cloud sets with a downstream metric
implementation using the explicit result artifacts.

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-generation-sweep \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category chair --extract_mode test \
    --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task generation_noise_sweep \
    --mapping_noise_levels 0.8 1.2 1.6 \
    --mapping_n_iter_rand 4
```

`--mapping_noise_levels` accepts distinct, finite values in `[0.0, 2.4]` with
at most two decimal places. This range covers the historical generation sweep
used by the repository and prevents invalid model values and output paths.
Level `0.0` runs once as the no-noise baseline; each
nonzero level runs `--mapping_n_iter_rand` times. Output groups retain stable
tags such as `noise0_80` and `noise1_60`. Omitting the option preserves the
historical default `1.6`.

The noise sweep reads source shapes from the configured `STRUCTRE_PARTNET_ROOT` and `<category>_<extract_mode>.lst` filelist. Prepare those inputs before running the task.

## Custom Mesh Code Distribution And Generation

This workflow first encodes every `.ply` directly inside one directory into a shared shape-code HDF5. It then fits the existing Gaussian sampler to that code collection and runs bottom-up StructMapping generation. It is distribution-based generation from a mesh collection, not one-to-one reconstruction of each input mesh.

Set `STRUCTRE_CUSTOM_MESH_DIR` and `STRUCTRE_CUSTOM_SHAPE_CODES` to your input directory and desired HDF5 path, respectively.

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-public-v1 \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category chair --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task mesh_encode \
    --mapping_input_mesh_dir "$STRUCTRE_CUSTOM_MESH_DIR" \
    --mapping_shape_code_path "$STRUCTRE_CUSTOM_SHAPE_CODES"
```

The output HDF5 maps each input filename stem to its PartVAE shape code. Use the same file for generation:

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-public-v1 \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category chair --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task test_mesh_uncond_gen \
    --mapping_shape_code_path "$STRUCTRE_CUSTOM_SHAPE_CODES"
```

By default the generation task samples 1,000 codes and processes all of them. `--mapping_idx_seg N` limits one invocation to the zero-based block `N` of 500 samples. The sampled-code cache is written next to the HDF5 as `<stem>_sample_1k.npy` and is regenerated when the HDF5 is newer. Decoded temporary parts and final hierarchy outputs stay under the selected StructMapping experiment directory.

The same HDF5 can drive an explicit source-to-target interpolation. Keys are
the input filename stems written by `mesh_encode`; both endpoints are included.

```bash
python exp_runner.py --mode infer_rewriting --conf ./confs/structmapping.conf \
    --exp_name stage3-shape-code-interpolation \
    --is_continue --path_ckpt "$STRUCTRE_STAGE3_CHECKPOINT" \
    --path_ckpt_stage1 "$STRUCTRE_STAGE1_CHECKPOINT" \
    --path_ckpt_stage2 "$STRUCTRE_STAGE2_CHECKPOINT" \
    --category chair --use_no_dataloader --should_load_sparse_vae \
    --mapping_test_name_task shape_code_interpolate \
    --mapping_shape_code_path "$STRUCTRE_CUSTOM_SHAPE_CODES" \
    --mapping_shape_code_source <source-stem> \
    --mapping_shape_code_target <target-stem> \
    --mapping_shape_code_steps 6
```

These tasks require all three checkpoints because mesh encoding and bottom-up generation load the nested SparseVAE and PartVAE models. Run `python exp_runner.py --list_inference_tasks` to inspect task names, aliases, required options, stage dependencies, and artifact contracts without instantiating the models.
