# Path Configuration

StructRe keeps path configuration in `confs/path.py`. All paths have repository-relative defaults, so importing the module does not require local datasets, checkpoints, or a recognized hostname.

## Core Variables

| Variable | Default | Purpose |
|---|---|---|
| `STRUCTRE_DATA_ROOT` | `<repo>/data` | Parent directory for dataset defaults |
| `STRUCTRE_EXPERIMENT_ROOT` | `<repo>/exps` | Training outputs and default checkpoint layout |
| `STRUCTRE_STAGE1_ARTIFACT_ID` | Default Stage 1 artifact ID | Names Stage 1 code folders and its default checkpoint directory |
| `STRUCTRE_STAGE2_ARTIFACT_ID` | Default Stage 2 artifact ID | Names Stage 2 code and hierarchy-pair folders and its default checkpoint directory |
| `STRUCTRE_STAGE1_CHECKPOINT` | Stage 1 path under `STRUCTRE_EXPERIMENT_ROOT` | Sparse VAE checkpoint |
| `STRUCTRE_STAGE2_CHECKPOINT` | Stage 2 path under `STRUCTRE_EXPERIMENT_ROOT` | Part VAE checkpoint |
| `STRUCTRE_STAGE3_CHECKPOINT` | Stage 3 path under `STRUCTRE_EXPERIMENT_ROOT` | Rewriting model checkpoint |
| `STRUCTRE_PARTNET_ROOT` | `<data>/partnet/data_hierarchy_v3` | Processed PartNet root |
| `STRUCTRE_SHAPENET_PROCESSED_ROOT` | `<data>/shapenet_ours_v2` | Processed ShapeNet root |

Artifact IDs are single directory names, not paths. Stage 1 extraction and the Stage 2 loader share `STRUCTRE_STAGE1_ARTIFACT_ID`; Stage 2 extraction, hierarchy-pair preprocessing, and the Stage 3 loader share `STRUCTRE_STAGE2_ARTIFACT_ID`. Keeping each producer and its consumers on the same ID prevents a newly trained checkpoint from silently using legacy code folders.

## Checkpoint Precedence

`--path_ckpt` selects the checkpoint for the model currently run by `exp_runner.py`. A bare filename resolves under the current experiment's `checkpoints/` directory; an absolute path or a relative path containing a directory component is used directly. A requested file that does not exist raises `FileNotFoundError` instead of silently loading another checkpoint. Downstream models can override nested dependencies separately:

| CLI argument | Dependency |
|---|---|
| `--path_ckpt_stage1` | Sparse VAE loaded by Stage 2/3 |
| `--path_ckpt_stage2` | Part VAE loaded by Stage 3 |
| `--path_ckpt_stage3` | Rewriting model used for pretrained initialization |

The precedence is stage-specific CLI argument, then `STRUCTRE_STAGE*_CHECKPOINT`, then the repository-relative default.

## Example

```bash
export STRUCTRE_DATA_ROOT=/datasets/structre
export STRUCTRE_EXPERIMENT_ROOT=/outputs/structre
export STRUCTRE_STAGE1_ARTIFACT_ID=stage1-public-v1
export STRUCTRE_STAGE2_ARTIFACT_ID=stage2-public-v1
export STRUCTRE_STAGE1_CHECKPOINT=/checkpoints/structre/stage1.pt
export STRUCTRE_STAGE2_CHECKPOINT=/checkpoints/structre/stage2.pt
export STRUCTRE_STAGE3_CHECKPOINT=/checkpoints/structre/stage3.pt

python -c "import confs.path as p; print(p.DIR_ROOT_PARTNET)"

python exp_runner.py --conf ./confs/structmapping.conf \
    --path_ckpt_stage1 /checkpoints/structre/stage1.pt \
    --path_ckpt_stage2 /checkpoints/structre/stage2.pt
```

Paths are normalized to absolute paths but are not created or validated during import. Training and inference callers should validate the resources they actually use and report missing files at that boundary.

## Additional Overrides

The following variables support optional datasets and workflows:

- `STRUCTRE_COLOR_MAP`
- `STRUCTRE_VIEWPOINT_MESH`
- `STRUCTRE_IS_SERVER`

`Map_Color` and `Lis_ViewPoints_Sphere` remain deterministic in-memory fallback values. Production consumers call `get_color_map()` and `get_viewpoints_sphere()` at runtime: existing `.npy` and `.ply` resources are loaded from disk, while missing resources use the deterministic fallback.
