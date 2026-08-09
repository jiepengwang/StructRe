# Environment

StructRe uses CUDA extensions that must match the active PyTorch, CUDA toolkit, compiler, and target GPU architecture. The versions below define the repository's reference software stack; they do not describe a specific machine.

## Reference Software Stack

| Component | Version |
|---|---|
| Python | 3.12 |
| CUDA toolkit | 12.8 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| MinkowskiEngine | 0.5.4 |
| PyTorch3D | 0.7.9 from commit `32a33e24428d07171ef54e359d902205eab95b9b` |

The dependency manifests have distinct roles:

- `requirements/runtime.txt` pins the direct Python dependencies used by StructRe.
- `requirements/platform-cu128.txt` installs the pip-resolvable PyTorch/CUDA 12.8 foundation.
- `requirements/environment-cu128.lock.txt` records a complete package snapshot for version comparison and debugging.

## Installation

Create and activate a Python 3.12 environment, then install the platform and runtime manifests:

```bash
python -m pip install -r requirements/platform-cu128.txt
python -m pip install -r requirements/runtime.txt
```

MinkowskiEngine and PyTorch3D must be compiled against the active PyTorch and CUDA toolkit. Build MinkowskiEngine 0.5.4 from a source revision compatible with that toolchain. The package version alone does not encode the source revision, patches, compiler flags, or target architecture, so this repository does not claim binary reproducibility from the version string alone.

Build PyTorch3D from the pinned upstream revision after PyTorch is installed:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH FORCE_CUDA=1 python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@32a33e24428d07171ef54e359d902205eab95b9b'
```

`--no-build-isolation` lets the PyTorch3D build use the already installed PyTorch package. Set compiler parallelism and CUDA architecture flags for the target deployment environment when needed.

## Verification

Check the installed dependency metadata:

```bash
python -m pip check
```

Model execution should be validated with the checkpoints and prepared artifacts required by the intended training or inference workflow.

## Compatibility Notes

- PyTorch 2.9.1 and torchvision 0.24.1 are an official version pair.
- PyTorch3D is installed from a pinned source revision because a matching prebuilt wheel may not be available.
- Rebuild MinkowskiEngine and PyTorch3D when changing PyTorch, CUDA, the compiler toolchain, or the target GPU architecture.
- Optional workflows may require packages such as libigl that are not part of the core runtime manifests.
