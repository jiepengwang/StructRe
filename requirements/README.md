# Dependency Manifests

The dependency files serve different purposes:

| File | Purpose |
|---|---|
| `runtime.txt` | Direct, platform-independent Python dependencies imported by the public three-stage runtime |
| `platform-cu128.txt` | Pip-installable foundation for the Python 3.12 / PyTorch 2.9.1 / CUDA 12.8 software stack |
| `environment-cu128.lock.txt` | Complete package snapshot for version comparison, including transitive and development packages |

Use `requirements/runtime.txt` as the portable project manifest. The environment lock is a reference snapshot, not a claim that every frozen package is required by StructRe and not a one-command installer for another CUDA platform.

The PyTorch3D source-build recipe and platform compatibility notes are in [`docs/environment.md`](../docs/environment.md).

## Reference Software Stack

| Component | Version |
|---|---|
| Python | 3.12 |
| CUDA toolkit | 12.8 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| MinkowskiEngine | 0.5.4 |
| PyTorch3D | 0.7.9 from commit `32a33e24428d07171ef54e359d902205eab95b9b` |

## Installation Order

Create and activate a Python 3.12 virtual environment, then install the pip-resolvable platform and project dependencies:

```bash
python -m pip install -r requirements/platform-cu128.txt
python -m pip install -r requirements/runtime.txt
```

Build MinkowskiEngine 0.5.4 against the active PyTorch, CUDA toolkit, compiler, and target architecture. The installed distribution reports its version but does not encode the exact source revision, patches, or build flags; the lock file therefore records the Python package version without claiming CUDA binary reproducibility.

Build the pinned PyTorch3D revision after PyTorch is installed:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH FORCE_CUDA=1 python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@32a33e24428d07171ef54e359d902205eab95b9b'
```

Finally, verify the installed package metadata:

```bash
python -m pip check
```

Use the lock snapshot to compare package versions when diagnosing environment differences. It includes transitive and development packages that are intentionally absent from the direct runtime manifest; keep `runtime.txt` as the source of direct project dependencies.
