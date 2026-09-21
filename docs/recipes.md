# MLIP recipes — install, fetch, and run each framework

<!-- GENERATED FILE — do not edit by hand. -->
<!-- Regenerate: python3 scripts/gen_recipes.py --write   ·   CI: --check -->

One section per framework: how to install it, how to get its weights, and the
calculator line that runs it. Blocks **A** and **B** are what each framework's own
documentation says, with the source URL linked. Block **C** comes verbatim from
`models.json` — keep those lines as written, with `$OMM` replaced by the absolute
path of your clone (`oh_my_mlip.resolve()` returns them already filled in).

Upstream install gives the supported way; the pin gives one combination of versions
known to work together.

```bash
export OMM=$(pwd)          # this clone
source $OMM/env.sh         # shared caches + D3/CUDA environment, once per shell
```

`<prefix>` is wherever that env is installed — `$OMM/envs/<env>` for a stock
`install.sh` build, or your own path if you adopted an existing env
(`scripts/adopt_env.py`). Resolve it programmatically with
`oh_my_mlip.resolve(<model>)["python"]` rather than hardcoding.

## The short path (let the hub do it)

The per-model blocks below are the *explicit* procedure — useful to audit what
happens, to run air-gapped, or to port elsewhere. Day to day, four commands cover
every model in this registry:

```bash
# 1. is the env already here?   ready -> skip to 3
python3 $OMM/scripts/setup_survey.py --table

# 2. build it (or adopt an env you already have, at zero disk cost)
$OMM/install.sh <env>
python3 $OMM/scripts/adopt_env.py <env> <prefix>      # alternative to building

# 3. materialise the weights (handles URL rewriting, HF auth, unpacking)
python3 -c "import sys; sys.path.insert(0,'$OMM'); from oh_my_mlip import fetch; \
    print(fetch.ensure_weights('<Framework>', version='<Variant>'))"

# 4. check it works: energy + forces, exit 0 on success
python3 $OMM/scripts/setup_verify.py <Variant> --json
```

A bare name resolves family-first, so pass `--version` to reach a variant whose
name its family shadows.

---

## SevenNet — env `sevennet`

pin: python 3.11.13 · torch==2.7.1+cu126 · variants: `SevenNet-MF-OMPA`, `SevenNet-Omni`

### A. install

Upstream ([source](https://sevennet.readthedocs.io/en/latest/)) — requires: Python >= 3.10, PyTorch >= 2.0.0:

```bash
pip install sevenn
```

Pinned combination:

```bash
conda env create -f $OMM/envs/sevennet.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.6.* cuda-nvrtc-dev=12.6.85 cuda-driver-dev=12.6.77 ase=3.28.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126 sevenn @ git+https://github.com/MDIL-SNU/SevenNet.git@e72eb2c9ebf6895a9d6fa3318f0ae4409d0f9f73 cuequivariance==0.8.0 cuequivariance-torch==0.8.0 e3nn==0.5.6 openequivariance==0.6.6 ninja==1.13.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html)):

```bash
# Checkpoints ship inside the sevenn package — pass the model keyword only:
#   '7net-mf-ompa' / 'SevenNet-mf-ompa',  '7net-omni' / 'SevenNet-Omni'
# modal='mpa' is the documented PBE-level default for both.
```

### C. ASE calculator

`SevenNet-MF-OMPA` *(default)*

```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator('7net-mf-ompa', modal='mpa', enable_oeq=True)
atoms.calc = calc
```

`SevenNet-Omni`

```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator('7net-omni', modal='mpa', enable_oeq=True)
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py SevenNet-MF-OMPA --json` (prints energy + forces; exit 0 on success).

> **Note:** Upstream also exposes enable_cueq / enable_flash. The pinned recipe uses enable_oeq=True, which works against the cuequivariance wheel it already installs — no separate accelerator install is needed.

---

## MACE — env `mace`

pin: python 3.11.13 · torch==2.7.1+cu126 · variants: `MACE-MPA-0`, `MACE-MH-1-OMAT`, `MACE-MH-1-OC20`

### A. install

Upstream ([source](https://github.com/ACEsuit/mace)):

```bash
pip install --upgrade pip
pip install mace-torch
```

Pinned combination:

```bash
conda env create -f $OMM/envs/mace.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.6.* ase=3.25.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126 ase==3.29.0 mace-torch @ git+https://github.com/ACEsuit/mace.git@100a29149d90a5945eddec1f0940bd88a6e3b363 cuequivariance==0.5.1 cuequivariance-torch==0.5.1 e3nn==0.4.4
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/ACEsuit/mace-foundations/releases)):

```bash
# mace_mp downloads by name into ~/.cache/mace (override with XDG_CACHE_HOME)
<prefix>/bin/python -c "from mace.calculators import mace_mp; mace_mp(model='medium-mpa-0', device='cpu')"
<prefix>/bin/python -c "from mace.calculators import mace_mp; mace_mp(model='mh-1', device='cpu')"
#
# recorded fingerprints — verify with: python3 $OMM/scripts/verify_weights_integrity.py
#   MACE-MPA-0: 75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638  (79,462,305 B)
#   MACE-MH-1-OMAT: a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47  (59,208,139 B)
#   MACE-MH-1-OC20: a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47  (59,208,139 B)
```

### C. ASE calculator

`MACE-MPA-0` *(default)*

```python
from mace.calculators import mace_mp
calc = mace_mp(model='medium-mpa-0', dispersion=False, default_dtype='float64', device='cuda')
atoms.calc = calc
```

`MACE-MH-1-OMAT`

```python
from mace.calculators import mace_mp
calc = mace_mp(model='mh-1', default_dtype='float64', device='cuda', head='omat_pbe')
atoms.calc = calc
```

`MACE-MH-1-OC20`

```python
from mace.calculators import mace_mp
calc = mace_mp(model='mh-1', default_dtype='float64', device='cuda', head='oc20_usemppbe')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py MACE-MPA-0 --json` (prints energy + forces; exit 0 on success).

> **Note:** MH-1 is multi-head: pass head='omat_pbe' or head='oc20_usemppbe' (there is no plain 'oc20' head). Both heads load the same weight file. default_dtype selects float32 or float64 — the calculator lines above use float64.

---

## NequIP — env `nequip`

pin: python 3.11.13 · torch==2.9.1+cu128 · variants: `NequIP-OAM-XL`, `NequIP-OAM-L`

### A. install

Upstream ([source](https://nequip.readthedocs.io/en/latest/guide/getting-started/install.html)):

```bash
pip install nequip
pip install openequivariance   # accelerator, separate
```

Pinned combination:

```bash
conda env create -f $OMM/envs/nequip.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.8.* cuda-nvrtc-dev=12.8.93 cuda-driver-dev=12.8.90 ase=3.26.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128 nequip==0.15.0 e3nn==0.5.7 openequivariance==0.4.1 ninja==1.13.0 setuptools==80.9.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://www.nequip.net/)):

```bash
# Download and compile are ONE command: nequip-compile takes a nequip.net URI
#   input_path = 'nequip.net:<group>/<model>:<version>'   (nequip-compile --help)
ARCH=$(<prefix>/bin/python -c "import torch;c=torch.cuda.get_device_capability();print(f'sm{c[0]}{c[1]}')")
PATH="<prefix>/bin:$PATH" MAX_JOBS=4 <prefix>/bin/nequip-compile \
    nequip.net:mir-group/NequIP-OAM-XL:0.1 \
    $OMM/models/compiled/$ARCH/NequIP-OAM-XL_${ARCH}.nequip.pt2 \
    --mode aotinductor --device cuda --target ase \
    --modifiers enable_OpenEquivariance
# same for OAM-L (nequip.net:mir-group/NequIP-OAM-L:0.1)
# PATH must include <prefix>/bin: without it openequivariance cannot find ninja
#   and dies with 'Ninja is required to load C++ extensions'.
# MAX_JOBS=4 is required: oeq's first import JIT-builds with ninja at nproc and
#   can wedge on a stale build lock.
```

### C. ASE calculator

`NequIP-OAM-XL` *(default)*

```python
import openequivariance
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(compile_path='$OMM/models/compiled/${OMM_ARCH}/NequIP-OAM-XL_${OMM_ARCH}.nequip.pt2', device='cuda')
atoms.calc = calc
```

`NequIP-OAM-L`

```python
import openequivariance
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(compile_path='$OMM/models/compiled/${OMM_ARCH}/NequIP-OAM-L_${OMM_ARCH}.nequip.pt2', device='cuda')
atoms.calc = calc
```

Run: `MAX_JOBS=4 <prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py NequIP-OAM-XL --json` (prints energy + forces; exit 0 on success).

> **Note:** NequIPCalculator only loads via from_compiled_model, so the per-arch .pt2 is required rather than optional and must be rebuilt for each GPU architecture.

---

## Allegro — env `allegro`

pin: python 3.11.13 · torch==2.8.0+cu128 · variants: `Allegro-OAM-L`

### A. install

Upstream ([source](https://github.com/mir-group/allegro)):

```bash
pip install nequip-allegro
```

Pinned combination:

```bash
conda env create -f $OMM/envs/allegro.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.8.* ase=3.29.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 nequip==0.15.0 nequip-allegro==0.7.1 cuequivariance==0.8.1 cuequivariance-torch==0.8.1 cuequivariance-ops-cu12==0.8.1 cuequivariance-ops-torch-cu12==0.8.1 e3nn==0.5.7 ninja==1.13.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://www.nequip.net/)):

```bash
# Same compiler and URI scheme as NequIP, with the CuEquivariance modifier
ARCH=$(<prefix>/bin/python -c "import torch;c=torch.cuda.get_device_capability();print(f'sm{c[0]}{c[1]}')")
PATH="<prefix>/bin:$PATH" <prefix>/bin/nequip-compile \
    nequip.net:mir-group/Allegro-OAM-L:0.1 \
    $OMM/models/compiled/$ARCH/Allegro-OAM-L_${ARCH}.nequip.pt2 \
    --mode aotinductor --device cuda --target ase \
    --modifiers enable_CuEquivarianceContracter
# Load: `import cuequivariance_torch` BEFORE NequIPCalculator.from_compiled_model,
#   else 'Could not find schema for cuequivariance_ops::tensor_product_uniform_1d_jit'.
# AOT Inductor + CuEquivariance does not support float64 models (use torchscript).
```

### C. ASE calculator

`Allegro-OAM-L`

```python
import cuequivariance_torch
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(compile_path='$OMM/models/compiled/${OMM_ARCH}/Allegro-OAM-L_${OMM_ARCH}.nequip.pt2', device='cuda')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py Allegro-OAM-L --json` (prints energy + forces; exit 0 on success).

> **Note:** Allegro's recommended accelerated path is CuEquivariance: compile with --modifiers enable_CuEquivarianceContracter (needs cuequivariance-torch + cuequivariance-ops-torch-cu12) and import cuequivariance_torch before loading.

---

## Nequix — env `nequix`

pin: python 3.11.13 · torch==2.10.0+cu126 · variants: `Nequix-MP-1`

### A. install

Upstream ([source](https://github.com/atomicarchitects/nequix)) — requires: JAX backend:

```bash
pip install nequix
```

Pinned combination:

```bash
conda env create -f $OMM/envs/nequix.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.6.* ase=3.29.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 nequix==0.4.3 jax==0.6.2 jax-cuda12-plugin==0.6.2 jax-cuda12-pjrt==0.6.2 jaxlib==0.6.2 e3nn-jax==0.20.8 equinox==0.13.6
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/atomicarchitects/nequix)):

```bash
# Upstream supports alias auto-download: NequixCalculator('nequix-mp-1').
# The commit-pinned file below is fetched directly so the digest can be checked.
mkdir -p $OMM/models/nequix
curl -L "https://github.com/atomicarchitects/nequix/raw/7c2854de8e754b1a60274c7d9d2e014989ed632e/models/nequix-mp-1.nqx" -o $OMM/models/nequix/nequix-mp-1.nqx
echo "1647af8e627e1afa11e2bfadeab0797326513a521afd7dd01606a96f56783eee  $OMM/models/nequix/nequix-mp-1.nqx" | sha256sum -c -
# Keep the .nqx extension — renaming to .pt makes the loader treat it as torch.
```

### C. ASE calculator

`Nequix-MP-1`

```python
from nequix.calculator import NequixCalculator
calc = NequixCalculator(model_path='$OMM/models/nequix/nequix-mp-1.nqx', backend='jax', use_kernel=False)
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py Nequix-MP-1 --json` (prints energy + forces; exit 0 on success).

> **Note:** use_kernel toggles the fused kernel; the calculator line above leaves it off.

---

## DeePMD — env `deepmd`

pin: python 3.11.13 · torch==2.8.0+cu128 · variants: `DPA-3.1-3M-FT`

### A. install

Upstream ([source](https://docs.deepmodeling.com/projects/deepmd/)) — requires: PyTorch backend:

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/deepmodeling/deepmd-kit@v3.1.0
```

Pinned combination:

```bash
conda env create -f $OMM/envs/deepmd.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.8.* ase=3.27.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 deepmd-kit==3.1.2 setuptools==70.2.0 mpich==5.0.1 dpdata==1.1.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M)):

```bash
# A multi-task training checkpoint: it must be frozen to one head before inference
mkdir -p $OMM/models/deepmd
curl -L "https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt" \
     -o $OMM/models/deepmd/DPA-3.1-3M.pt
<prefix>/bin/dp --pt freeze -c $OMM/models/deepmd/DPA-3.1-3M.pt \
     -o $OMM/models/deepmd/frozen-omat24.pth --head Omat24
# upstream equivalent: dp --pt freeze -c <ckpt> -o frozen_model.pth --model-branch <head>
#
# recorded fingerprints — verify with: python3 $OMM/scripts/verify_weights_integrity.py
#   DPA-3.1-3M-FT: 86dd3a804d78ca5d203ebf98747e8f16dff9713ba8950097ceb760b161e19907  (47,176,032 B)
```

### C. ASE calculator

`DPA-3.1-3M-FT`

```python
from deepmd.calculator import DP
calc = DP(model='$OMM/models/deepmd/frozen-omat24.pth')
atoms.calc = calc
```

Run: `LD_LIBRARY_PATH="" <prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py DPA-3.1-3M-FT --json` (prints energy + forces; exit 0 on success).

> **Note:** The published checkpoint is multi-task, so it must be frozen to a single head before inference. The recorded digest is for the downloaded checkpoint, not the frozen output. At run time DeePMD needs LD_LIBRARY_PATH="" to avoid an Intel oneAPI libfabric clash.

---

## ORB — env `orb`

pin: python 3.11.13 · torch==2.7.1+cu126 · variants: `ORB-v3`

### A. install

Upstream ([source](https://github.com/orbital-materials/orb-models)):

```bash
pip install orb-models
```

Pinned combination:

```bash
conda env create -f $OMM/envs/orb.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.6.* ase=3.27.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126 orb-models==0.5.4 torch-dftd==0.5.1
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/orbital-materials/orb-models/blob/main/MODELS.md)):

```bash
# Calling the named pretrained function is what triggers the download
<prefix>/bin/python -c "from orb_models.forcefield import pretrained; \
    pretrained.orb_v3_conservative_inf_omat(device='cpu', precision='float32-high')"
#
# recorded fingerprints — verify with: python3 $OMM/scripts/verify_weights_integrity.py
#   ORB-v3: 0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0  (102,097,517 B)
```

### C. ASE calculator

`ORB-v3`

```python
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
orbff = pretrained.orb_v3_conservative_inf_omat(device='cuda', precision='float32-high')
calc = ORBCalculator(orbff, device='cuda')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py ORB-v3 --json` (prints energy + forces; exit 0 on success).

> **Note:** Upstream's README unpacks a (orbff, atoms_adapter) tuple; the pinned version returns a single object, so use the calculator line above. precision='float32-high' selects TF32 matmul, which makes repeated runs numerically close rather than bit-identical.

---

## GRACE — env `grace`

pin: python 3.11.11 · — · variants: `GRACE-2L-OAM`

### A. install

Upstream ([source](https://gracemaker.readthedocs.io/)) — requires: TensorFlow backend (no torch):

```bash
pip install tensorpotential
```

Pinned combination:

```bash
conda env create -f $OMM/envs/grace.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge python=3.11.11 pip ase=3.29.0
#     <prefix>/bin/pip install tensorflow[and-cuda]==2.16.2 tensorpotential==0.5.3
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/)):

```bash
# Upstream CLI (documented): list / download / checkpoint
<prefix>/bin/grace_models list
<prefix>/bin/grace_models download GRACE-2L-OAM
#   default cache $HOME/.cache/grace  (override with GRACE_CACHE)
# The SavedModel lands nested (<t>/<n>/<n>/saved_model.pb) and must be flattened
# so saved_model.pb sits directly under the target directory:
python3 $OMM/scripts/prepare_grace_weights.py --name GRACE-2L-OAM \
    --target-dir $OMM/models/grace/GRACE-2L-OAM
#
# recorded fingerprints — verify with: python3 $OMM/scripts/verify_weights_integrity.py
#   GRACE-2L-OAM: b4e5d384e8e3f5222225748f6eed8479c2363cf0022e9ca8dfa4a4f2fe4325ec  (97,295,519 B)
```

### C. ASE calculator

`GRACE-2L-OAM`

```python
from tensorpotential.calculator import TPCalculator
calc = TPCalculator('$OMM/models/grace/GRACE-2L-OAM')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py GRACE-2L-OAM --json` (prints energy + forces; exit 0 on success).

> **Note:** TensorFlow backend, so D3 dispersion (which needs torch) is unavailable. grace_models leaves the SavedModel nested and it must be flattened before TPCalculator can load it; the digest covers the whole SavedModel tree, not a single file. On GPU, TF 2.16 loads libcudnn.so.8 while catbench's torch installs cuDNN 9 — sideload cuDNN 8.9 or TF skips GPU registration silently. Upstream also offers the grace_fm('<name>') convenience loader.

---

## MatterSim — env `mattersim`

pin: python 3.10.16 · torch==2.6.0+cu124 · variants: `MatterSim-v1-5M`

### A. install

Upstream ([source](https://github.com/microsoft/mattersim)):

```bash
pip install mattersim
```

Pinned combination:

```bash
conda env create -f $OMM/envs/mattersim.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.10.16 pip cuda-nvcc=12.4.* ase=3.24.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 mattersim @ git+https://github.com/microsoft/mattersim.git@bf13be6f6fd470fd4e299855c755ad13968620bb torch-geometric==2.6.1 e3nn==0.5.6 setuptools==75.8.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/microsoft/mattersim)):

```bash
# Checkpoints ship in the package (./pretrained_models/); 1M is the default,
# the 5M model is selected with load_path.
```

### C. ASE calculator

`MatterSim-v1-5M`

```python
from mattersim.forcefield import MatterSimCalculator
calc = MatterSimCalculator(load_path='MatterSim-v1.0.0-5M.pth', device='cuda')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py MatterSim-v1-5M --json` (prints energy + forces; exit 0 on success).

> **Note:** The package ships both the 1M and 5M checkpoints and defaults to 1M; load_path selects the 5M one. Pass device explicitly — otherwise the calculator can fall back to CPU.

---

## CHGNet — env `chgnet`

pin: python 3.11.13 · torch==2.7.1+cu126 · variants: `CHGNet-v0.3.0`

### A. install

Upstream ([source](https://github.com/CederGroupHub/chgnet)):

```bash
pip install chgnet
```

Pinned combination:

```bash
conda env create -f $OMM/envs/chgnet.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.6.* ase=3.25.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126 chgnet==0.4.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/CederGroupHub/chgnet)):

```bash
# CHGNet.load() prepares the weights itself — no download command needed.
```

### C. ASE calculator

`CHGNet-v0.3.0`

```python
from chgnet.model import CHGNet, CHGNetCalculator
model = CHGNet.load(model_name='0.3.0')
calc = CHGNetCalculator(model=model)
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py CHGNet-v0.3.0 --json` (prints energy + forces; exit 0 on success).

> **Note:** CHGNet.load() takes a model_name, so the version is selected in code rather than at download time.

---

## AlphaNet — env `alphanet`

pin: python 3.11.13 · torch==2.1.2+cu121 · variants: `AlphaNet-v1-OMA`

### A. install

Upstream ([source](https://github.com/zmyybc/AlphaNet)) — requires: Upstream README shows python 3.8; the pinned recipe uses python 3.11 + torch 2.1.2+cu121:

```bash
git clone https://github.com/zmyybc/AlphaNet.git
cd AlphaNet
pip install -e .
```

Pinned combination:

```bash
bash $OMM/envs/alphanet.build.sh <prefix>      # single conda solve impossible -> the sidecar owns the build
#   what the sidecar actually runs:
#     conda create -y --prefix <prefix> -c conda-forge -c nvidia \
#     python=3.11.13 "cuda-nvcc=12.1.*" ase pip
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
#     -f https://data.pyg.org/whl/torch-2.1.2+cu121.html \
#     torch==2.1.2+cu121 torch-geometric==2.6.1 torch_scatter==2.1.2+pt21cu121 \
#     lightning==2.6.6 tensorboard==2.21.0 "numpy==1.26.4"
#     <prefix>/bin/pip install --no-deps "alphanet @ git+https://github.com/zmyybc/AlphaNet.git@65f8ea9330459e0106867d1c694aec4139c6cb19"
#     <prefix>/bin/pip install pydantic==2.13.5 pydantic_settings==2.15.0 rich==15.0.0 scikit-learn==1.9.1 "numpy==1.26.4"
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/zmyybc/AlphaNet)):

```bash
# Both the figshare checkpoint AND the in-repo config are required
mkdir -p $OMM/models/alphanet
curl -L "https://ndownloader.figshare.com/files/53851139" -o $OMM/models/alphanet/alex_0410.ckpt
echo "879a477c09e2dd2614e029aeb352892ad21d0bdf80eaccbee40f47cf888e48c5  $OMM/models/alphanet/alex_0410.ckpt" | sha256sum -c -
curl -L "https://raw.githubusercontent.com/zmyybc/AlphaNet/65f8ea9330459e0106867d1c694aec4139c6cb19/pretrained/OMA/oma.json" \
     -o $OMM/models/alphanet/oma.json
```

### C. ASE calculator

`AlphaNet-v1-OMA`

```python
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.config import All_Config
calc = AlphaNetCalculator(ckpt_path='$OMM/models/alphanet/alex_0410.ckpt', device='cuda', precision='32', config=All_Config().from_json('$OMM/models/alphanet/oma.json'))
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py AlphaNet-v1-OMA --json` (prints energy + forces; exit 0 on success).

> **Note:** Needs both the checkpoint and its config JSON — loading the checkpoint alone fails. The recipe installs a pinned commit because package HEAD changes model behaviour. Do not install matscipy: it forces numpy>=2 and breaks the torch 2.1.2 ABI.

---

## Eqnorm — env `eqnorm`

pin: python 3.11.13 · torch==2.6.0+cu118 · variants: `Eqnorm-MPtrj`

### A. install

Upstream ([source](https://github.com/yzchen08/eqnorm)):

```bash
pip install git+https://github.com/yzchen08/eqnorm.git
```

Pinned combination:

```bash
conda env create -f $OMM/envs/eqnorm.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=11.8.* ase=3.25.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu118 --find-links https://data.pyg.org/whl/torch-2.6.0+cu118.html torch==2.6.0+cu118 eqnorm @ git+https://github.com/yzchen08/eqnorm.git@57527db44fee9fab3fb5775dd860ca4bbe014f44 torch-geometric==2.6.1 torch-scatter==2.1.2+pt26cu118 e3nn==0.5.6 vesin==0.3.7 setuptools==78.1.1
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/yzchen08/eqnorm)):

```bash
# Upstream: EqnormCalculator(model_variant='eqnorm-mptrj') downloads on its own.
# But that downloader hits plain figshare.com, which 202-blocks on some networks
# and leaves a 0-byte file the package then treats as present — never retrying.
# Pre-stage from the working subdomain instead:
mkdir -p ~/.cache/eqnorm
curl -L "https://ndownloader.figshare.com/files/55429685" -o ~/.cache/eqnorm/eqnorm-mptrj.pt
echo "9fd5b97a069e03697e41d2e4c468c5c9b487fc42a2842861ea171a23b9706de5  $HOME/.cache/eqnorm/eqnorm-mptrj.pt" | sha256sum -c -
```

### C. ASE calculator

`Eqnorm-MPtrj`

```python
from eqnorm.calculator import EqnormCalculator
calc = EqnormCalculator(model_name='eqnorm', model_variant='eqnorm-mptrj', device='cuda')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py Eqnorm-MPtrj --json` (prints energy + forces; exit 0 on success).

> **Note:** Upstream documents neither the cache path nor the download URL — both come from the package's own url_dict.

---

## fairchemv1 — env `fairchemv1`

pin: python 3.11.13 · torch==2.4.1+cu121 · variants: `eSEN-30M-OAM`

### A. install

Upstream ([source](https://github.com/facebookresearch/fairchem)):

```bash
pip install fairchem-core
```

Pinned combination:

```bash
conda env create -f $OMM/envs/fairchemv1.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.1.* ase=3.25.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 --find-links https://data.pyg.org/whl/torch-2.4.1+cu121.html torch==2.4.1+cu121 fairchem-core==1.10.0 scipy==1.16.0 torch-geometric==2.6.1 torch_scatter==2.1.2+pt24cu121 torch_sparse==0.6.18+pt24cu121 e3nn==0.5.6 setuptools==78.1.1
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://huggingface.co/facebook/OMAT24)):

```bash
# GATED: accept the facebook/OMAT24 license with your own HF account first
hf auth login
<prefix>/bin/python -c "from huggingface_hub import hf_hub_download; \
    print(hf_hub_download('facebook/OMAT24','esen_30m_oam.pt', local_dir='$OMM/models/fairchem'))"
echo "adf7d38e5bccb8e0334434c0bd65ac75661fb646891df17ecc89c19d111efde1  $OMM/models/fairchem/esen_30m_oam.pt" | sha256sum -c -
```

### C. ASE calculator

`eSEN-30M-OAM`

```python
from fairchem.core import OCPCalculator
calc = OCPCalculator(checkpoint_path='$OMM/models/fairchem/esen_30m_oam.pt', cpu=False)
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py eSEN-30M-OAM --json` (prints energy + forces; exit 0 on success).

> **Note:** Gated weights: the download fails without an accepted license and a token. OCPCalculator defaults to cpu=True, so pass cpu=False to run on GPU.

---

## EquiformerV3 — env `equiformer_v3`

pin: python 3.11.15 · torch==2.7.1+cu128 · variants: `EqV3-OMatMPtrjSalex`

### A. install

Upstream ([source](https://github.com/atomicarchitects/equiformer_v3)) — requires: scipy==1.16.0 (1.17 removed sph_harm):

```bash
# Upstream publishes no pip package — it vendors a fairchem fork (sha a7300c58),
# which envs/equiformer_v3.yml installs editable. There is no upstream install
# procedure to quote — use the pinned recipe below.
```

Pinned combination:

```bash
conda env create -f $OMM/envs/equiformer_v3.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.15 pip cuda-nvcc=12.8.* ase=3.29.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 --find-links https://data.pyg.org/whl/torch-2.7.1+cu128.html torch==2.7.1+cu128 torch-geometric==2.7.0 torch_scatter==2.1.2+pt27cu128 torch_sparse==0.6.18+pt27cu128 torch_cluster==1.6.3+pt27cu128 torch_spline_conv==1.2.2+pt27cu128 e3nn==0.5.6 scipy==1.16.0 fairchem-core @ git+https://github.com/atomicarchitects/equiformer_v3.git@a7300c58df683dc99cb48027d5bfd4c887486c48#subdirectory=packages/fairchem-core
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://huggingface.co/mirror-physics/equiformer_v3)):

```bash
mkdir -p $OMM/models/equiformer_v3/checkpoint
curl -L "https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24-mptrj-salex_gradient.pt" \
     -o $OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt
echo "429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98  $OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt" | sha256sum -c -
# The checkpoint/ path prefix is required — the bare filename 404s.
```

### C. ASE calculator

`EqV3-OMatMPtrjSalex`

```python
from fairchem.core import OCPCalculator
calc = OCPCalculator(checkpoint_path='$OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt', cpu=False)
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py EqV3-OMatMPtrjSalex --json` (prints energy + forces; exit 0 on success).

> **Note:** Upstream publishes no pip package — it vendors a fairchem fork, which the recipe installs editable. The checkpoint URL needs its checkpoint/ path prefix; the bare filename 404s.

---

## UMA — env `uma`

pin: python 3.11.13 · torch==2.8.0+cu128 · variants: `UMA-m-1p1-OC20`, `UMA-m-1p1-OMAT`, `UMA-s-1p1-OC20`, `UMA-s-1p1-OMAT`, `UMA-s-1p2-OC20`, `UMA-s-1p2-OC22`, `UMA-s-1p2-OC25`, `UMA-s-1p2-OMAT`

### A. install

Upstream ([source](https://github.com/facebookresearch/fairchem)):

```bash
pip install fairchem-core
```

Pinned combination:

```bash
conda env create -f $OMM/envs/uma.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.13 pip cuda-nvcc=12.8.* ase=3.27.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 fairchem-core==2.16.0 e3nn==0.5.6 scipy==1.16.0 setuptools==78.1.1
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://huggingface.co/facebook/UMA)):

```bash
# GATED: your HF account must be approved for facebook/UMA (upstream documents this)
hf auth login
<prefix>/bin/python -c "from fairchem.core import pretrained_mlip; \
    pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu')"
#   same pattern for uma-s-1p1 / uma-m-1p1 — three checkpoints cover the eight variants
```

### C. ASE calculator

`UMA-m-1p1-OC20`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-m-1p1', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc20')
atoms.calc = calc
```

`UMA-m-1p1-OMAT`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-m-1p1', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='omat')
atoms.calc = calc
```

`UMA-s-1p1-OC20`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-s-1p1', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc20')
atoms.calc = calc
```

`UMA-s-1p1-OMAT`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-s-1p1', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='omat')
atoms.calc = calc
```

`UMA-s-1p2-OC20`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc20')
atoms.calc = calc
```

`UMA-s-1p2-OC22`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc22')
atoms.calc = calc
```

`UMA-s-1p2-OC25`

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc25')
atoms.calc = calc
```

`UMA-s-1p2-OMAT` *(default)*

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='omat')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py UMA-s-1p2-OMAT --json` (prints energy + forces; exit 0 on success).

> **Note:** Gated weights: the download fails without an accepted license and a token. The checkpoint is uma-s-1p2, uma-s-1p1 or uma-m-1p1; task_name selects the variant — oc20 (catalysis), oc22 (oxides), oc25 (catalyst-electrolyte interfaces), omat (inorganic materials). The -m- checkpoint is 11.2 GB and is loaded into host RAM before it reaches the GPU, so it needs 32 GB or more of system memory; the -s- checkpoints do not.

---

## PET — env `pet`

pin: python 3.11.14 · torch==2.9.1+cu128 · variants: `PET-OAM-XL`

### A. install

Upstream ([source](https://github.com/metatensor/metatrain)):

```bash
pip install metatrain
```

Pinned combination:

```bash
conda env create -f $OMM/envs/pet.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.14 pip cuda-nvcc=12.8.* ase=3.27.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128 metatrain==2026.1 upet==0.1.0 metatomic-torch==0.1.7 metatensor-torch==0.8.3 setuptools==80.9.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://docs.metatensor.org/metatrain/latest/dev-docs/cli/export.html)):

```bash
# Only a .ckpt is published; inference needs an exported metatomic .pt
mkdir -p $OMM/models/pet
curl -L "https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt" \
     -o $OMM/models/pet/pet-oam-xl-v1.0.0.ckpt          # 2.92 GB
# upstream syntax: mtt export <path> <output>   (path may be local, an HF repo id, or a URL)
<prefix>/bin/mtt export $OMM/models/pet/pet-oam-xl-v1.0.0.ckpt \
     $OMM/models/pet/pet-oam-xl-v1.0.0.pt
```

### C. ASE calculator

`PET-OAM-XL`

```python
from metatomic.torch.ase_calculator import MetatomicCalculator
calc = MetatomicCalculator('$OMM/models/pet/pet-oam-xl-v1.0.0.pt', device='cuda')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py PET-OAM-XL --json` (prints energy + forces; exit 0 on success).

> **Note:** The export output is not byte-reproducible: its serialization differs between runs even for the same input.

---

## EquFlash — env `equflash`

pin: python 3.12.13 · torch==2.9.1+cu126 · variants: `EquFlashV2`, `EquFlash-v1`

### A. install

Upstream ([source](https://github.com/SamsungDS/GGNN)) — requires: Python 3.12, PyTorch 2.9.1 (CUDA 12.6) — upstream's stated requirement:

```bash
pip install -r requirements.txt
pip install --no-deps -r requirements-no-deps.txt
```

Pinned combination:

```bash
bash $OMM/envs/equflash.build.sh <prefix>      # single conda solve impossible -> the sidecar owns the build
#   what the sidecar actually runs:
#     conda create -y --prefix <prefix> -c conda-forge -c nvidia \
#     python=3.12.13 "cuda-nvcc=12.6.*" ase pip
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
#     -f https://data.pyg.org/whl/torch-2.9.1+cu126.html \
#     torch==2.9.1+cu126 torch-geometric==2.6.1 torch_scatter==2.1.2+pt29cu126 \
#     torch_sparse==0.6.18+pt29cu126 e3nn==0.5.6 cuequivariance==0.6.0 \
#     cuequivariance-torch==0.6.0 cuequivariance-ops-torch-cu12==0.6.0 \
#     "git+https://github.com/SamsungDS/GGNN@16b5cae474370977b59120e8bc57e4bcc19cd093#egg=GGNN"
#     <prefix>/bin/pip install --no-deps fairchem-core==1.10.0
#     <prefix>/bin/pip install pyyaml==6.0.3 lmdb==2.3.0 numba==0.67.0 scipy==1.16.0 pymatgen==2026.5.4 orjson==3.12.0 submitit==1.5.4 wandb==0.30.0 \
#     torchtnt==0.2.4 pydantic==2.13.5 huggingface_hub==1.31.0 hydra-core==1.3.6 tqdm==4.70.1 pynvml==13.0.1 \
#     nvalchemi-toolkit-ops==0.3.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/SamsungDS/GGNN)):

```bash
mkdir -p $OMM/models/equflash
curl -L "https://ndownloader.figshare.com/files/65435007" -o $OMM/models/equflash/EquFlashV2.pt
echo "068ce25767d5064fae41ffac352484ae59a8544ee671607a09dffdc6d62402a2  $OMM/models/equflash/EquFlashV2.pt" | sha256sum -c -
curl -L "https://ndownloader.figshare.com/files/65435004" -o $OMM/models/equflash/EquFlash.pt
echo "4a1eb19e4f719282146285dd1655178a0d1cc70edee7f8e1401e9b1336383bfa  $OMM/models/equflash/EquFlash.pt" | sha256sum -c -
# plain figshare.com 202-blocks — always use the ndownloader subdomain
```

### C. ASE calculator

`EquFlashV2` *(default)*

```python
from GGNN.common.calculator import UCalculator
calc = UCalculator(checkpoint_path='$OMM/models/equflash/EquFlashV2.pt', cpu=False)
atoms.calc = calc
```

`EquFlash-v1`

```python
from GGNN.common.calculator import UCalculator
calc = UCalculator(checkpoint_path='$OMM/models/equflash/EquFlash.pt', cpu=False)
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py EquFlashV2 --json` (prints energy + forces; exit 0 on success).

> **Note:** Upstream splits its requirements across two files, which is why a single conda solve cannot build this env and the recipe uses a two-pass sidecar. V2 is cueq-only — conv_type='flashtp' is a v1-only backend and V2 rejects it.

---

## MatRIS — env `matris`

pin: python 3.11.15 · torch==2.12.1+cu130 · variants: `MatRIS-10M-OAM`

### A. install

Upstream ([source](https://github.com/HPC-AI-Team/MatRIS)) — requires: torch > 2.6.0:

```bash
# Upstream states requirements but no install command:
#   ase>=3.23.0, numpy>=2.0.0, pymatgen>2024.9.10, torch>2.6.0
```

Pinned combination:

```bash
conda env create -f $OMM/envs/matris.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.15 pip cuda-nvcc=13.0.* ase=3.29.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.12.1+cu130 matris @ git+https://github.com/HPC-AI-Team/MatRIS@c16f569ca08e6905e91b64e2ee68614303e46f7f
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://github.com/HPC-AI-Team/MatRIS)):

```bash
# Upstream selects by model key (matris_10m_oam) and downloads internally.
# That downloader has the same 0-byte trap as eqnorm, so pre-stage it:
mkdir -p ~/.cache/matris
curl -L "https://ndownloader.figshare.com/files/59142728" -o ~/.cache/matris/MatRIS_10M_OAM.pth.tar
echo "c033abc53601a74f10d9b7fec0f658220c013c3b11d4d405f1d32136d4c2b067  $HOME/.cache/matris/MatRIS_10M_OAM.pth.tar" | sha256sum -c -
```

### C. ASE calculator

`MatRIS-10M-OAM`

```python
from matris.applications.base import MatRISCalculator
calc = MatRISCalculator(model='matris_10m_oam', task='efsm', device='cuda')
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py MatRIS-10M-OAM --json` (prints energy + forces; exit 0 on success).

> **Note:** Selected by model key. Built against torch cu130, so GPU use needs a CUDA-13-class driver — below that, run with device='cpu'.

---

## DPA4 — env `dpa4`

pin: python 3.11.15 · torch==2.11.0+cu130 · variants: `DPA-4.0.1-pro-MPtrj`

### A. install

Upstream ([source](https://docs.deepmodeling.com/projects/deepmd/)) — requires: PyTorch backend:

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/deepmodeling/deepmd-kit@v3.1.0
```

Pinned combination:

```bash
conda env create -f $OMM/envs/dpa4.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.15 pip cuda-nvcc=13.0.* ase=3.28.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 deepmd-kit==3.2.0b0 e3nn==0.6.0 vesin==0.5.8 vesin-torch==0.5.8 mpich==5.0.1 dpdata==1.1.0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://matbench-discovery.materialsproject.org/models/dpa-4.0.1-pro-mptrj)):

```bash
mkdir -p $OMM/models/dpa4
curl -L "https://ndownloader.figshare.com/files/65469204" -o $OMM/models/dpa4/dpa-4.0.1-pro-mptrj.pt
echo "446a6e893101fc52190cfccabb9d2328863d651aca36981fd8d3401c71a61221  $OMM/models/dpa4/dpa-4.0.1-pro-mptrj.pt" | sha256sum -c -
# (the registry's weights_source is the matbench-discovery model page; the
#  direct artifact is the figshare URL above)
```

### C. ASE calculator

`DPA-4.0.1-pro-MPtrj`

```python
from deepmd.calculator import DP
calc = DP(model='$OMM/models/dpa4/dpa-4.0.1-pro-mptrj.pt')
atoms.calc = calc
```

Run: `LD_LIBRARY_PATH="/usr/lib/wsl/lib" <prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py DPA-4.0.1-pro-MPtrj --json` (prints energy + forces; exit 0 on success).

> **Note:** Already single-head, so unlike DPA-3.1 there is no freeze step. On WSL it needs LD_LIBRARY_PATH="/usr/lib/wsl/lib" to keep the driver mount while evicting libfabric. Built against torch cu130 — below a CUDA-13-class driver, run on CPU.

---

## TACE — env `tace`

pin: python 3.11.15 · torch==2.11.0+cu130 · variants: `TACE-OAM-L`

### A. install

Upstream ([source](https://tace.readthedocs.io/en/latest/install/install.html)) — requires: Python >= 3.9, PyTorch >= 2.4 (AOTInductor export needs torch >= 2.13):

```bash
pip install tace
# pick ONE accelerator backend, never both:
#   pip install "tace[oeq]"   |   pip install "tace[cueq12]"   |   pip install "tace[cueq13]"
```

Pinned combination:

```bash
conda env create -f $OMM/envs/tace.yml -p <prefix>
#   equivalent explicit form:
#     conda create -y -p <prefix> -c conda-forge -c nvidia python=3.11.15 pip cuda-nvcc=13.0.* ase=3.29.0
#     <prefix>/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torch-geometric==2.8.0 e3nn==0.6.0 TACE @ git+https://github.com/xvzemin/tace.git@2b2214e040207b559d6bb4f92072ec1299827ba0
<prefix>/bin/pip install catbench==1.1.4    # every env: D3 dispersion + adsorption benchmarking
```

### B. weights

Upstream acquisition ([source](https://tace.readthedocs.io/en/latest/guide/foundation.html)):

```bash
# Upstream: looking the key up IS the download; it lands in ~/.cache/tace/
<prefix>/bin/python -c "from tace.foundations import tace_foundations; tace_foundations['TACE-OAM-L']"
echo "448ff14c92d0fe29b8f038e98ffad02bccfc77012d85cc29cc2e7a7a8b1e4a5a  $HOME/.cache/tace/TACE-OAM-L.pt" | sha256sum -c -
# manual download: HF xvzemin/tace-foundations
```

### C. ASE calculator

`TACE-OAM-L`

```python
from tace.foundations import tace_foundations
from tace.interface.ase import TACEAseCalc
model = tace_foundations['TACE-OAM-L']
calc = TACEAseCalc(model=model, dtype='float32', device='cuda', fidelity_idx=0, target_property=['energy','forces','stress'])
atoms.calc = calc
```

Run: `<prefix>/bin/python script.py` — never `conda activate`.

Check it works: `python3 $OMM/scripts/setup_verify.py TACE-OAM-L --json` (prints energy + forces; exit 0 on success).

> **Note:** Looking up the foundation key is what downloads the model. Enable only one accelerator backend. torch.cuda.is_available() can return true on a host below the driver floor and fail later at use time, so check the driver and pass device='cpu' from the start if it is short.

---
