"""Upstream-documented install / weight-fetch recipes, one entry per framework.

Every `pip` and `fetch` value is what that framework's OWN documentation says;
`src` / `fetch_src` are the URLs it was read from (2026-08). This is kept
deliberately separate from the pinned recipe (`envs/*.yml`): upstream tells you
the *supported* way, the pin gives one combination known to work together.
`scripts/gen_recipes.py` renders both side by side.

`note` records where the two diverge, and library behaviour that affects how you
call it. Placeholders `{sha}` / `{url}` are filled from `models.json` at
generation time and are what the drift guards check.

When upstream changes its documented procedure, edit THIS file — never the
generated `docs/recipes.md`.
"""

UPSTREAM: dict[str, dict] = {
    "SevenNet": dict(
        src="https://sevennet.readthedocs.io/en/latest/",
        pip=["pip install sevenn"],
        req="Python >= 3.10, PyTorch >= 2.0.0",
        fetch_src="https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html",
        fetch=["# Checkpoints ship inside the sevenn package — pass the model keyword only:",
               "#   '7net-mf-ompa' / 'SevenNet-mf-ompa',  '7net-omni' / 'SevenNet-Omni'",
               "# modal='mpa' is the documented PBE-level default for both."],
        note="Upstream also exposes enable_cueq / enable_flash. The pinned recipe uses "
             "enable_oeq=True, which works against the cuequivariance wheel it already "
             "installs — no separate accelerator install is needed.",
    ),
    "MACE": dict(
        src="https://github.com/ACEsuit/mace",
        pip=["pip install --upgrade pip", "pip install mace-torch"],
        req=None,
        fetch_src="https://github.com/ACEsuit/mace-foundations/releases",
        fetch=["# mace_mp downloads by name into ~/.cache/mace (override with XDG_CACHE_HOME)",
               "<prefix>/bin/python -c \"from mace.calculators import mace_mp; mace_mp(model='medium-mpa-0', device='cpu')\"",
               "<prefix>/bin/python -c \"from mace.calculators import mace_mp; mace_mp(model='mh-1', device='cpu')\""],
        note="MH-1 is multi-head: pass head='omat_pbe' or head='oc20_usemppbe' (there is no "
             "plain 'oc20' head). Both heads load the same weight file. default_dtype "
             "selects float32 or float64 — the calculator lines above use float64.",
    ),
    "NequIP": dict(
        src="https://nequip.readthedocs.io/en/latest/guide/getting-started/install.html",
        pip=["pip install nequip", "pip install openequivariance   # accelerator, separate"],
        req=None,
        fetch_src="https://www.nequip.net/",
        fetch=["# Download and compile are ONE command: nequip-compile takes a nequip.net URI",
               "#   input_path = 'nequip.net:<group>/<model>:<version>'   (nequip-compile --help)",
               "ARCH=$(<prefix>/bin/python -c \"import torch;c=torch.cuda.get_device_capability();print(f'sm{c[0]}{c[1]}')\")",
               "PATH=\"<prefix>/bin:$PATH\" MAX_JOBS=4 <prefix>/bin/nequip-compile \\",
               "    nequip.net:mir-group/NequIP-OAM-XL:0.1 \\",
               "    $OMM/models/compiled/$ARCH/NequIP-OAM-XL_${ARCH}.nequip.pt2 \\",
               "    --mode aotinductor --device cuda --target ase \\",
               "    --modifiers enable_OpenEquivariance",
               "# same for OAM-L (nequip.net:mir-group/NequIP-OAM-L:0.1)",
               "# PATH must include <prefix>/bin: without it openequivariance cannot find ninja",
               "#   and dies with 'Ninja is required to load C++ extensions'.",
               "# MAX_JOBS=4 is required: oeq's first import JIT-builds with ninja at nproc and",
               "#   can wedge on a stale build lock."],
        note="NequIPCalculator only loads via from_compiled_model, so the per-arch .pt2 is "
             "required rather than optional and must be rebuilt for each GPU architecture.",
    ),
    "Allegro": dict(
        src="https://github.com/mir-group/allegro",
        pip=["pip install nequip-allegro"],
        req=None,
        fetch_src="https://www.nequip.net/",
        fetch=["# Same compiler and URI scheme as NequIP, with the CuEquivariance modifier",
               "ARCH=$(<prefix>/bin/python -c \"import torch;c=torch.cuda.get_device_capability();print(f'sm{c[0]}{c[1]}')\")",
               "PATH=\"<prefix>/bin:$PATH\" <prefix>/bin/nequip-compile \\",
               "    nequip.net:mir-group/Allegro-OAM-L:0.1 \\",
               "    $OMM/models/compiled/$ARCH/Allegro-OAM-L_${ARCH}.nequip.pt2 \\",
               "    --mode aotinductor --device cuda --target ase \\",
               "    --modifiers enable_CuEquivarianceContracter",
               "# Load: `import cuequivariance_torch` BEFORE NequIPCalculator.from_compiled_model,",
               "#   else 'Could not find schema for cuequivariance_ops::tensor_product_uniform_1d_jit'.",
               "# AOT Inductor + CuEquivariance does not support float64 models (use torchscript)."],
        note="Allegro's recommended accelerated path is CuEquivariance: compile with "
             "--modifiers enable_CuEquivarianceContracter (needs cuequivariance-torch + "
             "cuequivariance-ops-torch-cu12) and import cuequivariance_torch before loading.",
    ),
    "Nequix": dict(
        src="https://github.com/atomicarchitects/nequix",
        pip=["pip install nequix"],
        req="JAX backend",
        fetch_src="https://github.com/atomicarchitects/nequix",
        fetch=["# Upstream supports alias auto-download: NequixCalculator('nequix-mp-1').",
               "# The commit-pinned file below is fetched directly so the digest can be checked.",
               "mkdir -p $OMM/models/nequix",
               "curl -L \"{url}\" -o $OMM/models/nequix/nequix-mp-1.nqx",
               "echo \"{sha}  $OMM/models/nequix/nequix-mp-1.nqx\" | sha256sum -c -",
               "# Keep the .nqx extension — renaming to .pt makes the loader treat it as torch."],
        note="use_kernel toggles the fused kernel; the calculator line above leaves it off.",
    ),
    "DeePMD": dict(
        src="https://docs.deepmodeling.com/projects/deepmd/",
        pip=["pip install torch torchvision torchaudio",
             "pip install git+https://github.com/deepmodeling/deepmd-kit@v3.1.0"],
        req="PyTorch backend",
        fetch_src="https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M",
        fetch=["# A multi-task training checkpoint: it must be frozen to one head before inference",
               "mkdir -p $OMM/models/deepmd",
               "curl -L \"https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt\" \\",
               "     -o $OMM/models/deepmd/DPA-3.1-3M.pt",
               "<prefix>/bin/dp --pt freeze -c $OMM/models/deepmd/DPA-3.1-3M.pt \\",
               "     -o $OMM/models/deepmd/frozen-omat24.pth --head Omat24",
               "# upstream equivalent: dp --pt freeze -c <ckpt> -o frozen_model.pth --model-branch <head>"],
        note="The published checkpoint is multi-task, so it must be frozen to a single head "
             "before inference. The recorded digest is for the downloaded checkpoint, not "
             "the frozen output. At run time DeePMD needs LD_LIBRARY_PATH=\"\" to avoid an "
             "Intel oneAPI libfabric clash.",
    ),
    "ORB": dict(
        src="https://github.com/orbital-materials/orb-models",
        pip=["pip install orb-models"],
        req=None,
        fetch_src="https://github.com/orbital-materials/orb-models/blob/main/MODELS.md",
        fetch=["# Calling the named pretrained function is what triggers the download",
               "<prefix>/bin/python -c \"from orb_models.forcefield import pretrained; \\",
               "    pretrained.orb_v3_conservative_inf_omat(device='cpu', precision='float32-high')\""],
        note="Upstream's README unpacks a (orbff, atoms_adapter) tuple; the pinned version "
             "returns a single object, so use the calculator line above. "
             "precision='float32-high' selects TF32 matmul, which makes repeated runs "
             "numerically close rather than bit-identical.",
    ),
    "GRACE": dict(
        src="https://gracemaker.readthedocs.io/",
        pip=["pip install tensorpotential"],
        req="TensorFlow backend (no torch)",
        fetch_src="https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/",
        fetch=["# Upstream CLI (documented): list / download / checkpoint",
               "<prefix>/bin/grace_models list",
               "<prefix>/bin/grace_models download GRACE-2L-OAM",
               "#   default cache $HOME/.cache/grace  (override with GRACE_CACHE)",
               "# The SavedModel lands nested (<t>/<n>/<n>/saved_model.pb) and must be flattened",
               "# so saved_model.pb sits directly under the target directory:",
               "python3 $OMM/scripts/prepare_grace_weights.py --name GRACE-2L-OAM \\",
               "    --target-dir $OMM/models/grace/GRACE-2L-OAM"],
        note="TensorFlow backend, so D3 dispersion (which needs torch) is unavailable. "
             "grace_models leaves the SavedModel nested and it must be flattened before "
             "TPCalculator can load it; the digest covers the whole SavedModel tree, not a "
             "single file. On GPU, TF 2.16 loads libcudnn.so.8 while catbench's torch "
             "installs cuDNN 9 — sideload cuDNN 8.9 or TF skips GPU registration silently. "
             "Upstream also offers the grace_fm('<name>') convenience loader.",
    ),
    "MatterSim": dict(
        src="https://github.com/microsoft/mattersim",
        pip=["pip install mattersim"],
        req=None,
        fetch_src="https://github.com/microsoft/mattersim",
        fetch=["# Checkpoints ship in the package (./pretrained_models/); 1M is the default,",
               "# the 5M model is selected with load_path."],
        note="The package ships both the 1M and 5M checkpoints and defaults to 1M; load_path "
             "selects the 5M one. Pass device explicitly — otherwise the calculator can fall "
             "back to CPU.",
    ),
    "CHGNet": dict(
        src="https://github.com/CederGroupHub/chgnet",
        pip=["pip install chgnet"],
        req=None,
        fetch_src="https://github.com/CederGroupHub/chgnet",
        fetch=["# CHGNet.load() prepares the weights itself — no download command needed."],
        note="CHGNet.load() takes a model_name, so the version is selected in code rather "
             "than at download time.",
    ),
    "AlphaNet": dict(
        src="https://github.com/zmyybc/AlphaNet",
        pip=["git clone https://github.com/zmyybc/AlphaNet.git", "cd AlphaNet", "pip install -e ."],
        req="Upstream README shows python 3.8; the pinned recipe uses python 3.11 + torch 2.1.2+cu121",
        fetch_src="https://github.com/zmyybc/AlphaNet",
        fetch=["# Both the figshare checkpoint AND the in-repo config are required",
               "mkdir -p $OMM/models/alphanet",
               "curl -L \"{url}\" -o $OMM/models/alphanet/alex_0410.ckpt",
               "echo \"{sha}  $OMM/models/alphanet/alex_0410.ckpt\" | sha256sum -c -",
               "curl -L \"https://raw.githubusercontent.com/zmyybc/AlphaNet/65f8ea9330459e0106867d1c694aec4139c6cb19/pretrained/OMA/oma.json\" \\",
               "     -o $OMM/models/alphanet/oma.json"],
        note="Needs both the checkpoint and its config JSON — loading the checkpoint alone "
             "fails. The recipe installs a pinned commit because package HEAD changes model "
             "behaviour. Do not install matscipy: it forces numpy>=2 and breaks the torch "
             "2.1.2 ABI.",
    ),
    "Eqnorm": dict(
        src="https://github.com/yzchen08/eqnorm",
        pip=["pip install git+https://github.com/yzchen08/eqnorm.git"],
        req=None,
        fetch_src="https://github.com/yzchen08/eqnorm",
        fetch=["# Upstream: EqnormCalculator(model_variant='eqnorm-mptrj') downloads on its own.",
               "# But that downloader hits plain figshare.com, which 202-blocks on some networks",
               "# and leaves a 0-byte file the package then treats as present — never retrying.",
               "# Pre-stage from the working subdomain instead:",
               "mkdir -p ~/.cache/eqnorm",
               "curl -L \"https://ndownloader.figshare.com/files/55429685\" -o ~/.cache/eqnorm/eqnorm-mptrj.pt",
               "echo \"{sha}  $HOME/.cache/eqnorm/eqnorm-mptrj.pt\" | sha256sum -c -"],
        note="Upstream documents neither the cache path nor the download URL — both come "
             "from the package's own url_dict.",
    ),
    "fairchemv1": dict(
        src="https://github.com/facebookresearch/fairchem",
        pip=["pip install fairchem-core"],
        req=None,
        fetch_src="https://huggingface.co/facebook/OMAT24",
        fetch=["# GATED: accept the facebook/OMAT24 license with your own HF account first",
               "hf auth login",
               "<prefix>/bin/python -c \"from huggingface_hub import hf_hub_download; \\",
               "    print(hf_hub_download('facebook/OMAT24','esen_30m_oam.pt', local_dir='$OMM/models/fairchem'))\"",
               "echo \"{sha}  $OMM/models/fairchem/esen_30m_oam.pt\" | sha256sum -c -"],
        note="Gated weights: the download fails without an accepted license and a token. "
             "OCPCalculator defaults to cpu=True, so pass cpu=False to run on GPU.",
    ),
    "EquiformerV3": dict(
        src="https://github.com/atomicarchitects/equiformer_v3",
        pip=["# Upstream publishes no pip package — it vendors a fairchem fork (sha a7300c58),",
             "# which envs/equiformer_v3.yml installs editable. There is no upstream install",
             "# procedure to quote — use the pinned recipe below."],
        req="scipy==1.16.0 (1.17 removed sph_harm)",
        fetch_src="https://huggingface.co/mirror-physics/equiformer_v3",
        fetch=["mkdir -p $OMM/models/equiformer_v3/checkpoint",
               "curl -L \"https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24-mptrj-salex_gradient.pt\" \\",
               "     -o $OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt",
               "echo \"{sha}  $OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt\" | sha256sum -c -",
               "# The checkpoint/ path prefix is required — the bare filename 404s."],
        note="Upstream publishes no pip package — it vendors a fairchem fork, which the "
             "recipe installs editable. The checkpoint URL needs its checkpoint/ path "
             "prefix; the bare filename 404s.",
    ),
    "UMA": dict(
        src="https://github.com/facebookresearch/fairchem",
        pip=["pip install fairchem-core"],
        req=None,
        fetch_src="https://huggingface.co/facebook/UMA",
        fetch=["# GATED: your HF account must be approved for facebook/UMA (upstream documents this)",
               "hf auth login",
               "<prefix>/bin/python -c \"from fairchem.core import pretrained_mlip; \\",
               "    pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu')\"",
               "#   same pattern for uma-s-1p1 / uma-m-1p1 — three checkpoints cover the eight variants"],
        note="Gated weights: the download fails without an accepted license and a token. "
             "The checkpoint is uma-s-1p2, uma-s-1p1 or uma-m-1p1; task_name selects the "
             "variant — oc20 (catalysis), oc22 (oxides), oc25 (catalyst-electrolyte "
             "interfaces), omat (inorganic materials). The -m- checkpoint is 11.2 GB and is "
             "loaded into host RAM before it reaches the GPU, so it needs 32 GB or more of "
             "system memory; the -s- checkpoints do not.",
    ),
    "PET": dict(
        src="https://github.com/metatensor/metatrain",
        pip=["pip install metatrain"],
        req=None,
        fetch_src="https://docs.metatensor.org/metatrain/latest/dev-docs/cli/export.html",
        fetch=["# Only a .ckpt is published; inference needs an exported metatomic .pt",
               "mkdir -p $OMM/models/pet",
               "curl -L \"https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt\" \\",
               "     -o $OMM/models/pet/pet-oam-xl-v1.0.0.ckpt          # 2.92 GB",
               "# upstream syntax: mtt export <path> <output>   (path may be local, an HF repo id, or a URL)",
               "<prefix>/bin/mtt export $OMM/models/pet/pet-oam-xl-v1.0.0.ckpt \\",
               "     $OMM/models/pet/pet-oam-xl-v1.0.0.pt"],
        note="The export output is not byte-reproducible: its serialization differs between "
             "runs even for the same input.",
    ),
    "EquFlash": dict(
        src="https://github.com/SamsungDS/GGNN",
        pip=["pip install -r requirements.txt",
             "pip install --no-deps -r requirements-no-deps.txt"],
        req="Python 3.12, PyTorch 2.9.1 (CUDA 12.6) — upstream's stated requirement",
        fetch_src="https://github.com/SamsungDS/GGNN",
        fetch=["mkdir -p $OMM/models/equflash",
               "curl -L \"https://ndownloader.figshare.com/files/65435007\" -o $OMM/models/equflash/EquFlashV2.pt",
               "echo \"068ce25767d5064fae41ffac352484ae59a8544ee671607a09dffdc6d62402a2  $OMM/models/equflash/EquFlashV2.pt\" | sha256sum -c -",
               "curl -L \"https://ndownloader.figshare.com/files/65435004\" -o $OMM/models/equflash/EquFlash.pt",
               "echo \"4a1eb19e4f719282146285dd1655178a0d1cc70edee7f8e1401e9b1336383bfa  $OMM/models/equflash/EquFlash.pt\" | sha256sum -c -",
               "# plain figshare.com 202-blocks — always use the ndownloader subdomain"],
        note="Upstream splits its requirements across two files, which is why a single conda "
             "solve cannot build this env and the recipe uses a two-pass sidecar. V2 is "
             "cueq-only — conv_type='flashtp' is a v1-only backend and V2 rejects it.",
    ),
    "MatRIS": dict(
        src="https://github.com/HPC-AI-Team/MatRIS",
        pip=["# Upstream states requirements but no install command:",
             "#   ase>=3.23.0, numpy>=2.0.0, pymatgen>2024.9.10, torch>2.6.0"],
        req="torch > 2.6.0",
        fetch_src="https://github.com/HPC-AI-Team/MatRIS",
        fetch=["# Upstream selects by model key (matris_10m_oam) and downloads internally.",
               "# That downloader has the same 0-byte trap as eqnorm, so pre-stage it:",
               "mkdir -p ~/.cache/matris",
               "curl -L \"https://ndownloader.figshare.com/files/59142728\" -o ~/.cache/matris/MatRIS_10M_OAM.pth.tar",
               "echo \"{sha}  $HOME/.cache/matris/MatRIS_10M_OAM.pth.tar\" | sha256sum -c -"],
        note="Selected by model key. Built against torch cu130, so GPU use needs a "
             "CUDA-13-class driver — below that, run with device='cpu'.",
    ),
    "DPA4": dict(
        src="https://docs.deepmodeling.com/projects/deepmd/",
        pip=["pip install torch torchvision torchaudio",
             "pip install git+https://github.com/deepmodeling/deepmd-kit@v3.1.0"],
        req="PyTorch backend",
        fetch_src="https://matbench-discovery.materialsproject.org/models/dpa-4.0.1-pro-mptrj",
        fetch=["mkdir -p $OMM/models/dpa4",
               "curl -L \"https://ndownloader.figshare.com/files/65469204\" -o $OMM/models/dpa4/dpa-4.0.1-pro-mptrj.pt",
               "echo \"{sha}  $OMM/models/dpa4/dpa-4.0.1-pro-mptrj.pt\" | sha256sum -c -",
               "# (the registry's weights_source is the matbench-discovery model page; the",
               "#  direct artifact is the figshare URL above)"],
        note="Already single-head, so unlike DPA-3.1 there is no freeze step. On WSL it "
             "needs LD_LIBRARY_PATH=\"/usr/lib/wsl/lib\" to keep the driver mount while "
             "evicting libfabric. Built against torch cu130 — below a CUDA-13-class driver, "
             "run on CPU.",
    ),
    "TACE": dict(
        src="https://tace.readthedocs.io/en/latest/install/install.html",
        pip=["pip install tace",
             "# pick ONE accelerator backend, never both:",
             "#   pip install \"tace[oeq]\"   |   pip install \"tace[cueq12]\"   |   pip install \"tace[cueq13]\""],
        req="Python >= 3.9, PyTorch >= 2.4 (AOTInductor export needs torch >= 2.13)",
        fetch_src="https://tace.readthedocs.io/en/latest/guide/foundation.html",
        fetch=["# Upstream: looking the key up IS the download; it lands in ~/.cache/tace/",
               "<prefix>/bin/python -c \"from tace.foundations import tace_foundations; tace_foundations['TACE-OAM-L']\"",
               "echo \"{sha}  $HOME/.cache/tace/TACE-OAM-L.pt\" | sha256sum -c -",
               "# manual download: HF xvzemin/tace-foundations"],
        note="Looking up the foundation key is what downloads the model. Enable only one "
             "accelerator backend. torch.cuda.is_available() can return true on a host below "
             "the driver floor and fail later at use time, so check the driver and pass "
             "device='cpu' from the start if it is short.",
    ),
}
