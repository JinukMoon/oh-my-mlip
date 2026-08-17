"""Upstream-documented install / weight-fetch recipes, one entry per framework.

Every `pip` and `fetch` value is what that framework's OWN documentation says;
`src` / `fetch_src` are the URLs it was read from (2026-08). This is kept
deliberately separate from our pinned recipe (`envs/*.yml`): upstream tells you
the *supported* way, the pin tells you the exact combination that was
energy-validated. `scripts/gen_recipes.py` renders both side by side.

`note` records where the two diverge, and any upstream behaviour that bites in
practice. Placeholders `{sha}` / `{url}` are filled from `models.json` at
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
        note="Upstream also exposes enable_cueq / enable_flash. Our pin was validated with "
             "enable_oeq=True, which works against the cuequivariance 0.8.0 wheel already in "
             "the recipe (no separate install).",
    ),
    "MACE": dict(
        src="https://github.com/ACEsuit/mace",
        pip=["pip install --upgrade pip", "pip install mace-torch"],
        req=None,
        fetch_src="https://github.com/ACEsuit/mace-foundations/releases",
        fetch=["# mace_mp downloads by name into ~/.cache/mace (override with XDG_CACHE_HOME)",
               "<prefix>/bin/python -c \"from mace.calculators import mace_mp; mace_mp(model='medium-mpa-0', device='cpu')\"",
               "<prefix>/bin/python -c \"from mace.calculators import mace_mp; mace_mp(model='mh-1', device='cpu')\""],
        note="Upstream's example uses default_dtype='float32'; we validated float64 (the "
             "equivalence-comparison basis). MH-1 is multi-head — head='omat_pbe' or "
             "'oc20_usemppbe' (not 'oc20'); both heads share one weight file.",
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
               "#   and dies with 'Ninja is required to load C++ extensions' (observed).",
               "# MAX_JOBS=4 is required: oeq's first import JIT-builds with ninja at nproc and",
               "#   can wedge on a stale build lock."],
        note="NequIPCalculator only supports from_compiled_model, so the per-arch .pt2 is "
             "required rather than optional, and it is not portable between GPU architectures. "
             "This exact recipe was executed on an sm89 host and produced a 44 MB .pt2.",
    ),
    "Allegro": dict(
        src="https://github.com/mir-group/allegro",
        pip=["pip install nequip-allegro"],
        req=None,
        fetch_src="https://www.nequip.net/",
        fetch=["# Same compiler and URI scheme as NequIP, but NO --modifiers (cueq backend)",
               "ARCH=$(<prefix>/bin/python -c \"import torch;c=torch.cuda.get_device_capability();print(f'sm{c[0]}{c[1]}')\")",
               "PATH=\"<prefix>/bin:$PATH\" <prefix>/bin/nequip-compile \\",
               "    nequip.net:mir-group/Allegro-OAM-L:0.1 \\",
               "    $OMM/models/compiled/$ARCH/Allegro-OAM-L_${ARCH}.nequip.pt2 \\",
               "    --mode aotinductor --device cuda --target ase"],
        note="Passing enable_OpenEquivariance here fails — Allegro runs on cuequivariance. "
             "cueq ships prebuilt kernels (no JIT risk); the sm86 compile measured ~90 s.",
    ),
    "Nequix": dict(
        src="https://github.com/atomicarchitects/nequix",
        pip=["pip install nequix"],
        req="JAX backend",
        fetch_src="https://github.com/atomicarchitects/nequix",
        fetch=["# Upstream supports alias auto-download: NequixCalculator('nequix-mp-1').",
               "# We pull the commit-pinned file and check its digest instead (reproducibility).",
               "mkdir -p $OMM/models/nequix",
               "curl -L \"{url}\" -o $OMM/models/nequix/nequix-mp-1.nqx",
               "echo \"{sha}  $OMM/models/nequix/nequix-mp-1.nqx\" | sha256sum -c -",
               "# Keep the .nqx extension — renaming to .pt makes the loader treat it as torch."],
        note="Upstream's example passes use_kernel=True; our validated path is use_kernel=False.",
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
        note="The recorded digest describes the DOWNLOADED upstream checkpoint, not the derived "
             "frozen model — check the source, not the freeze output. Freeze reproducibility is "
             "claimed same-host only. At run time DeePMD needs LD_LIBRARY_PATH=\"\" (Intel oneAPI "
             "libfabric clash).",
    ),
    "ORB": dict(
        src="https://github.com/orbital-materials/orb-models",
        pip=["pip install orb-models"],
        req=None,
        fetch_src="https://github.com/orbital-materials/orb-models/blob/main/MODELS.md",
        fetch=["# Calling the named pretrained function is what triggers the download",
               "<prefix>/bin/python -c \"from orb_models.forcefield import pretrained; \\",
               "    pretrained.orb_v3_conservative_inf_omat(device='cpu', precision='float32-high')\""],
        note="Upstream's README example unpacks a (orbff, atoms_adapter) 2-tuple; our pinned "
             "version returns a single object — use the block-C line, not the README's. "
             "precision='float32-high' selects TF32 matmul, so repeat single-points differ by "
             "~3.1e-5 eV/atom: the weights are digest-pinned but the energy is only "
             "equivalence-stable, never bitwise.",
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
        note="Upstream also offers the grace_fm('<name>') convenience loader "
             "(tensorpotential.calculator.foundation_models); we validated the flattened path "
             "handed straight to TPCalculator. The recorded digest is a TREE digest over the "
             "SavedModel directory, not a single file. On GPU, TF 2.16 wants libcudnn.so.8 while "
             "catbench's torch pulls cuDNN 9 — sideload cuDNN 8.9 or GPU registration is skipped "
             "silently.",
    ),
    "MatterSim": dict(
        src="https://github.com/microsoft/mattersim",
        pip=["pip install mattersim"],
        req=None,
        fetch_src="https://github.com/microsoft/mattersim",
        fetch=["# Checkpoints ship in the package (./pretrained_models/); 1M is the default,",
               "# the 5M model is selected with load_path."],
        note="Upstream defaults to the 1M model. We use load_path='MatterSim-v1.0.0-5M.pth' and "
             "pass device='cuda' explicitly — without it the calculator has fallen back to CPU.",
    ),
    "CHGNet": dict(
        src="https://github.com/CederGroupHub/chgnet",
        pip=["pip install chgnet"],
        req=None,
        fetch_src="https://github.com/CederGroupHub/chgnet",
        fetch=["# CHGNet.load() prepares the weights itself — no download command needed."],
        note="We pin the version explicitly: CHGNet.load(model_name='0.3.0').",
    ),
    "AlphaNet": dict(
        src="https://github.com/zmyybc/AlphaNet",
        pip=["git clone https://github.com/zmyybc/AlphaNet.git", "cd AlphaNet", "pip install -e ."],
        req="Upstream README shows python 3.8; the validated combination is python 3.11 + torch 2.1.2+cu121",
        fetch_src="https://github.com/zmyybc/AlphaNet",
        fetch=["# Both the figshare checkpoint AND the in-repo config are required",
               "mkdir -p $OMM/models/alphanet",
               "curl -L \"{url}\" -o $OMM/models/alphanet/alex_0410.ckpt",
               "echo \"{sha}  $OMM/models/alphanet/alex_0410.ckpt\" | sha256sum -c -",
               "curl -L \"https://raw.githubusercontent.com/zmyybc/AlphaNet/65f8ea9330459e0106867d1c694aec4139c6cb19/pretrained/OMA/oma.json\" \\",
               "     -o $OMM/models/alphanet/oma.json"],
        note="Installing upstream HEAD drifts gas-molecule energies by up to 0.24 eV/atom, so the "
             "recipe pins commit 65f8ea93. Do not install matscipy: it forces numpy>=2 and breaks "
             "the torch 2.1.2 ABI.",
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
        note="Upstream documents neither the cache path nor the URL — both were read out of the "
             "package's own url_dict.",
    ),
    "fairchemv1": dict(
        src="https://github.com/facebookresearch/fairchem",
        pip=["pip install fairchem-core"],
        req=None,
        fetch_src="https://huggingface.co/facebook/OMAT24",
        fetch=["# GATED: accept the facebook/OMAT24 license with your own HF account first",
               "huggingface-cli login",
               "<prefix>/bin/python -c \"from huggingface_hub import hf_hub_download; \\",
               "    print(hf_hub_download('facebook/OMAT24','esen_30m_oam.pt', local_dir='$OMM/models/fairchem'))\"",
               "echo \"{sha}  $OMM/models/fairchem/esen_30m_oam.pt\" | sha256sum -c -"],
        note="OCPCalculator defaults to cpu=True — pass cpu=False or it silently runs on CPU.",
    ),
    "EquiformerV3": dict(
        src="https://github.com/atomicarchitects/equiformer_v3",
        pip=["# Upstream publishes no pip package — it vendors a fairchem fork (sha a7300c58),",
             "# which envs/equiformer_v3.yml installs editable. There is no upstream install",
             "# procedure to quote; the recipe below is the only reproducible path."],
        req="scipy==1.16.0 (1.17 removed sph_harm)",
        fetch_src="https://huggingface.co/mirror-physics/equiformer_v3",
        fetch=["mkdir -p $OMM/models/equiformer_v3/checkpoint",
               "curl -L \"https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24-mptrj-salex_gradient.pt\" \\",
               "     -o $OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt",
               "echo \"{sha}  $OMM/models/equiformer_v3/checkpoint/omat24-mptrj-salex_gradient.pt\" | sha256sum -c -",
               "# The checkpoint/ path prefix is required — the bare filename 404s."],
        note="Research-code release: the README documents training, not installation.",
    ),
    "UMA": dict(
        src="https://github.com/facebookresearch/fairchem",
        pip=["pip install fairchem-core"],
        req=None,
        fetch_src="https://huggingface.co/facebook/UMA",
        fetch=["# GATED: your HF account must be approved for facebook/UMA (upstream documents this)",
               "huggingface-cli login",
               "<prefix>/bin/python -c \"from fairchem.core import pretrained_mlip; \\",
               "    pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu')\"",
               "#   same pattern for uma-s-1p1 / uma-m-1p1 — three checkpoints cover seven variants"],
        note="task_name is what separates the variants: oc20 (catalysis) / oc22 (oxides) / omat "
             "(inorganic) / oc25 / omol / odac / omc. uma-m-1p1 (11.2 GB) needs materially more "
             "than 19 GB of host RAM: on a 19 GB host the loader is OOM-killed (exit 137) and "
             "surfaces only as 'worker produced no handshake'.",
    ),
    "PET": dict(
        src="https://github.com/metatensor/metatrain",
        pip=["pip install metatrain"],
        req=None,
        fetch_src="http://docs.metatensor.org/metatrain/latest/dev-docs/cli/export.html",
        fetch=["# Only a .ckpt is published; inference needs an exported metatomic .pt",
               "mkdir -p $OMM/models/pet",
               "curl -L \"https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt\" \\",
               "     -o $OMM/models/pet/pet-oam-xl-v1.0.0.ckpt          # 2.92 GB",
               "# upstream syntax: mtt export <path> <output>   (path may be local, an HF repo id, or a URL)",
               "<prefix>/bin/mtt export $OMM/models/pet/pet-oam-xl-v1.0.0.ckpt \\",
               "     $OMM/models/pet/pet-oam-xl-v1.0.0.pt"],
        note="The export output is not byte-reproducible (serialization differs per run), so this "
             "model's determinism oracle is the energy, not a sha256.",
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
        note="Upstream's two-file requirements split is exactly why a single conda solve is "
             "impossible; our sidecar does the same work as a 2-pass pip. V2 is cueq-only — do "
             "not inject conv_type='flashtp' (that is a v1-only backend).",
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
        note="Built against torch cu130, so GPU use needs a CUDA-13-class driver (>= ~580). "
             "Below that, run with device='cpu' — the energy is unchanged, only slower.",
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
        note="Unlike DPA-3.1 this checkpoint is already single-head, so no freeze step. At run "
             "time it needs LD_LIBRARY_PATH=\"/usr/lib/wsl/lib\" on WSL (keeps the driver mount "
             "while evicting libfabric).",
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
        note="On a host below the driver floor torch.cuda.is_available() has returned a false "
             "positive and the failure surfaced only at use time — check the driver first and "
             "pass device='cpu' from the start if it is short.",
    ),
}
