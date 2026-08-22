# Fine-tuning recipes — continuing training from a foundation checkpoint

<!-- GENERATED FILE — do not edit by hand. -->
<!-- Regenerate: python3 scripts/gen_finetune.py --write   ·   CI: --check -->

One section per framework: **A** is what upstream's own fine-tuning docs say
(verbatim command, source linked); **B** is the dataset format the trainer
actually reads; **C** is the single command this hub generates per variant —
`scripts/ft_run.py` resolves the variant's `finetune` block in `models.json`,
converts the dataset, writes the patched config/command, and (when runnable)
executes it; **D** is the per-variant support matrix. `status` is a
doc-honesty state, not an execution claim — see `demonstrated` for that.

```bash
export OMM=$(pwd)          # this clone
source $OMM/env.sh         # shared caches + D3/CUDA environment, once per shell
```

A variant refused with `exit 2` is `not-supported` or `code-excavation-needed`
— no amount of retrying fixes it, see the reason and evidence in block D/C.
A variant refused with `exit 3` is `documented` but `runnable_as_installed:
false` — the listed blockers are fixable (usually one `pip install`, or a git
clone); fix them and rerun the same command.

---

## SevenNet — env `sevennet`

### A. What upstream documents

Upstream ([source](https://sevennet.readthedocs.io/en/latest/user_guide/cli.html)):

```bash
sevenn_preset fine_tune > input.yaml     # or: sevenn preset fine_tune > input.yaml
#  ... edit input.yaml: train.continue.checkpoint + data.load_trainset_path ...
sevenn train input.yaml -s
# with an accelerator: sevenn train input.yaml -s --enable_oeq   (also --enable_cueq, --enable_flash)
```

> **Note:** Installed sevenn 0.12.2.dev0 ships two presets beyond the CLI doc's list: fine_tune_le (liquid-electrolyte recipe: lr dropped 40x, shift/scale made trainable) and mf_ompa_fine_tune (path checkpoint for SevenNet-MF-ompa). The README's advertised 'fine-tuning with forgetting prevention' (replay + EWC) has no corresponding flag or YAML key in this version — an open gap, not claimed here.

### B. Dataset format

Anything ase.io.read handles (extxyz, OUTCAR, ...), the legacy structure_list format, or a pre-built graph sevenn_data/*.pt. Energy/force/stress are read from the ASE calculator by default (atoms.get_potential_energy()/get_forces()), NOT from info keys; sevenn graph_build --kwargs / data.data_format_args rename the keys when needed. A custom stress key is negated on ingest (dataload.py:343) while the calculator path is not.

### C. The command this hub generates

`SevenNet-MF-OMPA`:

```bash
python3 $OMM/scripts/ft_run.py SevenNet-MF-OMPA --dataset <your-dataset> --out ft_sevennet-mf-ompa
```

`SevenNet-Omni`:

```bash
python3 $OMM/scripts/ft_run.py SevenNet-Omni --dataset <your-dataset> --out ft_sevennet-omni
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `SevenNet-MF-OMPA` | documented | True | — | ft_verify: energy=-16.3878 eV on a 2-epoch demo fine-tune, 2026-08-22 |
| `SevenNet-Omni` | documented | True | — | — |

---

## MACE — env `mace`

### A. What upstream documents

Upstream ([source](https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html)):

```bash
mace_run_train \
    --name="MACE" \
    --foundation_model="medium-omat-0" \
    --multiheads_finetuning=False \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --energy_key=REF_energy --forces_key=REF_forces \
    --E0s="average" --lr=0.01 --batch_size=2 --max_num_epochs=100 \
    --ema --ema_decay=0.99 --default_dtype="float64" --device=cuda
```

> **Note:** --foundation_head selects which head of a multi-head foundation model to fine-tune (MH-1: 'omat_pbe' or 'oc20_usemppbe' — there is no plain 'oc20' head); this flag exists in the installed 0.3.15 CLI but appears in no official multihead-finetuning example. --pt_train_file=mp auto-downloads a Materials Project replay set; --pseudolabel_replay lets the foundation model label the replay data itself instead of shipping labels. LoRA fine-tuning (--lora_alpha, rank default 4) exists in the CLI with no documented example.

### B. Dataset format

extxyz via --train_file/--valid_file/--test_file. Default keys are NOT the plain ASE names: energy REF_energy, forces REF_forces, virials REF_virials, stress REF_stress (--energy_key/--forces_key/... override). Data written via ASE's SinglePointCalculator needs --energy_key=energy --forces_key=forces, or the info/arrays keys must be renamed to REF_*.

### C. The command this hub generates

`MACE-MPA-0`:

```bash
python3 $OMM/scripts/ft_run.py MACE-MPA-0 --dataset <your-dataset> --out ft_mace-mpa-0
```

`MACE-MH-1-OMAT`:

```bash
# LICENCE: ASL -- https://github.com/ACEsuit/mace-foundations
#   (Part 8 decision (a)/option B: disclosed, not refused --
#    oh-my-mlip is MIT and redistributes no weights either way)
python3 $OMM/scripts/ft_run.py MACE-MH-1-OMAT --dataset <your-dataset> --out ft_mace-mh-1-omat
```

`MACE-MH-1-OC20`:

```bash
# LICENCE: ASL -- https://github.com/ACEsuit/mace-foundations
#   (Part 8 decision (a)/option B: disclosed, not refused --
#    oh-my-mlip is MIT and redistributes no weights either way)
python3 $OMM/scripts/ft_run.py MACE-MH-1-OC20 --dataset <your-dataset> --out ft_mace-mh-1-oc20
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `MACE-MPA-0` | documented | True | — | ft_verify: energy=-16.4872 eV on a 2-epoch demo fine-tune, 2026-08-22 |
| `MACE-MH-1-OMAT` | documented | True | ASL | — |
| `MACE-MH-1-OC20` | documented | True | ASL | — |

---

## NequIP — env `nequip`

### A. What upstream documents

Upstream ([source](https://nequip.readthedocs.io/en/latest/guide/training-techniques/fine_tuning.html)):

```bash
nequip-train -cp full/path/to/config/directory -cn config_name.yaml
```

> **Note:** cutoff_radius and model_type_names MUST match the foundation model exactly (a different r_max breaks the neighbour list) — current docs read them off the package via ${type_names_from_package:...}/${cutoff_radius_from_package:...} resolvers. DOC-VS-INSTALLED: those resolvers exist in the installed NequIP env (nequip 0.17.1) but NOT in the installed Allegro env (nequip 0.15.0) — see the Allegro entry. Re-referencing energies to a new DFT setting uses nequip.model.modify + modify_PerTypeScaleShift (per-element shifts); only SUBSETS of the foundation model's atom types are supported, new types cannot be added. Post-FT packaging: nequip-package build <ckpt> <out>.nequip.zip, nequip-compile ... --mode aotinductor.

### B. Dataset format

extxyz (or any ASE-readable format) via nequip.data.datamodule.ASEDataModule (train_file_path/val_file_path/test_file_path). Precedence when a quantity appears in more than one place: atoms.arrays < atoms.info < atoms.calc.results — a SinglePointCalculator-carrying extxyz needs no key_mapping at all.

### C. The command this hub generates

`NequIP-OAM-XL`:

```bash
python3 $OMM/scripts/ft_run.py NequIP-OAM-XL --dataset <your-dataset> --out ft_nequip-oam-xl
```

`NequIP-OAM-L`:

```bash
python3 $OMM/scripts/ft_run.py NequIP-OAM-L --dataset <your-dataset> --out ft_nequip-oam-l
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `NequIP-OAM-XL` | documented | True | — | — |
| `NequIP-OAM-L` | documented | True | — | — |

---

## Allegro — env `allegro`

### A. What upstream documents

Upstream ([source](https://github.com/mir-group/allegro)):

```bash
nequip-train -cp full/path/to/config/directory -cn config_name.yaml
```

> **Note:** Allegro is a model plugin inside the NequIP trainer, not a separate CLI — the Allegro env's bin/ contains only the nequip-* entry points, no allegro-train. The official tutorial config (configs/tutorial.yaml) nests the model block under training_module:, while the NequIP fine-tuning doc shows a top-level model: — both forms appear in official material, and the installed 0.15.0 tutorial uses the training_module: nesting; which shape oh-my-mlip should emit is unresolved without a smoke run.

### B. Dataset format

Same as NequIP — ASEDataModule over extxyz, keys taken from the ASE calculator with key_mapping available for overrides.

### C. The command this hub generates

`Allegro-OAM-L`:

```bash
python3 $OMM/scripts/ft_run.py Allegro-OAM-L --dataset <your-dataset> --out ft_allegro-oam-l
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `Allegro-OAM-L` | documented (inherited from NequIP) | True | — | — |

---

## Nequix — env `nequix`

### A. What upstream documents

Upstream ([source](https://github.com/atomicarchitects/nequix)):

```bash
nequix_train <config>.yml       # JAX backend, default
# torch backend (single GPU): uv run nequix/torch_impl/train.py <config>.yml
# torch backend (multi-GPU):  uv run torchrun --nproc_per_node=<gpus> nequix/torch_impl/train.py <config>.yml
# Phonon fine-tuning (JAX-only): uv run nequix/pft/train.py configs/nequix-oam-1-pft.yml
```

> **Note:** JAX finetune_from RAISES NotImplementedError if atom_energies is also present in the config — a JAX fine-tune must not redefine isolated-atom energies (the torch backend can). batch_size in the config is per-device. nequix_train is the only installed console script; the torch/PFT trainers must be invoked by file path or via torchrun.

### B. Dataset format

AseDBDataset reads train_path/valid_path as a directory globbed for *.aselmdb, or a single ASE-db file. scripts/preprocess_data.py (repo-only, not in the installed wheel) converts any ASE-readable file: ase.io.read(file, index=':') -> ase.db.connect(...).write(atoms, data=atoms.info). Self-contained local equivalent: a ~10-line loop writing an .aselmdb with the installed ase_db_backends (verified working in-env).

### C. The command this hub generates

`Nequix-MP-1`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/, data/ and scripts/preprocess_data.py are repo-only -- not in the installed nequix 0.4.3 wheel; needs a git clone or a vendored config/converter
python3 $OMM/scripts/ft_run.py Nequix-MP-1 --dataset <your-dataset> --out ft_nequix-mp-1
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `Nequix-MP-1` | documented | False | — | — |

---

## DeePMD — env `deepmd`

### A. What upstream documents

Upstream ([source](https://docs.deepmodeling.com/projects/deepmd/en/master/train/finetuning.html)):

```bash
dp --pt train input.json --finetune pretrained.pt                             # single-task, load fitting net from checkpoint
dp --pt train input.json --finetune pretrained.pt --model-branch RANDOM       # single-task, randomly re-init fitting net
dp --pt train input.json --finetune pretrained.pt --use-pretrain-script       # model section unknown -> take it from the checkpoint
dp --pt show multitask_pretrained.pt model-branch                             # list heads of a multi-task checkpoint
dp --pt train input.json --finetune multitask_pretrained.pt --model-branch CHOOSEN_BRANCH   # multi-task, pick a head (doc really spells it CHOOSEN_BRANCH)
dp --pt freeze -c <ckpt-dir-or-prefix> -o model.pth --head <branch>           # post-FT deployment
```

> **Note:** Multi-task fine-tuning (keeping pretraining branches alive alongside the downstream branch, anti-forgetting) uses a multi_input.json with model.model_dict.<DOWNSTREAM>.finetune_head: <PRE_DATA_branch> plus training.model_prob branch-sampling weights (verbatim doc shape in report B §1). The TensorFlow-backend equivalent (dp train input.json --finetune pretrained.pb) exists but is single-task only and overwrites type_map from the pretrained model; kept for completeness, not used by this hub (PyTorch-only registry).

### B. Dataset format

DeePMD's own 'system' layout (type.raw/type_map.raw/set.NNN/{coord,energy,force}.npy), NOT extxyz/ASE directly. Converter is dpdata (NOT installed in the local env — pip install dpdata first): dpdata.MultiSystems.from_file('train.extxyz', fmt='extxyz').to('deepmd/npy', 'deepmd_data'). DPA-style heterogeneous compositions use the deepmd/npy/mixed layout (only the DPA-1/DPA-2 descriptors support it).

### C. The command this hub generates

`DPA-3.1-3M-FT`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - dpdata is not installed in the deepmd env -- pip install dpdata before converting any ASE-readable data
#   - the shipped weight dpa-3.1-3m-ft.pth must be symlinked to a .pt name before dp --pt show/--finetune will dispatch on it correctly
python3 $OMM/scripts/ft_run.py DPA-3.1-3M-FT --dataset <your-dataset> --out ft_dpa-3.1-3m-ft
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `DPA-3.1-3M-FT` | documented | False | — | ft_verify: energy=-16.4316 eV on a 2-epoch demo fine-tune, 2026-08-22 |

---

## ORB — env `orb`

### A. What upstream documents

Upstream ([source](https://github.com/orbital-materials/orb-models/blob/main/FINETUNING_GUIDE.md)):

```bash
python finetune.py --data_path /path/to/dataset.db --dataset my_dataset --base_model orb_v3_conservative_inf_omat --batch_size 32 --num_steps 100 --max_epochs 50 --lr 3e-4 --save_every_x_epochs 5 --checkpoint_path ./ckpts
```

> **Note:** Version pinning matters: the main-branch docs describe orbmol_v2, loss_weights=, train_reference_energies=, and a (model, atoms_adapter) tuple return from pretrained.* — NONE of these exist in the installed 0.5.5 (pretrained.orb_v3_conservative_inf_omat returns a single model object; passing main-branch-only kwargs TypeErrors). No early stopping (use --save_every_x_epochs); the LR schedule is hardcoded to OneCycleLR over max_epochs*num_steps.

### B. Dataset format

ASE sqlite .db (row.energy/row.forces/row.stress read via ase.db). AseSqliteDataset.__getitem__ does db.get(idx+1) — rows must be 1-INDEXED and DENSE. Convert any ASE-readable trajectory with a plain read() + SinglePointCalculator(..., stress=atoms.get_stress(voigt=True)) + db.write(atoms) loop (stress required unless model.has_stress is False).

### C. The command this hub generates

`ORB-v3`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - finetune.py is not shipped in the orb-models wheel -- curl it from the repo at the tag matching the installed version (v0.5.5)
#   - wandb is a hard top-level import in v0.5.5's finetune.py and is not installed in the orb env -- pip install wandb (and set WANDB_MODE=offline since --wandb cannot be disabled from the CLI in this version)
python3 $OMM/scripts/ft_run.py ORB-v3 --dataset <your-dataset> --out ft_orb-v3
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `ORB-v3` | documented | False | — | — |

---

## GRACE — env `grace`

### A. What upstream documents

Upstream ([source](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/)):

```bash
grace_models list                      # only names showing a CHECKPOINT: line are fine-tunable
grace_models checkpoint <MODEL-NAME>   # optional -- gracemaker auto-downloads it if missing
gracemaker input.yaml
```

> **Note:** fit.trainable_variable_names (e.g. ['I2/reducing_','rho/reducing_','I1/reducing_']) restricts training to variables whose name matches one of the given prefixes — GRACE's closest analogue to LoRA/head-only fine-tuning. TensorFlow-based: gracemaker --help emits benign cuFFT/cuDNN 'already registered' warnings on every invocation. Export after FT: gracemaker -s (SavedModel) or -sf (FS model for LAMMPS).

### B. Dataset format

data.filename accepts extxyz (dispatched by extension, read via ase.io.read(..., format='extxyz')) or a pandas pickle (.pkl.gz) with columns ase_atoms/energy_corrected/forces/stress/reference_energy. For extxyz, energy and forces come from the ASE calculator (ase_atoms.get_potential_energy()/get_forces()), same convention as SevenNet. grace_collect builds a dataset from a tree of VASP run directories; grace_preprocess pre-tokenises.

### C. The command this hub generates

`GRACE-2L-OAM`:

```bash
python3 $OMM/scripts/ft_run.py GRACE-2L-OAM --dataset <your-dataset> --out ft_grace-2l-oam
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `GRACE-2L-OAM` | documented | True | — | — |

---

## MatterSim — env `mattersim`

### A. What upstream documents

Upstream ([source](https://github.com/microsoft/mattersim/blob/main/docs/user_guide/finetune.rst)):

```bash
torchrun --nproc_per_node=1 <env>/lib/python3.10/site-packages/mattersim/training/finetune_mattersim.py --load_model_path mattersim-v1.0.0-5m --train_data_path train.xyz --valid_data_path valid.xyz --save_path ./results --save_checkpoint --epochs 100 --batch_size 16 --lr 2e-4 --include_forces --include_stresses --device cuda
```

> **Note:** MUST be launched under torchrun even for a single GPU/CPU run — the script reads os.environ['LOCAL_RANK'] unconditionally at module load and calls torch.distributed.init_process_group; plain python finetune_mattersim.py dies with KeyError. --save_checkpoint defaults to False (nothing is written to --save_path without it). --include_stresses defaults False while --include_forces defaults True. --re_normalize recomputes scale/shift from the new data when the fine-tune labels come from a different functional/reference. The docs' own finetune.html page 404s; the live source is docs/user_guide/finetune.rst.

### B. Dataset format

Anything ase.io.read handles, multi-frame (AtomsAdaptor.from_file), or a pickled list of Atoms (.pkl). Plain extxyz with attached energies/forces/stress in the ASE calculator is the zero-work path. The script itself divides get_stress(voigt=False) by ase.units.GPa — supply plain ASE stress (eV/A^3), do not pre-convert.

### C. The command this hub generates

`MatterSim-v1-5M`:

```bash
python3 $OMM/scripts/ft_run.py MatterSim-v1-5M --dataset <your-dataset> --out ft_mattersim-v1-5m
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `MatterSim-v1-5M` | documented | True | — | — |

---

## CHGNet — env `chgnet`

### A. What upstream documents

Upstream ([source](https://github.com/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb)):

```bash
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
chgnet = CHGNet.load()   # model_name='0.3.0' by default
trainer = Trainer(model=chgnet, targets='efsm', optimizer='Adam', scheduler='CosLR', criterion='MSE', epochs=5, learning_rate=1e-2, use_device='cuda')
trainer.train(train_loader, val_loader, test_loader)
```

> **Note:** Energy label MUST be eV/atom, not total energy (the notebook variable is literally energies_per_atom). Stress units are VASP raw kBar, with a documented SIGN FLIP (verbatim): '...the -10 unit conversion modifies it to be kbar in VASP raw unit... If you're using stress labels from VASP, you don't need to do any unit conversions.' GGA/GGA+U-compatible data should go through MaterialsProject2020Compatibility first; non-MP functionals (r2SCAN, QE, Gaussian, ...) should refit the isolated-atom reference via Trainer(..., train_composition_model=True) or AtomRef.fit(...) directly. Selective layer freezing (chgnet.atom_embedding, bond_embedding, ..., atom_conv_layers[:-1], ...) is shown as an optional step before constructing the Trainer.

### B. Dataset format

pymatgen Structure objects + energies_per_atom (eV/ATOM, not total energy) + forces (eV/A) + optional stresses/magmoms, via StructureData. From ASE: AseAtomsAdaptor.get_structure(atoms) + atoms.get_potential_energy()/len(atoms). Stress must be VASP raw kBar = ASE eV/A^3 * 1602.1766208 (with a documented sign flip, see note). parse_vasp_dir(file_root=...) / StructureData.from_vasp(...) build a dataset directly from a VASP run directory.

### C. The command this hub generates

`CHGNet-v0.3.0`:

```bash
python3 $OMM/scripts/ft_run.py CHGNet-v0.3.0 --dataset <your-dataset> --out ft_chgnet-v0.3.0
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `CHGNet-v0.3.0` | documented | True | — | — |

---

## AlphaNet — env `alphanet`

### A. What upstream documents

Upstream ([source](https://github.com/zmyybc/AlphaNet)):

```bash
alpha-train mp.json --num_devices 1 --resume --ckpt_path /path/to/checkpoint.ckpt   # nearest DOCUMENTED command -- see semantics: this is a Lightning RESUME, not a fine-tune
```

> **Note:** Loss-weight guidance from alphanet/config.py's own docstring: energy:force:stress around 4:100:100 for systems under ~300 atoms, or a dynamic ramp starting at 0.01:100:100 with lr 5e-4 for larger/high-per-atom-energy systems. To fine-tune the shipped foundation checkpoints the architecture block must match pretrained/MPtrj/mp.json or pretrained/OMA/oma.json verbatim (num_layers, hidden_channels, cutoff, num_radial, compute_forces/compute_stress, use_pbc, ...). TrainConfig defaults force_coef/stress_coef to 0.0 and AlphaConfig defaults compute_forces to False — both must be set explicitly for an MLIP run.

### B. Dataset format

Pickled dict {'E','F','R','z','cell','natoms','stress'} under dataset/<name>/raw/*.pickle, consumed by CustomPickleDataset (a PyG InMemoryDataset). Repo converters: scripts/dp2pic_batch.py (from deepmd) and scripts/xyz2pic.py (from extxyz) — the latter REQUIRES atoms.info['virial'] (frames without it are skipped) and reads atoms.arrays['force'], not get_forces().

### C. The command this hub generates

`AlphaNet-v1-OMA` -- **code-excavation-needed**. `ft_run.py AlphaNet-v1-OMA` refuses (exit 2): alpha-train's only checkpoint-consuming argument is a Lightning resume that expects the Lightning checkpoint envelope (state_dict + optimizer_states + epoch); the released foundation weights are bare state_dict files the ASE calculator loads directly, and no shipped code path connects the two

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `AlphaNet-v1-OMA` | code-excavation-needed | False | — | — |

---

## Eqnorm — env `eqnorm`

### A. What upstream documents

Upstream ([source](https://github.com/yzchen08/eqnorm)):

_No fine-tuning command exists upstream — see block D for why._

> **Note:** The README states the shipped eqnorm-mptrj weights were 'trained on the MPtrj dataset' but gives no instructions for a user to train or fine-tune; the eqnorm-*.py files are model-ARCHITECTURE definitions and the yaml files are inference-side settings only. The full repo tree (GitHub trees API, main) confirmed no scripts/ or examples/ directory. Writing fine-tuning support would mean building a training loop (loss, optimizer, dataloader, graph construction) from scratch against the model classes — beyond 'excavation'.

### B. Dataset format

n/a.

### C. The command this hub generates

`Eqnorm-MPtrj` -- **not-supported**. `ft_run.py Eqnorm-MPtrj` refuses (exit 2): zero training code exists in the released package or repo
  - evidence: https://github.com/yzchen08/eqnorm/blob/main/README.md
  - evidence: https://github.com/yzchen08/eqnorm (full repo tree, GitHub trees API)

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `Eqnorm-MPtrj` | not-supported | False | — | — |

---

## fairchemv1 — env `fairchemv1`

### A. What upstream documents

Upstream ([source](https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.10.0/docs/core/common_tasks/fine-tuning/fine-tuning.md)):

```bash
fairchem --mode train --config-yml config.yml --checkpoint /path/to/esen_30m_oam.pt --run-dir fine-tuning --identifier ft-esen --num-gpus 1 --amp
# repo checkout only (doc's own example, {} = python-side interpolation):
# python {fairchem_main()} --mode train --config-yml {yml} --checkpoint {checkpoint_path} --run-dir fine-tuning --identifier ft-oxides --cpu
```

> **Note:** eSEN-specific excavation (code, not docs): config['trainer'] == 'mlip_trainer' (registered at fairchem.core.models.esen.trainers.trainer; keep as-is, do NOT delete it via generate_yml_config's delete= list, which was written for GemNet-OC). config['model'] is a hydra composite ({'name':'hydra','backbone':{'model':'esen_backbone',...},'heads':{'mptrj':{'module':'esen_mlp_efs_head'}}}) with ONE head named 'mptrj' — head selection means editing model.heads plus the matching config['outputs']/config['loss_functions'] keys. optim.loss_force and optim.load_balancing must still be deleted from the generated YAML (the tutorial's delete list) or the run errors. There is no eSEN training YAML anywhere in the v1.10.0 repo — every eSEN-specific fact here was excavated from the installed package and the loaded checkpoint, not from a doc page.

### B. Dataset format

ASE db (ase_db format, .db written via ase.db.connect + SinglePointCalculator) or LMDB. train_test_val_split(...) splits one db into train/test/val. Set a2g_args.r_energy/r_forces (and r_stress for eSEN's EFS head) to match the labels actually present.

### C. The command this hub generates

`eSEN-30M-OAM`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - fairchem_main() resolves to <repo_root>/main.py, which does not exist in the pip-installed fairchemv1 env -- use the installed fairchem console script instead (identical flags)
python3 $OMM/scripts/ft_run.py eSEN-30M-OAM --dataset <your-dataset> --out ft_esen-30m-oam
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `eSEN-30M-OAM` | documented (generic v1) + code-excavation for eSEN specifics | False | — | — |

---

## EquiformerV3 — env `equiformer_v3`

### A. What upstream documents

Upstream ([source](https://github.com/atomicarchitects/equiformer_v3)):

```bash
git clone https://github.com/atomicarchitects/equiformer_v3.git      # the pip package ships only fairchem/experimental/{models,trainers} -- configs/scripts/main.py are repo-only
python experimental/tasks/remove_key_from_checkpoint.py --input-path <checkpoint.pt> --remove-name energy_block   # README says --remove-key; the actual typer flag is --remove-name
torchrun --nproc_per_node=<N> my_main.py --num-gpus <N> --num-nodes 1 --mode train --amp --config-yml <edited-config>.yml --run-dir <log-dir> --print-every 200 --seed 1 --identifier my_finetune --optim.num_workers=0
```

> **Note:** No documented procedure exists for fine-tuning the released HuggingFace checkpoint on user data — the README documents the AUTHORS' pretraining and gradient-fine-tuning runs with exact commands, but the mechanism that makes a user fine-tune possible (optim.load_pretrained_weights) is an undocumented YAML key, confirmed only by reading the installed trainer source. use_denoising_pos: False is the fine-tune setting (DeNS regularization is on during pretraining, off during gradient FT).

### B. Dataset format

.aselmdb plus a metadata.npz recording per-structure edge counts for load balancing. experimental/datasets/mptrj_convert_json_to_aselmdb.py + create_metadata_num_edges.py --input-dir <dir> (then cp metadata_num-edges.npz metadata.npz) are the repo's own converters; there is no documented extxyz-to-aselmdb converter in this repo specifically — the practical route is fairchem's create_finetune_dataset.py (shared with UMA/eSEN) followed by create_metadata_num_edges.py. README data-hygiene rule: structures with any atom having no neighbor within 6A are removed.

### C. The command this hub generates

`EqV3-OMatMPtrjSalex` -- **code-excavation-needed**. `ft_run.py EqV3-OMatMPtrjSalex` refuses (exit 2): no documented procedure exists for fine-tuning the released HF checkpoint on user data -- the mechanism (optim.load_pretrained_weights) is an undocumented YAML key confirmed only by reading the installed trainer source

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `EqV3-OMatMPtrjSalex` | code-excavation-needed | False | — | — |

---

## UMA — env `uma`

### A. What upstream documents

Upstream ([source](https://fair-chem.github.io/core/common_tasks/fine_tuning.html)):

```bash
git clone git@github.com:facebookresearch/fairchem.git
pip install -e fairchem/src/packages/fairchem-core[dev]
python src/fairchem/core/scripts/create_uma_finetune_dataset.py --train-dir <train_dir> --val-dir <val_dir> --output-dir <out> --uma-task=<task> --regression-tasks <e|ef|efs>   # NOTE: --regression-tasks is PLURAL and required on the installed 2.19.1 — the doc's singular --regression-task fails as written
fairchem -c <out>/uma_sm_finetune_template.yaml base_model_name=uma-s-1p2 epochs=2 lr=2e-4 job.run_dir=<run_dir> +job.timestamp_id=<id>
fairchem -c <run_dir>/<id>/checkpoints/final/resume.yaml    # resume, NOT fine-tune -- restores training state
```

> **Note:** 'While UMA was trained in a multi-task fashion, we ONLY support fine-tuning on a single UMA task at a time' (doc, verbatim) — --uma-task selects the task (a per-variant divergence, carried as variant_args in models.json). DOC-VS-INSTALLED (report B, all three real): (1) --regression-task (doc) vs --regression-tasks (installed, required, plural) — folded into cmd above; (2) doc lists uma-task choices omol/odac/oc20/oc22/oc25/omat/omc but the installed UMATask enum is only ['omol','omat','odac','oc20','oc25','omc'] — oc22 is GONE and rejected by argparse choices (see UMA-s-1p2-OC22 in models.json, which does NOT inherit this family's runnable_as_installed); (3) the configs/ blocker above. --regression-tasks in {e, ef, efs} controls which of energy/energy+forces/energy+forces+stress are trained (gradients for the others remain computable regardless). Security note from the docs, echoed here: never run a YAML config from an untrusted source — Hydra instantiates Python objects from the _target_ key.

### B. Dataset format

ASE-LMDB (.aselmdb). Doc, verbatim: 'the only requirement is that you have input files that can be read as ASE atoms objects by ase.io.read and that they contain energy (forces, stress) in the correct format' — so plain extxyz/.traj/.cif in a train dir + a val dir feed create_uma_finetune_dataset.py directly, which also computes the force RMS normalizer and per-element linear reference and stamps them into the data YAML.

### C. The command this hub generates

`UMA-m-1p1-OC20`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-m-1p1-OC20 --dataset <your-dataset> --out ft_uma-m-1p1-oc20
```

`UMA-m-1p1-OMAT`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-m-1p1-OMAT --dataset <your-dataset> --out ft_uma-m-1p1-omat
```

`UMA-s-1p1-OC20`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-s-1p1-OC20 --dataset <your-dataset> --out ft_uma-s-1p1-oc20
```

`UMA-s-1p1-OMAT`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-s-1p1-OMAT --dataset <your-dataset> --out ft_uma-s-1p1-omat
```

`UMA-s-1p2-OC20`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-s-1p2-OC20 --dataset <your-dataset> --out ft_uma-s-1p2-oc20
```

`UMA-s-1p2-OC22`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
#   - oc22 absent from the installed UMATask enum (report B:487-489): the enum is ['omol','omat','odac','oc20','oc25','omc'] -- --uma-task=oc22 is rejected by argparse choices, so this variant does NOT inherit the UMA family's runnable_as_installed even relative to the other blockers
python3 $OMM/scripts/ft_run.py UMA-s-1p2-OC22 --dataset <your-dataset> --out ft_uma-s-1p2-oc22
```

`UMA-s-1p2-OC25`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-s-1p2-OC25 --dataset <your-dataset> --out ft_uma-s-1p2-oc25
```

`UMA-s-1p2-OMAT`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 wheel -- create_uma_finetune_dataset.py hardcodes a relative TEMPLATE_DIR; needs the fairchem repo cloned with cwd at the repo root
python3 $OMM/scripts/ft_run.py UMA-s-1p2-OMAT --dataset <your-dataset> --out ft_uma-s-1p2-omat
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `UMA-m-1p1-OC20` | documented | False | — | — |
| `UMA-m-1p1-OMAT` | documented | False | — | — |
| `UMA-s-1p1-OC20` | documented | False | — | — |
| `UMA-s-1p1-OMAT` | documented | False | — | — |
| `UMA-s-1p2-OC20` | documented | False | — | — |
| `UMA-s-1p2-OC22` | documented | False | — | — |
| `UMA-s-1p2-OC25` | documented | False | — | — |
| `UMA-s-1p2-OMAT` | documented | False | — | — |

---

## PET — env `pet`

### A. What upstream documents

Upstream ([source](https://docs.metatensor.org/metatrain/latest/concepts/fine-tuning.html)):

```bash
mtt train options-ft.yaml -o model-ft.pt
mtt eval model-ft.pt options-ft-eval.yaml -o output-ft.xyz
```

> **Note:** Three fine-tune methods share the read_from key: method: full (all weights trainable), method: heads (backbone frozen, config.head_modules/last_layer_modules list what stays trainable), method: lora (config.target_modules/rank/alpha; the concept page has no LoRA YAML example — this shape is excavated from metatrain/pet/modules/finetuning.py and its own test file). Fine-tuning support is PET/FlashMD/FlashMDSymplectic/SPACE only, not GAP/SOAP-BPNN. A full or lora run creates a NEW head named energy/<variantname> and DROPS every target not in that run's training set; a heads run keeps them. CRITICAL FOR OH-MY-MLIP: the resulting model does not answer to the plain 'energy' output — MetatomicCalculator(..., variants={'energy': '<variantname>'}) is required at calculator construction (and pair_style metatomic ... variant <name> in LAMMPS), which differs from the hub's stock PET-OAM-XL calculator construct.

### B. Dataset format

Direct — no conversion step. training_set.systems.read_from: <file> + reader: ase, and targets.<name>.read_from/key for energy and forces (units declared explicitly, e.g. length_unit: angstrom, unit: eV).

### C. The command this hub generates

`PET-OAM-XL`:

```bash
python3 $OMM/scripts/ft_run.py PET-OAM-XL --dataset <your-dataset> --out ft_pet-oam-xl
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `PET-OAM-XL` | documented | True | — | — |

---

## EquFlash — env `equflash`

### A. What upstream documents

Upstream ([source](https://github.com/SamsungDS/GGNN)):

```bash
python -m GGNN.main --mode train --config-yml equflash-ft.yml --num-gpus 1  # installed-package equivalent; repo root ships python main.py --mode train --config-yml configs/equflash/equflash.yml
```

> **Note:** LICENCE HAZARD: GGNN/LICENSE.md is CC BY-NC-SA 4.0 (NonCommercial + ShareAlike) — a fine-tuned EquFlash model is a derivative work and inherits both restrictions; third-party code inside the repo (fairchem-core, NequIP, SevenNet) is separately MIT, but the GGNN package as a whole is not. Fairchem-v1 vintage API throughout (--config-yml, registry, submitit), not fairchem v2. Released checkpoints to fine-tune from: equflash-OAM, equflash-OMat24, equflashv2-OAM, equflashv2-OMat24 (README table, figshare URLs).

### B. Dataset format

format: ase_db (fairchem-v1 AseDBDataset) is the safe/tested path — any format ase.db.connect accepts (.db, .aselmdb), built with a plain ase.io.read(...) + ase.db.connect(...).write(atoms) loop. format: ase_read_multi reads a directory of extxyz files directly without a conversion step (GGNN's own ase_arrays_read_multi is also registered). a2g_args.{r_energy,r_forces,r_stress} selects which labels are read.

### C. The command this hub generates

`EquFlashV2`:

```bash
# LICENCE: CC-BY-NC-SA-4.0 -- https://creativecommons.org/licenses/by-nc-sa/4.0/
#   (Part 8 decision (a)/option B: disclosed, not refused --
#    oh-my-mlip is MIT and redistributes no weights either way)
# BLOCKED -- ft_run.py exits 3 until:
#   - main.py and configs/ are repo-only -- not present in the installed GGNN 0.1 package; use python -m GGNN.main as the installed equivalent
python3 $OMM/scripts/ft_run.py EquFlashV2 --dataset <your-dataset> --out ft_equflashv2
```

`EquFlash-v1`:

```bash
# LICENCE: CC-BY-NC-SA-4.0 -- https://creativecommons.org/licenses/by-nc-sa/4.0/
#   (Part 8 decision (a)/option B: disclosed, not refused --
#    oh-my-mlip is MIT and redistributes no weights either way)
# BLOCKED -- ft_run.py exits 3 until:
#   - main.py and configs/ are repo-only -- not present in the installed GGNN 0.1 package; use python -m GGNN.main as the installed equivalent
python3 $OMM/scripts/ft_run.py EquFlash-v1 --dataset <your-dataset> --out ft_equflash-v1
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `EquFlashV2` | documented (config-level) | False | CC-BY-NC-SA-4.0 | — |
| `EquFlash-v1` | documented (config-level) | False | CC-BY-NC-SA-4.0 | — |

---

## MatRIS — env `matris`

### A. What upstream documents

Upstream ([source](https://github.com/HPC-AI-Team/MatRIS)):

_No fine-tuning command exists upstream — see block D for why._

> **Note:** The README states matris_10m_oam was itself 'trained on the OMat24 dataset, and finetuned on sAlex+Mptrj dataset' — i.e. the authors fine-tuned internally with code they did not release. Fine-tuning MatRIS would require writing a trainer from scratch against matris.model.model; open repo issues concern checkpoint URLs and a torch-sim interface, none provide training code. Licence is BSD-3-Clause (permissive) — not the blocker here.

### B. Dataset format

n/a.

### C. The command this hub generates

`MatRIS-10M-OAM` -- **not-supported**. `ft_run.py MatRIS-10M-OAM` refuses (exit 2): public release is inference-only -- applications/ contains only the calculator, StructOptimizer and MolecularDynamics, no training code
  - evidence: https://github.com/HPC-AI-Team/MatRIS
  - evidence: https://arxiv.org/html/2603.02002v1

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `MatRIS-10M-OAM` | not-supported | False | — | — |

---

## DPA4 — env `dpa4`

### A. What upstream documents

Upstream ([source](https://docs.deepmodeling.com/projects/deepmd/en/master/train/finetuning.html)):

```bash
dp --pt train input.json --finetune "$OH_MY_MLIP_HOME/models/dpa4/dpa-4.0.1-pro-mptrj.pt" --use-pretrain-script
```

> **Note:** Byte-for-byte the same CLI surface as DeePMD/DPA-3.1 (verified: dp --pt train --help is character-for-character identical between the two envs). --model-branch is meaningless here because the shipped checkpoint is single-task, unlike DPA-3.1-3M-FT's 31-head multi-task checkpoint — do not pass it. Also needs LD_LIBRARY_PATH="/usr/lib/wsl/lib" (this hub's env_run for dpa4) at run time.

### B. Dataset format

Identical to DeePMD/DPA-3.1 — deepmd 'system' npy/hdf5 layout, converted from extxyz via dpdata (also missing from this env).

### C. The command this hub generates

`DPA-4.0.1-pro-MPtrj`:

```bash
# BLOCKED -- ft_run.py exits 3 until:
#   - dpdata is not installed in the dpa4 env -- pip install dpdata before converting any ASE-readable data
#   - the dpa4 descriptor type only exists in deepmd-kit 3.2.0b0 (the dpa4 env) -- the deepmd env's 3.1.2 cannot parse it at all
python3 $OMM/scripts/ft_run.py DPA-4.0.1-pro-MPtrj --dataset <your-dataset> --out ft_dpa-4.0.1-pro-mptrj
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `DPA-4.0.1-pro-MPtrj` | documented | False | — | — |

---

## TACE — env `tace`

### A. What upstream documents

Upstream ([source](https://tace.readthedocs.io/en/latest/guide/finetune.html)):

```bash
tace-finetune -m TACE-OMat24-7M.pt        # generates finetune_config.yaml (LoRA + full parameter freeze, by default)
rm -f finetune_config.yaml                # omit this step to keep the generated LoRA config; delete it to force FULL-parameter fine-tuning
tace-train -cn tace                        # edit tace.yaml first: finetune_from_model: /path/to/TACE-OMat24-7M.pt
tace-convert -m checkpoints_epoch/last.ckpt     # merges LoRA weights into the base model -- see blockers: omit --type, its documented value is broken
```

> **Note:** REAL BUG in the installed TACE 0.2.0: tace-convert's --type flag defaults to 'merge_lora' but ALLOWED_TYPE only contains 'merged_lora' — the documented --type merge_lora is rejected by argparse choices, and passing --type merged_lora instead reaches a body check that only accepts the literal 'merge_lora' and raises ValueError. Workaround: omit -t/--type entirely (argparse does not validate its own default against choices, so the broken default reaches the merge branch and works) — worth reporting upstream. tace-train is Hydra-driven with config_path=Path.cwd() and MUST be run from the directory containing the config. Replay data (mixing foundation-model training data back in) is explicitly NOT implemented during fine-tuning. Full-parameter fine-tuning is discouraged by the docs for small datasets (overfitting + catastrophic forgetting) in favor of the LoRA default. Licence: MIT, no constraints.

### B. Dataset format

Best of the 20: no conversion needed. dataset.type: ase reads any file ase.io.read handles directly, or ase-db (.db / .aselmdb). Property keys remappable under dataset.keys (~28 keys: energy_key, forces_key, stress_key, ...). storage_mode: lmdb for large datasets.

### C. The command this hub generates

`TACE-OAM-L`:

```bash
python3 $OMM/scripts/ft_run.py TACE-OAM-L --dataset <your-dataset> --out ft_tace-oam-l
```

### D. Per-variant support matrix

| variant | status | runnable_as_installed | licence | demonstrated |
|---|---|---|---|---|
| `TACE-OAM-L` | documented | True | — | — |

---
