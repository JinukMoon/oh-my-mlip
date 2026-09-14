"""Upstream-documented fine-tuning procedures, one entry per framework.

Sibling table to `upstream_recipes.py` (install/fetch), but for the FT-from-a-
foundation-checkpoint path: `mace_run_train --foundation_model=...`,
`sevenn train input.yaml` with `train.continue.checkpoint`, and so on for all
20 registry families. Every command, config key and blocker below is either
quoted verbatim from official upstream documentation, or excavated from the
source of the version actually installed in `/home/jumoon/miniconda3/envs/`
(read-only research, 2026-08-22; full write-ups: `.omc/research/finetune/
{A-major-cli,B-deepmodeling-meta,C-misc-torch,D-rest}.md`). Provenance is
carried through from those reports: **[DOC]** = verbatim from official docs,
**[CODE]** = excavated from installed package source / `--help` output,
**[LOCAL-VERIFIED]** = flag/key confirmed present in our pinned version.

`scripts/gen_finetune.py` (P2b) renders this table into `docs/finetune.md`,
mirroring how `gen_recipes.py` renders `UPSTREAM` into `docs/recipes.md`. When
upstream changes its documented fine-tuning procedure, edit THIS file — never
a generated doc.

Field guide (one row per family; per-variant divergences live in
`models.json`'s `finetune.variant_args`, not here — see AGENTS.md §3C):

  status                  'documented' | 'code-excavation-needed' | 'not-supported'
                           (doc-honesty state; NOT an execution claim — see
                           `demonstrated` in models.json for that)
  entrypoint               the console command or API this hub shells out to
  src                       doc URL the `cmd`/`config` were read from
  cmd                       exact FT command(s), verbatim from upstream, with any
                            correction the reports found against the INSTALLED
                            version folded in (e.g. UMA's `--regression-tasks`,
                            plural, not the doc's singular `--regression-task`)
  config                    minimal training-config snippet the command consumes,
                            or None where the framework is flag-only (ORB,
                            MatterSim) or API-only (CHGNet)
  data_format               what the trainer actually reads, and the shortest
                            path there from an ASE-readable file/trajectory
  foundation                how the pretrained checkpoint is referenced/selected
  semantics                 the finetune-vs-resume distinction for this
                            framework: `resume_from`/`--restart`/
                            `resume_from_model`/Lightning `ckpt_path` restore
                            optimizer/scheduler/epoch state; `finetune_from`/
                            `finetune.read_from`/`task.finetune.checkpoint`/
                            `finetune_from_model`/`--foundation_model`/
                            `--finetune`/`--checkpoint` load weights only.
                            Conflating the two yields a wrong run that still
                            **exits 0** — invisible to every oracle in this repo.
  selector_kind             'cli_flag' | 'config_key' | None (None only for the
                            two not-supported families and for AlphaNet, where
                            no fine-tune selector exists in the shipped code)
  selector_path             the flag name or config dotted-path that names the
                            checkpoint
  runnable_as_installed     True iff every script/config/dependency the `cmd`
                            needs is present in THIS repo's pinned env, with no
                            git clone and no extra pip install
  blockers                  list[str]; empty iff runnable_as_installed
  licence                   only set where the FT'd checkpoint itself carries a
                            non-permissive licence a user should see before
                            running (EquFlash: CC BY-NC-SA 4.0; MACE MH-1
                            checkpoints: ASL) — None everywhere else. This repo
                            is MIT and redistributes no weights either way; the
                            field is disclosure, not a legal gate (Part 8
                            decision (a) in the consensus plan: advertise +
                            disclose, do not refuse).
  note                      free-form caveats a fine-tune codegen must respect
  evidence                  list[str] of URLs; populated only for the two
                            not-supported families (Eqnorm, MatRIS), where
                            `status` asserts an ABSENCE and needs a citation.
"""

UPSTREAM_FT: dict[str, dict] = {
    "SevenNet": dict(
        status="documented",
        entrypoint="sevenn train input.yaml",
        src="https://sevennet.readthedocs.io/en/latest/user_guide/cli.html",
        cmd=["sevenn_preset fine_tune > input.yaml     # or: sevenn preset fine_tune > input.yaml",
             "#  ... edit input.yaml: train.continue.checkpoint + data.load_trainset_path ...",
             "sevenn train input.yaml -s",
             "# with an accelerator: sevenn train input.yaml -s --enable_oeq   "
             "(also --enable_cueq, --enable_flash)"],
        config=["train:",
                "    continue:",
                "        reset_optimizer: True",
                "        reset_scheduler: True",
                "        reset_epoch: True",
                "        checkpoint: 'SevenNet-0_11July2024'   # keyword or path to a .pth",
                "data:",
                "    load_trainset_path: ['./train_*.extxyz']",
                "    load_validset_path: ['./valid.extxyz']"],
        data_format="Anything ase.io.read handles (extxyz, OUTCAR, ...), the legacy "
                     "structure_list format, or a pre-built graph sevenn_data/*.pt. Energy/"
                     "force/stress are read from the ASE calculator by default "
                     "(atoms.get_potential_energy()/get_forces()), NOT from info keys; "
                     "sevenn graph_build --kwargs / data.data_format_args rename the keys "
                     "when needed. A custom stress key is negated on ingest "
                     "(dataload.py:343) while the calculator path is not.",
        foundation="train.continue.checkpoint accepts either a shipped keyword "
                    "(SevenNet_0__11Jul2024, SevenNet_omat, SevenNet_MF_ompa, SevenNet_omni, "
                    "...) or a path to a .pth; newer keywords auto-download from GitHub "
                    "release tags.",
        semantics="finetune: continue.checkpoint + reset_optimizer/reset_scheduler/"
                   "reset_epoch: True (all True in the shipped fine_tune preset) loads "
                   "weights only. Leaving those three resets False/absent instead resumes "
                   "training state from the same checkpoint key — the flags, not the key, "
                   "are what distinguish fine-tune from resume here.",
        selector_kind="config_key",
        selector_path="train.continue.checkpoint",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="Installed sevenn 0.12.2.dev0 ships two presets beyond the CLI doc's list: "
             "fine_tune_le (liquid-electrolyte recipe: lr dropped 40x, shift/scale made "
             "trainable) and mf_ompa_fine_tune (path checkpoint for SevenNet-MF-ompa). The "
             "README's advertised 'fine-tuning with forgetting prevention' (replay + EWC) "
             "has no corresponding flag or YAML key in this version — an open gap, not "
             "claimed here.",
    ),
    "MACE": dict(
        status="documented",
        entrypoint="mace_run_train",
        src="https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html",
        cmd=["mace_run_train \\",
             "    --name=\"MACE\" \\",
             "    --foundation_model=\"medium-omat-0\" \\",
             "    --multiheads_finetuning=False \\",
             "    --train_file=\"train.xyz\" \\",
             "    --valid_fraction=0.05 \\",
             "    --energy_key=REF_energy --forces_key=REF_forces \\",
             "    --E0s=\"average\" --lr=0.01 --batch_size=2 --max_num_epochs=100 \\",
             "    --ema --ema_decay=0.99 --default_dtype=\"float64\" --device=cuda"],
        config=["name: mace_ft",
                "foundation_model: medium-omat-0",
                "multiheads_finetuning: false",
                "train_file: train.xyz",
                "valid_fraction: 0.05",
                "energy_key: REF_energy",
                "forces_key: REF_forces",
                "E0s: average",
                "lr: 0.01",
                "batch_size: 2",
                "max_num_epochs: 100",
                "default_dtype: float64",
                "device: cuda"],
        data_format="extxyz via --train_file/--valid_file/--test_file. Default keys are "
                     "NOT the plain ASE names: energy REF_energy, forces REF_forces, "
                     "virials REF_virials, stress REF_stress (--energy_key/--forces_key/... "
                     "override). Data written via ASE's SinglePointCalculator needs "
                     "--energy_key=energy --forces_key=forces, or the info/arrays keys must "
                     "be renamed to REF_*.",
        foundation="--foundation_model accepts a mace_mp_urls shorthand (small/medium/"
                    "large/*-0b*/medium-mpa-0/*-omat-0/mh-0/mh-1/...), 'small_off'|"
                    "'medium_off'|'large_off' -> mace_off(...) (MACE-OFF-2023, ASL-licensed), "
                    "'mace_omol' -> mace_omol(...), or otherwise a local checkpoint path.",
        semantics="finetune: --foundation_model loads weights only. "
                   "--multiheads_finetuning DEFAULTS TO TRUE in the installed 0.3.15 — a "
                   "bare mace_run_train --foundation_model=... --train_file=... is already "
                   "a multihead/replay run and will try to acquire a replay set; pass "
                   "--multiheads_finetuning=False for a naive single-head fine-tune. MACE "
                   "documents no separate resume-with-optimizer-state flag for a foundation "
                   "checkpoint.",
        selector_kind="cli_flag",
        selector_path="--foundation_model",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="--foundation_head selects which head of a multi-head foundation model to "
             "fine-tune (MH-1: 'omat_pbe' or 'oc20_usemppbe' — there is no plain 'oc20' "
             "head); this flag exists in the installed 0.3.15 CLI but appears in no "
             "official multihead-finetuning example. --pt_train_file=mp auto-downloads a "
             "Materials Project replay set; --pseudolabel_replay lets the foundation model "
             "label the replay data itself instead of shipping labels. LoRA fine-tuning "
             "(--lora_alpha, rank default 4) exists in the CLI with no documented example.",
    ),
    "NequIP": dict(
        status="documented",
        entrypoint="nequip-train -cp <dir> -cn <cfg>",
        src="https://nequip.readthedocs.io/en/latest/guide/training-techniques/fine_tuning.html",
        cmd=["nequip-train -cp full/path/to/config/directory -cn config_name.yaml"],
        config=["model:",
                "  _target_: nequip.model.ModelFromPackage",
                "  package_path: nequip.net:mir-group/NequIP-OAM-L:0.1   # or a local .nequip.zip",
                "model_type_names: ${type_names_from_package:path/to/model.nequip.zip}",
                "cutoff_radius: ${cutoff_radius_from_package:path/to/model.nequip.zip}",
                "data:",
                "  _target_: nequip.data.datamodule.ASEDataModule",
                "  train_file_path: train.extxyz",
                "  val_file_path: val.extxyz",
                "  transforms:",
                "    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                "      model_type_names: ${model_type_names}",
                "    - _target_: nequip.data.transforms.NeighborListTransform",
                "      r_max: ${cutoff_radius}"],
        data_format="extxyz (or any ASE-readable format) via nequip.data.datamodule."
                     "ASEDataModule (train_file_path/val_file_path/test_file_path). "
                     "Precedence when a quantity appears in more than one place: "
                     "atoms.arrays < atoms.info < atoms.calc.results — a SinglePointCalculator"
                     "-carrying extxyz needs no key_mapping at all.",
        foundation="model._target_: nequip.model.ModelFromPackage + package_path pointing "
                    "at a .nequip.zip, OR a nequip.net model ID directly (package_path: "
                    "nequip.net:<group>/<model>:<version>, e.g. "
                    "nequip.net:mir-group/NequIP-OAM-L:0.1) which auto-downloads. "
                    "ModelFromCheckpoint + checkpoint_path is the .ckpt-file variant, "
                    "documented as less version-stable than a package.",
        semantics="finetune: ModelFromPackage/ModelFromCheckpoint loads weights only into "
                   "a fresh training run. nequip-train -cp ... -cn ... "
                   "++ckpt_path='path/to/ckpt' is the SEPARATE resume mechanism — it "
                   "restores optimizer/scheduler state and is not fine-tuning.",
        selector_kind="config_key",
        selector_path="model._target_: ModelFromPackage + package_path",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="ft_run.py builder (host-inspected nequip 0.17.1 sources): config.yaml "
             "with run: [train]; data: ASEDataModule(train_file_path/val_file_path, "
             "transforms ChemicalSpeciesToAtomTypeMapper(chemical_symbols) + "
             "NeighborListTransform(r_max)); trainer: lightning.Trainer with a "
             "ModelCheckpoint(dirpath=<out>/checkpoints, filename=best, "
             "monitor=val0_epoch/weighted_sum, save_last) -> <out>/checkpoints/last.ckpt; "
             "training_module: EMALightningModule with model: ModelFromPackage, loss "
             "EnergyForceLoss, val_metrics EnergyForceMetrics, optimizer Adam. A prestage "
             "step (nequip_prestage.py) pins the nequip.net package to "
             "models/nequip/<version>.nequip.zip (NEQUIP_CACHE_DIR also pointed at "
             "models/nequip/model_cache) and fills model_type_names / cutoff_radius from "
             "GraphModel.type_names / .metadata['r_max'] instead of the resolvers, so the "
             "same emitted config also works for Allegro's nequip 0.15.0. Reload for "
             "verification: NequIPCalculator._from_saved_model(<ckpt>). "
             "cutoff_radius and model_type_names MUST match the foundation model exactly "
             "(a different r_max breaks the neighbour list) — current docs read them off "
             "the package via ${type_names_from_package:...}/${cutoff_radius_from_package:"
             "...} resolvers. DOC-VS-INSTALLED: those resolvers exist in the installed "
             "NequIP env (nequip 0.17.1) but NOT in the installed Allegro env (nequip "
             "0.15.0) — see the Allegro entry. Re-referencing energies to a new DFT setting "
             "uses nequip.model.modify + modify_PerTypeScaleShift (per-element shifts); "
             "only SUBSETS of the foundation model's atom types are supported, new types "
             "cannot be added. Post-FT packaging: nequip-package build <ckpt> "
             "<out>.nequip.zip, nequip-compile ... --mode aotinductor.",
    ),
    "Allegro": dict(
        status="documented (inherited from NequIP)",
        entrypoint="nequip-train -cp <dir> -cn <cfg>",
        src="https://github.com/mir-group/allegro",
        cmd=["nequip-train -cp full/path/to/config/directory -cn config_name.yaml"],
        config=["model:",
                "  _target_: nequip.model.ModelFromPackage",
                "  package_path: nequip.net:mir-group/Allegro-OAM-L:0.1   # or a local .nequip.zip",
                "  compile_mode: eager",
                "data:",
                "  _target_: nequip.data.datamodule.ASEDataModule",
                "  train_file_path: train.extxyz",
                "  val_file_path: val.extxyz"],
        data_format="Same as NequIP — ASEDataModule over extxyz, keys taken from the ASE "
                     "calculator with key_mapping available for overrides.",
        foundation="Same ModelFromPackage mechanism as NequIP; foundation checkpoint is "
                    "mir-group/Allegro-OAM-L (pretrained OMat24, fine-tuned sAlex+MPtrj), "
                    "fetchable via package_path: nequip.net:mir-group/Allegro-OAM-L:0.1.",
        semantics="Identical to NequIP: ModelFromPackage loads weights only; "
                   "nequip-train ++ckpt_path=... resumes training state and is a different "
                   "operation.",
        selector_kind="config_key",
        selector_path="model._target_: ModelFromPackage + package_path",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="Allegro is a model plugin inside the NequIP trainer, not a separate CLI — "
             "the Allegro env's bin/ contains only the nequip-* entry points, no "
             "allegro-train. Config shape RESOLVED from the installed sources: "
             "nequip/scripts/train.py asserts 'model' in config.training_module in BOTH "
             "0.15.0 and 0.17.1, so ft_run.py nests model: under training_module: "
             "(the fine-tuning doc's top-level model: is a shorthand). The missing "
             "0.15.0 resolvers are worked around by ft_run.py's nequip_prestage.py "
             "step: it fetches the nequip.net package once into "
             "models/<env>/<version>.nequip.zip (0.15.0's _get_model_file_path "
             "otherwise re-downloads to a temp file on every run), reads "
             "GraphModel.type_names / .metadata['r_max'] from it and writes them as "
             "literals into config.yaml. Checkpoint: <out>/checkpoints/last.ckpt "
             "(lightning ModelCheckpoint, save_last). Reload: "
             "NequIPCalculator._from_saved_model(<ckpt>) (the only uncompiled ASE path; "
             "from_compiled_model needs a nequip-compile step first).",
    ),
    "GRACE": dict(
        status="documented",
        entrypoint="gracemaker input.yaml",
        src="https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/",
        cmd=["grace_models list                      # only names showing a CHECKPOINT: line are fine-tunable",
             "grace_models checkpoint <MODEL-NAME>   # optional -- gracemaker auto-downloads it if missing",
             "gracemaker input.yaml"],
        config=["potential:",
                "  finetune_foundation_model: GRACE-1L-OAM",
                "  reduce_elements: True",
                "  shift: auto",
                "fit:",
                "  opt_params: {learning_rate: 0.001}",
                "  eval_init_stats: True",
                "  trainable_variable_names: [\"I2/reducing_\", \"rho/reducing_\", \"I1/reducing_\"]"],
        data_format="data.filename accepts extxyz (dispatched by extension, read via "
                     "ase.io.read(..., format='extxyz')) or a pandas pickle (.pkl.gz) with "
                     "columns ase_atoms/energy_corrected/forces/stress/reference_energy. For "
                     "extxyz, energy and forces come from the ASE calculator "
                     "(ase_atoms.get_potential_energy()/get_forces()), same convention as "
                     "SevenNet. grace_collect builds a dataset from a tree of VASP run "
                     "directories; grace_preprocess pre-tokenises.",
        foundation="potential.finetune_foundation_model: <name> (e.g. GRACE-1L-OAM) in "
                    "input.yaml; only names whose grace_models list entry shows a "
                    "CHECKPOINT: line are fine-tunable (GRACE-FS/1L/2L-OAM and -OMAT on this "
                    "box; the MP-only models are not). gracemaker auto-downloads the "
                    "checkpoint via get_or_download_checkpoint if the explicit grace_models "
                    "checkpoint step is skipped.",
        semantics="finetune: potential.finetune_foundation_model loads the checkpoint's "
                   "weights (cutoff inherited as 'n/a', learning_rate defaults to 0.001 "
                   "instead of 0.01). Ignored entirely when -r/-rl (restart) flags are "
                   "passed — those instead resume optimizer/epoch state from a run "
                   "directory, a separate mechanism.",
        selector_kind="config_key",
        selector_path="potential.finetune_foundation_model",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="ft_run.py builder (host-inspected tensorpotential 0.5.3 sources): a "
             "single `gracemaker input.yaml` with potential.finetune_foundation_model: "
             "<version> + reduce_elements: True, data.filename/test_filename pointing at "
             "the canonical extxyz (cli/data.py reads energy/forces from the ase "
             "calculator), fit.loss huber energy/forces, fit.maxiter = epochs. Output "
             "lands in seed/<seed>/: checkpoints/checkpoint.* plus an UNCONDITIONAL "
             "final_model tf.saved_model export after training (cli/gracemaker.py:578), "
             "so no separate `gracemaker -s` step is needed; the reload witness is "
             "seed/<seed>/final_model/saved_model.pb via TPCalculator(<dir>). The "
             "fine-tune CHECKPOINT (dict format, distinct from the saved_model the "
             "single-point inference uses) is fetched by gracemaker on first use into "
             "$GRACE_CACHE/checkpoints/<model>; ft_run.py exports "
             "GRACE_CACHE=$OH_MY_MLIP_HOME/models/grace (the hub's saved-model layout "
             "already matches <cache>/<model>), and `grace_models checkpoint <model>` "
             "prestages it without a training run. "
             "fit.trainable_variable_names (e.g. ['I2/reducing_','rho/reducing_',"
             "'I1/reducing_']) restricts training to variables whose name matches one of "
             "the given prefixes — GRACE's closest analogue to LoRA/head-only fine-tuning. "
             "TensorFlow-based: gracemaker --help emits benign cuFFT/cuDNN 'already "
             "registered' warnings on every invocation. Export after FT: gracemaker -s "
             "(SavedModel) or -sf (FS model for LAMMPS).",
    ),
    "Nequix": dict(
        status="documented",
        entrypoint="nequix_train <cfg>.yml",
        src="https://github.com/atomicarchitects/nequix",
        cmd=["nequix_train <config>.yml       # JAX backend, default",
             "# torch backend (single GPU): uv run nequix/torch_impl/train.py <config>.yml",
             "# torch backend (multi-GPU):  uv run torchrun --nproc_per_node=<gpus> "
             "nequix/torch_impl/train.py <config>.yml",
             "# Phonon fine-tuning (JAX-only): uv run nequix/pft/train.py "
             "configs/nequix-oam-1-pft.yml"],
        config=["finetune_from: \"models/nequix-omat-1.pt\"",
                "train_path:",
                " - \"data/mptrj-aselmdb\"",
                "valid_path: \"data/salex/val\"",
                "# + the base model's own architecture/atomic_numbers/atom_energies/scale/",
                "#   shift/avg_n_*/max_n_* keys, copied verbatim from its config"],
        data_format="AseDBDataset reads train_path/valid_path as a directory globbed for "
                     "*.aselmdb, or a single ASE-db file. scripts/preprocess_data.py "
                     "(repo-only, not in the installed wheel) converts any ASE-readable "
                     "file: ase.io.read(file, index=':') -> "
                     "ase.db.connect(...).write(atoms, data=atoms.info). Self-contained "
                     "local equivalent: a ~10-line loop writing an .aselmdb with the "
                     "installed ase_db_backends (verified working in-env).",
        foundation="finetune_from: <path>.pt/.pkl in the YAML config; the model-"
                    "architecture block, atomic_numbers, atom_energies, scale/shift and "
                    "avg_n_*/max_n_* stats must be copied from the base model's own config "
                    "— there is no minimal from-scratch finetune.yml template upstream.",
        semantics="finetune_from loads weights only (nequix/train.py:267-271 for JAX, "
                   "torch_impl/train.py:331-342 for torch, where atom_energies/scale/shift "
                   "MAY also be overridden on the torch path). resume_from is the separate "
                   "mechanism restoring full training state (optimizer/EMA/epoch). For "
                   "Phonon Fine-Tuning (nequix/pft/train.py), finetune_from is MANDATORY, "
                   "not optional.",
        selector_kind="config_key",
        selector_path="finetune_from",
        runnable_as_installed=False,
        blockers=["configs/, data/ and scripts/preprocess_data.py are repo-only — not "
                  "present in the installed nequix 0.4.3 wheel (only "
                  "nequix/{train,pft/train,torch_impl/train}.py ship); needs a git clone "
                  "or a vendored config/converter"],
        licence=None,
        note="JAX finetune_from RAISES NotImplementedError if atom_energies is also "
             "present in the config — a JAX fine-tune must not redefine isolated-atom "
             "energies (the torch backend can). batch_size in the config is per-device. "
             "nequix_train is the only installed console script; the torch/PFT trainers "
             "must be invoked by file path or via torchrun.",
    ),
    "DeePMD": dict(
        status="documented",
        entrypoint="dp --pt train input.json --finetune <ckpt>",
        src="https://docs.deepmodeling.com/projects/deepmd/en/master/train/finetuning.html",
        cmd=["dp --pt train input.json --finetune pretrained.pt                             "
             "# single-task, load fitting net from checkpoint",
             "dp --pt train input.json --finetune pretrained.pt --model-branch RANDOM       "
             "# single-task, randomly re-init fitting net",
             "dp --pt train input.json --finetune pretrained.pt --use-pretrain-script       "
             "# model section unknown -> take it from the checkpoint",
             "dp --pt show multitask_pretrained.pt model-branch                             "
             "# list heads of a multi-task checkpoint",
             "dp --pt train input.json --finetune multitask_pretrained.pt "
             "--model-branch CHOOSEN_BRANCH   # multi-task, pick a head "
             "(doc really spells it CHOOSEN_BRANCH)",
             "dp --pt freeze -c <ckpt-dir-or-prefix> -o model.pth --head <branch>           "
             "# post-FT deployment"],
        config=["{",
                "  \"model\": {\"type_map\": [\"O\", \"H\"], \"descriptor\": {}, \"fitting_net\": {}},",
                "  \"learning_rate\": {\"type\": \"exp\", \"decay_steps\": 5000, \"start_lr\": 0.001, \"stop_lr\": 3e-05},",
                "  \"loss\": {\"type\": \"ener\", \"start_pref_e\": 0.2, \"limit_pref_e\": 20, "
                "\"start_pref_f\": 100, \"limit_pref_f\": 60},",
                "  \"training\": {\"training_data\": {\"systems\": [\"../data/data_0\"], \"batch_size\": 1},",
                "                \"validation_data\": {\"systems\": [\"../data/data_3\"], \"batch_size\": 1},",
                "                \"numb_steps\": 1000000}",
                "}"],
        data_format="DeePMD's own 'system' layout (type.raw/type_map.raw/set.NNN/"
                     "{coord,energy,force}.npy), NOT extxyz/ASE directly. Converter is "
                     "dpdata (pinned dpdata==1.1.0 in envs/deepmd.yml): "
                     "dpdata.MultiSystems.from_file('train.extxyz', fmt='extxyz')."
                     "to('deepmd/npy', 'deepmd_data'). scripts/ft_dataset.py --to deepmd "
                     "performs this conversion (dpdata when importable, its own numpy "
                     "writer of the same layout otherwise). DPA-style heterogeneous "
                     "compositions use the deepmd/npy/mixed layout (only the DPA-1/DPA-2 "
                     "descriptors support it).",
        foundation="--finetune <path> where <path> MUST have a .pt extension — DeePMD "
                    "dispatches its training/show entry points on the FILE SUFFIX (.pth = "
                    "frozen TorchScript inference model, .pt = training checkpoint). The "
                    "oh-my-mlip artifact models/deepmd/dpa-3.1-3m-ft.pth is itself a "
                    "training checkpoint wearing the frozen extension; symlink it to a .pt "
                    "name before dp --pt show/--finetune will accept it. dp --pt show "
                    "<ckpt.pt> model-branch lists all 31 heads on this box (Domains_Alloy, "
                    "..., Omat24, ...).",
        semantics="finetune: --finetune <ckpt> loads weights (the fitting net is "
                   "re-initialized unless --use-pretrain-script or a matching model "
                   "section is given). --model-branch selects which head's fitting net to "
                   "start from for a multi-task checkpoint (RANDOM = re-init instead). "
                   "This doc names no separate 'resume' flag alongside --finetune — "
                   "--finetune already IS the weights-only path.",
        selector_kind="cli_flag",
        selector_path="--finetune",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="Two former blockers are now handled by the recipe/builder rather than "
             "left to the user: (1) dpdata is pinned in envs/deepmd.yml (dpdata==1.1.0), "
             "and scripts/ft_dataset.py writes the deepmd/npy layout itself when the "
             "module is absent; (2) the .pth -> .pt suffix symlink for "
             "dpa-3.1-3m-ft.pth is performed by ft_run.py's build_deepmd inside the run "
             "directory before dp --pt train is invoked. "
             "Multi-task fine-tuning (keeping pretraining branches alive alongside the "
             "downstream branch, anti-forgetting) uses a multi_input.json with "
             "model.model_dict.<DOWNSTREAM>.finetune_head: <PRE_DATA_branch> plus "
             "training.model_prob branch-sampling weights (verbatim doc shape in report B "
             "§1). The TensorFlow-backend equivalent (dp train input.json --finetune "
             "pretrained.pb) exists but is single-task only and overwrites type_map from "
             "the pretrained model; kept for completeness, not used by this hub "
             "(PyTorch-only registry).",
    ),
    "DPA4": dict(
        status="documented",
        entrypoint="dp --pt train input.json --finetune <ckpt>",
        src="https://docs.deepmodeling.com/projects/deepmd/en/master/train/finetuning.html",
        cmd=["dp --pt train input.json --finetune "
             "\"$OH_MY_MLIP_HOME/models/dpa4/dpa-4.0.1-pro-mptrj.pt\" --use-pretrain-script"],
        config=["{",
                "  \"model\": {\"type_map\": [\"...\"], \"descriptor\": {}, \"fitting_net\": {}},",
                "  \"learning_rate\": {\"type\": \"exp\", \"start_lr\": 0.001, \"stop_lr\": 3e-05, "
                "\"decay_steps\": 5000},",
                "  \"loss\": {\"type\": \"ener\", \"start_pref_e\": 0.2, \"limit_pref_e\": 20, "
                "\"start_pref_f\": 100, \"limit_pref_f\": 60},",
                "  \"training\": {\"training_data\": {\"systems\": [\"../data/data_0\"], \"batch_size\": 1},",
                "                \"numb_steps\": 100000}",
                "}"],
        data_format="Identical to DeePMD/DPA-3.1 — deepmd 'system' npy/hdf5 layout, "
                     "converted from extxyz via dpdata (pinned dpdata==1.1.0 in "
                     "envs/dpa4.yml) or scripts/ft_dataset.py's numpy writer.",
        foundation="--finetune <path> — identical mechanism to DeePMD/DPA-3.1. The local "
                    "checkpoint dpa-4.0.1-pro-mptrj.pt is already .pt (no suffix-symlink "
                    "gotcha) and is SINGLE-TASK: dp --pt show ... model-branch raises "
                    "'The model-branch option requires a multitask model' — omit "
                    "--model-branch entirely.",
        semantics="Same as DeePMD: --finetune loads weights only; --use-pretrain-script "
                   "pulls the model section from the checkpoint, since the local "
                   "single-task checkpoint's architecture is not otherwise known to the "
                   "user's input.json.",
        selector_kind="cli_flag",
        selector_path="--finetune",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="dpdata is pinned in envs/dpa4.yml (dpdata==1.1.0); scripts/ft_dataset.py "
             "also writes the deepmd/npy layout without it. The 'dpa4' descriptor type "
             "only exists in deepmd-kit 3.2.0b0 (this env) — the deepmd env's 3.1.2 "
             "cannot parse a \"type\": \"dpa4\" descriptor, which is why ft_run.py "
             "always runs this family under the dpa4 interpreter (registry env), never "
             "under deepmd's; this is an env-selection fact, not a blocker. "
             "Byte-for-byte the same CLI surface as DeePMD/DPA-3.1 (verified: dp --pt "
             "train --help is character-for-character identical between the two envs). "
             "--model-branch is meaningless here because the shipped checkpoint is "
             "single-task, unlike DPA-3.1-3M-FT's 31-head multi-task checkpoint — do not "
             "pass it. Also needs LD_LIBRARY_PATH=\"/usr/lib/wsl/lib\" (this hub's "
             "env_run for dpa4) at run time.",
    ),
    "UMA": dict(
        status="documented",
        entrypoint="fairchem -c <template>.yaml",
        src="https://fair-chem.github.io/core/common_tasks/fine_tuning.html",
        cmd=["git clone git@github.com:facebookresearch/fairchem.git",
             "pip install -e fairchem/src/packages/fairchem-core[dev]",
             "python src/fairchem/core/scripts/create_uma_finetune_dataset.py "
             "--train-dir <train_dir> --val-dir <val_dir> --output-dir <out> "
             "--uma-task=<task> --regression-tasks <e|ef|efs>   "
             "# NOTE: --regression-tasks is PLURAL and required on the installed "
             "2.19.1 — the doc's singular --regression-task fails as written",
             "fairchem -c <out>/uma_sm_finetune_template.yaml base_model_name=uma-s-1p2 "
             "epochs=2 lr=2e-4 job.run_dir=<run_dir> +job.timestamp_id=<id>",
             "fairchem -c <run_dir>/<id>/checkpoints/final/resume.yaml    "
             "# resume, NOT fine-tune -- restores training state"],
        config=["base_model_name: uma-s-1p1   # this hub's default_version is uma-s-1p2 -- "
                "pass base_model_name=uma-s-1p2 (or --base-model on the dataset script)",
                "max_neighbors: 300",
                "epochs: 1",
                "batch_size: 2",
                "lr: 4e-4",
                "weight_decay: 1e-3",
                "# custom checkpoint instead of a named base model:",
                "model:",
                "  _target_: fairchem.core.units.mlip_unit.mlip_unit.initialize_finetuning_model",
                "  checkpoint_location: /path/to/your/checkpoint.pt"],
        data_format="ASE-LMDB (.aselmdb). Doc, verbatim: 'the only requirement is that you "
                     "have input files that can be read as ASE atoms objects by "
                     "ase.io.read and that they contain energy (forces, stress) in the "
                     "correct format' — so plain extxyz/.traj/.cif in a train dir + a val "
                     "dir feed create_uma_finetune_dataset.py directly, which also computes "
                     "the force RMS normalizer and per-element linear reference and stamps "
                     "them into the data YAML.",
        foundation="model._target_: fairchem.core.units.mlip_unit.mlip_unit."
                    "initialize_finetuning_model + checkpoint_location: /path/to/"
                    "checkpoint.pt for a custom checkpoint, OR the shipped template's "
                    "base_model_name: <name> (resolved via pretrained_mlip."
                    "pretrained_checkpoint_path_from_name) for a named foundation model.",
        semantics="finetune: checkpoint_location / base_model_name loads weights into a "
                   "NEW training run (fresh optimizer/scheduler). Resuming an interrupted "
                   "fine-tune uses the emitted <run>/checkpoints/final/resume.yaml, the "
                   "separate optimizer-state-restoring mechanism — do not conflate the two.",
        selector_kind="config_key",
        selector_path="model.checkpoint_location (or base_model_name)",
        runnable_as_installed=False,
        blockers=["configs/uma/finetune/ is not in the installed fairchem-core 2.19.1 "
                  "wheel — create_uma_finetune_dataset.py hardcodes a RELATIVE "
                  "TEMPLATE_DIR = Path('configs/uma/finetune'), so it only resolves with "
                  "the fairchem repo cloned and cwd at the repo root; running it from an "
                  "arbitrary directory dies on a missing "
                  "configs/uma/finetune/data/uma_conserving_data_task_*.yaml"],
        licence=None,
        note="'While UMA was trained in a multi-task fashion, we ONLY support "
             "fine-tuning on a single UMA task at a time' (doc, verbatim) — --uma-task "
             "selects the task (a per-variant divergence, carried as variant_args in "
             "models.json). DOC-VS-INSTALLED (report B, all three real): (1) "
             "--regression-task (doc) vs --regression-tasks (installed, required, plural) "
             "— folded into cmd above; (2) doc lists uma-task choices omol/odac/oc20/oc22/"
             "oc25/omat/omc but the installed UMATask enum is only ['omol','omat','odac',"
             "'oc20','oc25','omc'] — oc22 is GONE and rejected by argparse choices (see "
             "UMA-s-1p2-OC22 in models.json, which does NOT inherit this family's "
             "runnable_as_installed); (3) the configs/ blocker above. --regression-tasks "
             "in {e, ef, efs} controls which of energy/energy+forces/energy+forces+stress "
             "are trained (gradients for the others remain computable regardless). "
             "Security note from the docs, echoed here: never run a YAML config from an "
             "untrusted source — Hydra instantiates Python objects from the _target_ key.",
    ),
    "fairchemv1": dict(
        status="documented (generic v1 fine-tune path) + code-excavation for eSEN-specific config shape",
        entrypoint="fairchem --mode train --config-yml <yml> --checkpoint <ckpt>",
        src="https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.10.0/docs/"
            "core/common_tasks/fine-tuning/fine-tuning.md",
        cmd=["fairchem --mode train --config-yml config.yml --checkpoint "
             "/path/to/esen_30m_oam.pt --run-dir fine-tuning --identifier ft-esen "
             "--num-gpus 1 --amp",
             "# repo checkout only (doc's own example, {} = python-side interpolation):",
             "# python {fairchem_main()} --mode train --config-yml {yml} --checkpoint "
             "{checkpoint_path} --run-dir fine-tuning --identifier ft-oxides --cpu"],
        config=["# generate_yml_config(checkpoint_path, 'config.yml',",
                "#   delete=['slurm','cmd','logger','task','model_attributes',",
                "#           'optim.loss_force','optim.load_balancing',",
                "#           'dataset','test_dataset','val_dataset'],",
                "#   update={'gpus': 1, 'optim.eval_every': 10, 'optim.max_epochs': 1,",
                "#           'optim.batch_size': 4, 'logger': 'tensorboard',",
                "#           'dataset.train.src': 'train.db', 'dataset.train.format': 'ase_db',",
                "#           'dataset.train.a2g_args.r_energy': True,",
                "#           'dataset.train.a2g_args.r_forces': True, ...})"],
        data_format="ASE db (ase_db format, .db written via ase.db.connect + "
                     "SinglePointCalculator) or LMDB. train_test_val_split(...) splits one "
                     "db into train/test/val. Set a2g_args.r_energy/r_forces (and r_stress "
                     "for eSEN's EFS head) to match the labels actually present.",
        foundation="--checkpoint /path/to/esen_30m_oam.pt (or any v1 checkpoint from "
                    "fairchem/core/models/pretrained_models.json, e.g. eSEN-30M-OAM/"
                    "-OMAT24/-MP). generate_yml_config(checkpoint_path, ...) reads the "
                    "full v1 config bundled inside that checkpoint (state_dict/"
                    "normalizers/elementrefs/config/...) and renders it into a trainable "
                    "YAML.",
        semantics="finetune: --checkpoint loads model weights into a fresh trainer "
                   "(optimizer/scheduler state is not part of the checkpoint-derived "
                   "training YAML). fairchem v1 has no separate '--resume' flag in this "
                   "doc; the same --checkpoint flag pointed at your own in-progress run's "
                   "checkpoint.pt would resume it — treat any FT run here as effectively "
                   "finetune semantics.",
        selector_kind="cli_flag",
        selector_path="--checkpoint",
        runnable_as_installed=False,
        blockers=["fairchem_main() (used by the doc's IPython-cell command) resolves to "
                  "<repo_root>/main.py, which does not exist in the pip-installed "
                  "fairchemv1 env — use the installed fairchem console script instead "
                  "(identical flags), confirmed working via fairchem --help"],
        licence=None,
        note="eSEN-specific excavation (code, not docs): config['trainer'] == "
             "'mlip_trainer' (registered at fairchem.core.models.esen.trainers.trainer; "
             "keep as-is, do NOT delete it via generate_yml_config's delete= list, which "
             "was written for GemNet-OC). config['model'] is a hydra composite "
             "({'name':'hydra','backbone':{'model':'esen_backbone',...},"
             "'heads':{'mptrj':{'module':'esen_mlp_efs_head'}}}) with ONE head named "
             "'mptrj' — head selection means editing model.heads plus the matching "
             "config['outputs']/config['loss_functions'] keys. optim.loss_force and "
             "optim.load_balancing must still be deleted from the generated YAML (the "
             "tutorial's delete list) or the run errors. There is no eSEN training YAML "
             "anywhere in the v1.10.0 repo — every eSEN-specific fact here was excavated "
             "from the installed package and the loaded checkpoint, not from a doc page.",
    ),
    "EquiformerV3": dict(
        status="code-excavation-needed",
        entrypoint="torchrun ... my_main.py --mode train --config-yml <cfg>",
        src="https://github.com/atomicarchitects/equiformer_v3",
        cmd=["git clone https://github.com/atomicarchitects/equiformer_v3.git      "
             "# the pip package ships only fairchem/experimental/{models,trainers} -- "
             "configs/scripts/main.py are repo-only",
             "python experimental/tasks/remove_key_from_checkpoint.py --input-path "
             "<checkpoint.pt> --remove-name energy_block   "
             "# README says --remove-key; the actual typer flag is --remove-name",
             "torchrun --nproc_per_node=<N> my_main.py --num-gpus <N> --num-nodes 1 "
             "--mode train --amp --config-yml <edited-config>.yml --run-dir <log-dir> "
             "--print-every 200 --seed 1 --identifier my_finetune --optim.num_workers=0"],
        config=["optim:",
                "  batch_size: 8",
                "  lr_initial: 0.00005",
                "  max_epochs: 10",
                "  use_denoising_pos: False   # off for fine-tuning; on during pretraining",
                "  load_pretrained_weights: /path/to/checkpoint_no-energy_block.pt"],
        data_format=".aselmdb plus a metadata.npz recording per-structure edge counts for "
                     "load balancing. experimental/datasets/mptrj_convert_json_to_aselmdb.py "
                     "+ create_metadata_num_edges.py --input-dir <dir> (then cp "
                     "metadata_num-edges.npz metadata.npz) are the repo's own converters; "
                     "there is no documented extxyz-to-aselmdb converter in this repo "
                     "specifically — the practical route is fairchem's "
                     "create_finetune_dataset.py (shared with UMA/eSEN) followed by "
                     "create_metadata_num_edges.py. README data-hygiene rule: structures "
                     "with any atom having no neighbor within 6A are removed.",
        foundation="optim.load_pretrained_weights: /path/to/checkpoint_no-energy_block.pt "
                    "in the training YAML — NOT a --checkpoint CLI flag. The README's own "
                    "pretraining recipe first strips the energy_block key from the "
                    "released checkpoint (remove_key_from_checkpoint.py) before pointing "
                    "load_pretrained_weights at the stripped file, which is how a new "
                    "energy head gets randomly initialized during gradient fine-tuning.",
        semantics="finetune: optim.load_pretrained_weights does a WEIGHTS-ONLY, "
                   "non-strict-on-missing-keys, strict-on-matching-shape load "
                   "(equiformer_v3_dens_trainer.py:1103) — optimizer/EMA/scheduler state "
                   "is never touched. There is no --checkpoint-style resume flag in the "
                   "shipped launch script; resume would mean re-running with the same "
                   "run-dir and relying on the trainer's own checkpointing, a separate "
                   "mechanism not documented here.",
        selector_kind="config_key",
        selector_path="optim.load_pretrained_weights",
        runnable_as_installed=False,
        blockers=["the installed equiformer_v3 env (fairchem-core "
                  "0.1.dev10+ga7300c58d) contains only fairchem/core/ and "
                  "fairchem/experimental/{models,trainers} — no configs/, "
                  "experimental/scripts/, experimental/tasks/, experimental/datasets/, "
                  "main.py, or my_main.py, and no v3-style training console script (the "
                  "env's fairchem binary is the v1 CLI); every command above needs the "
                  "git checkout"],
        licence=None,
        note="No documented procedure exists for fine-tuning the released HuggingFace "
             "checkpoint on user data — the README documents the AUTHORS' pretraining and "
             "gradient-fine-tuning runs with exact commands, but the mechanism that makes "
             "a user fine-tune possible (optim.load_pretrained_weights) is an "
             "undocumented YAML key, confirmed only by reading the installed trainer "
             "source. use_denoising_pos: False is the fine-tune setting (DeNS "
             "regularization is on during pretraining, off during gradient FT).",
    ),
    "ORB": dict(
        status="documented",
        entrypoint="python finetune.py",
        src="https://github.com/orbital-materials/orb-models/blob/main/FINETUNING_GUIDE.md",
        cmd=["python finetune.py --data_path /path/to/dataset.db --dataset my_dataset "
             "--base_model orb_v3_conservative_inf_omat --batch_size 32 --num_steps 100 "
             "--max_epochs 50 --lr 3e-4 --save_every_x_epochs 5 --checkpoint_path ./ckpts"],
        config=None,
        data_format="ASE sqlite .db (row.energy/row.forces/row.stress read via ase.db). "
                     "AseSqliteDataset.__getitem__ does db.get(idx+1) — rows must be "
                     "1-INDEXED and DENSE. Convert any ASE-readable trajectory with a "
                     "plain read() + SinglePointCalculator(..., "
                     "stress=atoms.get_stress(voigt=True)) + db.write(atoms) loop (stress "
                     "required unless model.has_stress is False).",
        foundation="--base_model <name> selects the pretrained checkpoint by registry key "
                    "(installed 0.5.5 choices: orb_v3_conservative_inf_omat, "
                    "orb_v3_conservative_20_omat, orb_v3_direct_inf_omat, "
                    "orb_v3_direct_20_omat, orb_v2 — NOT orbmol_v2/other main-branch-only "
                    "names).",
        semantics="finetune: finetune.py always starts from --base_model's pretrained "
                   "weights with a fresh optimizer — the script has no resume/"
                   "continue-training flag at all; --checkpoint_path is only the OUTPUT "
                   "directory for periodic saves (--save_every_x_epochs), not an input.",
        selector_kind="cli_flag",
        selector_path="--base_model",
        runnable_as_installed=False,
        blockers=["finetune.py is not shipped in the orb-models wheel — must curl it "
                  "from the repo at the tag matching the installed version (v0.5.5): "
                  "curl -O https://raw.githubusercontent.com/orbital-materials/"
                  "orb-models/v0.5.5/finetune.py",
                  "wandb is a hard top-level import in v0.5.5's finetune.py (raises "
                  "ImportError immediately) and is not installed in the orb env — pip "
                  "install wandb, and since v0.5.5's --wandb defaults True with "
                  "action='store_true' (cannot be disabled from the CLI), also set "
                  "WANDB_MODE=offline or log in"],
        licence=None,
        note="Version pinning matters: the main-branch docs describe orbmol_v2, "
             "loss_weights=, train_reference_energies=, and a (model, atoms_adapter) "
             "tuple return from pretrained.* — NONE of these exist in the installed "
             "0.5.5 (pretrained.orb_v3_conservative_inf_omat returns a single model "
             "object; passing main-branch-only kwargs TypeErrors). No early stopping "
             "(use --save_every_x_epochs); the LR schedule is hardcoded to OneCycleLR "
             "over max_epochs*num_steps.",
    ),
    "MatterSim": dict(
        status="documented",
        entrypoint="torchrun ... finetune_mattersim.py",
        src="https://github.com/microsoft/mattersim/blob/main/docs/user_guide/finetune.rst",
        cmd=["torchrun --nproc_per_node=1 "
             "<env>/lib/python3.10/site-packages/mattersim/training/"
             "finetune_mattersim.py --load_model_path mattersim-v1.0.0-5m "
             "--train_data_path train.xyz --valid_data_path valid.xyz "
             "--save_path ./results --save_checkpoint --epochs 100 --batch_size 16 "
             "--lr 2e-4 --include_forces --include_stresses --device cuda"],
        config=None,
        data_format="Anything ase.io.read handles, multi-frame (AtomsAdaptor.from_file), "
                     "or a pickled list of Atoms (.pkl). Plain extxyz with attached "
                     "energies/forces/stress in the ASE calculator is the zero-work path. "
                     "The script itself divides get_stress(voigt=False) by ase.units.GPa "
                     "— supply plain ASE stress (eV/A^3), do not pre-convert.",
        foundation="--load_model_path accepts the literal names mattersim-v1.0.0-1m / "
                    "mattersim-v1.0.0-5m (case-insensitive, auto-downloaded) or a path to "
                    "a .pth checkpoint, via Potential.from_checkpoint.",
        semantics="finetune: Potential.from_checkpoint(..., load_training_state=False) "
                   "(the script's default path) loads weights only. "
                   "Potential.from_checkpoint(..., load_training_state=True) is the "
                   "SEPARATE resume mechanism that restores optimizer state — same entry "
                   "point, different kwarg.",
        selector_kind="cli_flag",
        selector_path="--load_model_path",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="ft_run.py builder (host-inspected mattersim 1.2.1 sources): `torchrun "
             "--nproc_per_node=1 --standalone -m mattersim.training.finetune_mattersim "
             "--load_model_path <name> --train_data_path/--valid_data_path <extxyz> "
             "--save_path <out>/results --save_checkpoint --epochs N --batch_size B "
             "--lr 2e-4 --include_forces --device cuda|cpu`; --save_checkpoint "
             "(BooleanOptionalAction, default False) is what makes potential.save_model "
             "write <out>/results/best_model.pth and last_model.pth. Reload: "
             "Potential.from_checkpoint(load_path=<best_model.pth>, "
             "load_training_state=False) + MatterSimCalculator(potential=...). "
             "MUST be launched under torchrun even for a single GPU/CPU run — the "
             "script reads os.environ['LOCAL_RANK'] unconditionally at module load and "
             "calls torch.distributed.init_process_group; plain python "
             "finetune_mattersim.py dies with KeyError. --save_checkpoint defaults to "
             "False (nothing is written to --save_path without it). --include_stresses "
             "defaults False while --include_forces defaults True. --re_normalize "
             "recomputes scale/shift from the new data when the fine-tune labels come "
             "from a different functional/reference. The docs' own finetune.html page "
             "404s; the live source is docs/user_guide/finetune.rst.",
    ),
    "CHGNet": dict(
        status="documented",
        entrypoint="chgnet.trainer.Trainer (python API)",
        src="https://github.com/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb",
        cmd=["from chgnet.model import CHGNet",
             "from chgnet.trainer import Trainer",
             "chgnet = CHGNet.load()   # model_name='0.3.0' by default",
             "trainer = Trainer(model=chgnet, targets='efsm', optimizer='Adam', "
             "scheduler='CosLR', criterion='MSE', epochs=5, learning_rate=1e-2, "
             "use_device='cuda')",
             "trainer.train(train_loader, val_loader, test_loader)"],
        config=None,
        data_format="pymatgen Structure objects + energies_per_atom (eV/ATOM, not total "
                     "energy) + forces (eV/A) + optional stresses/magmoms, via "
                     "StructureData. From ASE: AseAtomsAdaptor.get_structure(atoms) + "
                     "atoms.get_potential_energy()/len(atoms). Stress must be VASP raw "
                     "kBar = ASE eV/A^3 * 1602.1766208 (with a documented sign flip, see "
                     "note). parse_vasp_dir(file_root=...) / StructureData.from_vasp(...) "
                     "build a dataset directly from a VASP run directory.",
        foundation="CHGNet.load(model_name='0.3.0') loads the bundled pretrained model; "
                    "pass it directly as Trainer(model=chgnet, ...) — there is no "
                    "separate 'foundation checkpoint path' argument, only the model_name "
                    "registry key.",
        semantics="finetune: CHGNet.load() then Trainer(model=chgnet, ...).train(...) is "
                   "inherently weights-only fine-tuning — there is no optimizer-state "
                   "resume path documented for the pretrained model; Trainer.train's "
                   "starting_epoch kwarg is for a from-scratch run's own checkpoints, not "
                   "for restoring the foundation model's training state.",
        selector_kind="cli_flag",
        selector_path="CHGNet.load(model_name=...)",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="ft_run.py builder (host-inspected chgnet 0.4.0 sources): chgnet ships no "
             "training CLI, so ft_run.py emits finetune_chgnet.py and runs it with the "
             "env python: ase reads the canonical extxyz, pymatgen's AseAtomsAdaptor "
             "builds Structures, energies are divided by len(atoms) (eV/atom), "
             "StructureData + get_loader(batch_size) feed "
             "Trainer(model=CHGNet.load(model_name='0.3.0'), targets='ef', "
             "optimizer='Adam', scheduler='CosLR', criterion='MSE', epochs, "
             "learning_rate=1e-3, use_device).train(train, val, save_dir=<out>/chgnet_ft). "
             "Trainer.save_checkpoint writes epoch<N>_*.pth.tar every epoch and copies the "
             "best-val-energy epoch to bestE_epoch<N>_*.pth.tar; reload: "
             "CHGNet.from_file(<bestE_*.pth.tar>) + CHGNetCalculator(model=...). "
             "Energy label MUST be eV/atom, not total energy (the notebook variable is "
             "literally energies_per_atom). Stress units are VASP raw kBar, with a "
             "documented SIGN FLIP (verbatim): '...the -10 unit conversion modifies it to "
             "be kbar in VASP raw unit... If you're using stress labels from VASP, you "
             "don't need to do any unit conversions.' GGA/GGA+U-compatible data should go "
             "through MaterialsProject2020Compatibility first; non-MP functionals "
             "(r2SCAN, QE, Gaussian, ...) should refit the isolated-atom reference via "
             "Trainer(..., train_composition_model=True) or AtomRef.fit(...) directly. "
             "Selective layer freezing (chgnet.atom_embedding, bond_embedding, ..., "
             "atom_conv_layers[:-1], ...) is shown as an optional step before "
             "constructing the Trainer.",
    ),
    "AlphaNet": dict(
        status="code-excavation-needed",
        entrypoint="alpha-train <config.json> --resume --ckpt_path <ckpt>",
        src="https://github.com/zmyybc/AlphaNet",
        cmd=["alpha-train mp.json --num_devices 1 --resume --ckpt_path "
             "/path/to/checkpoint.ckpt   # nearest DOCUMENTED command -- see semantics: "
             "this is a Lightning RESUME, not a fine-tune"],
        config=["{",
                "  \"data\": {\"root\": \"dataset/\", \"dataset_name\": \"t1\", "
                "\"target\": \"energy_force\"},",
                "  \"model\": {\"name\": \"Alphanet\", \"num_layers\": 4, "
                "\"hidden_channels\": 176, \"cutoff\": 5, \"num_radial\": 8,",
                "             \"compute_forces\": true, \"compute_stress\": true, "
                "\"use_pbc\": true},",
                "  \"train\": {\"epochs\": 200, \"batch_size\": 24, \"lr\": 0.0001, "
                "\"energy_coef\": 4.0, \"force_coef\": 100.0, \"device\": \"cuda\"}",
                "}",
                "# the model block must match pretrained/MPtrj/mp.json or "
                "pretrained/OMA/oma.json verbatim to fine-tune those foundation weights"],
        data_format="Pickled dict {'E','F','R','z','cell','natoms','stress'} under "
                     "dataset/<name>/raw/*.pickle, consumed by CustomPickleDataset (a PyG "
                     "InMemoryDataset). Repo converters: scripts/dp2pic_batch.py (from "
                     "deepmd) and scripts/xyz2pic.py (from extxyz) — the latter REQUIRES "
                     "atoms.info['virial'] (frames without it are skipped) and reads "
                     "atoms.arrays['force'], not get_forces().",
        foundation="No documented or coded 'load pretrained weights, keep training "
                    "options' path exists. The released foundation weights (Figshare "
                    "53851133 MPtrj / 53851139 OMA) are bare state_dict files loaded only "
                    "by the ASE calculator (AlphaNetCalculator: "
                    "model.load_state_dict(torch.load(ckpt), strict=False)); "
                    "alpha-train's only checkpoint-consuming argument, --ckpt_path, is a "
                    "Lightning trainer.fit(ckpt_path=...) RESUME that expects the "
                    "Lightning envelope (state_dict + optimizer_states + epoch), not a "
                    "bare state_dict.",
        semantics="resume only, as shipped: alpha-train --resume --ckpt_path "
                   "<lightning.ckpt> restores full Lightning training state (optimizer, "
                   "LR scheduler, epoch counter) from a Lightning checkpoint — it CANNOT "
                   "consume the released bare-state_dict foundation weights. There is no "
                   "weights-only fine-tune path in the installed code; run_training also "
                   "unconditionally overwrites the model's scale/shift (config.a/"
                   "config.b) from the new dataset's statistics before construction, "
                   "discarding whatever the pretrained model used, and the dataset root "
                   "is hardcoded to ./dataset/ regardless of config.data.root.",
        selector_kind="cli_flag",
        selector_path="--ckpt_path (resume only; no fine-tune selector exists)",
        runnable_as_installed=False,
        blockers=["click and pyfiglet are missing from the AlphaNet env — alpha-train "
                  "--help raises ModuleNotFoundError: No module named 'click' "
                  "immediately; pip install click pyfiglet",
                  "no code path connects the released bare state_dict foundation weights "
                  "to a fine-tune run — doing so requires writing a custom script that "
                  "builds AlphaNetWrapper(config), loads the state_dict with "
                  "strict=False (mirroring infer/calc.py), pins config.a/config.b to the "
                  "pretrained values, and wraps it in alphanet.mul_trainer.Trainer for a "
                  "pl.Trainer — i.e. forking run_training, not calling it"],
        licence=None,
        note="Loss-weight guidance from alphanet/config.py's own docstring: "
             "energy:force:stress around 4:100:100 for systems under ~300 atoms, or a "
             "dynamic ramp starting at 0.01:100:100 with lr 5e-4 for larger/high-"
             "per-atom-energy systems. To fine-tune the shipped foundation checkpoints "
             "the architecture block must match pretrained/MPtrj/mp.json or "
             "pretrained/OMA/oma.json verbatim (num_layers, hidden_channels, cutoff, "
             "num_radial, compute_forces/compute_stress, use_pbc, ...). TrainConfig "
             "defaults force_coef/stress_coef to 0.0 and AlphaConfig defaults "
             "compute_forces to False — both must be set explicitly for an MLIP run.",
    ),
    "Eqnorm": dict(
        status="not-supported",
        entrypoint=None,
        src="https://github.com/yzchen08/eqnorm",
        cmd=None,
        config=None,
        data_format=None,
        foundation=None,
        semantics=None,
        selector_kind=None,
        selector_path=None,
        runnable_as_installed=False,
        blockers=["zero training code exists in the released package or repo — no "
                  "train.py, no CLI, no dataset module, no optimizer/loss code; "
                  "grep -rl 'def train|torch.optim|loss' over the installed 0.1.0 "
                  "package returns nothing"],
        licence=None,
        note="The README states the shipped eqnorm-mptrj weights were 'trained on the "
             "MPtrj dataset' but gives no instructions for a user to train or "
             "fine-tune; the eqnorm-*.py files are model-ARCHITECTURE definitions and "
             "the yaml files are inference-side settings only. The full repo tree "
             "(GitHub trees API, main) confirmed no scripts/ or examples/ directory. "
             "Writing fine-tuning support would mean building a training loop (loss, "
             "optimizer, dataloader, graph construction) from scratch against the model "
             "classes — beyond 'excavation'.",
        evidence=["https://github.com/yzchen08/eqnorm/blob/main/README.md",
                  "https://github.com/yzchen08/eqnorm (full repo tree, GitHub trees API)"],
    ),
    "PET": dict(
        status="documented",
        entrypoint="mtt train options-ft.yaml -o model-ft.pt",
        src="https://docs.metatensor.org/metatrain/latest/concepts/fine-tuning.html",
        cmd=["mtt train options-ft.yaml -o model-ft.pt",
             "mtt eval model-ft.pt options-ft-eval.yaml -o output-ft.xyz"],
        config=["architecture:",
                "  name: pet",
                "  training:",
                "    batch_size: 8",
                "    num_epochs: 10",
                "    learning_rate: 1e-3",
                "    finetune:",
                "      method: full",
                "      read_from: pet-mad-v1.1.0.ckpt",
                "training_set:",
                "  systems: {read_from: ethanol_reduced_100.xyz, reader: ase, "
                "length_unit: angstrom}",
                "  targets:",
                "    energy/finetune:",
                "      quantity: energy",
                "      read_from: ethanol_reduced_100.xyz",
                "      reader: ase",
                "      key: energy",
                "      unit: eV",
                "      forces: {read_from: ethanol_reduced_100.xyz, reader: ase, key: forces}"],
        data_format="Direct — no conversion step. training_set.systems.read_from: <file> "
                     "+ reader: ase, and targets.<name>.read_from/key for energy and "
                     "forces (units declared explicitly, e.g. length_unit: angstrom, "
                     "unit: eV).",
        foundation="architecture.training.finetune.read_from: <path>.ckpt (a downloaded "
                    "HF checkpoint, e.g. lab-cosmo/pet-mad's pet-mad-v1.1.0.ckpt, or any "
                    "local .ckpt).",
        semantics="finetune: the finetune: block RESETS optimizer/scheduler state (docs, "
                   "verbatim: 'In contrast to the training continuation, the optimizer "
                   "and scheduler state will be reset'; code: metatrain/pet/trainer.py:"
                   "341,350 skip loading optimizer/scheduler state when is_finetune). "
                   "mtt train --restart <ckpt> (or --restart auto) is the SEPARATE "
                   "training-continuation flag that DOES keep optimizer/scheduler state "
                   "— never combine the two.",
        selector_kind="config_key",
        selector_path="architecture.training.finetune.read_from",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="ft_run.py builder (host-inspected metatrain 2026.1 sources): `mtt train "
             "options.yaml -o <out>/model-ft.pt -e <out>/extensions` with "
             "architecture.training.finetune {read_from: "
             "models/pet/pet-oam-xl-v1.0.0.ckpt (the training checkpoint, NOT the "
             "exported .pt the single-point line uses), method: full}; training_set / "
             "validation_set as {systems: {read_from, reader: ase, length_unit: "
             "angstrom}, targets: {energy: {quantity: energy, key: energy, unit: eV, "
             "forces: {key: forces}}}} (utils/data/readers/ase.py copies a "
             "SinglePointCalculator's results into info['energy']/arrays['forces'] "
             "before reading those keys); test_set: 0.0. cli/train.py writes "
             "<out>/model-ft.ckpt and the exported <out>/model-ft.pt unconditionally "
             "after training; reload: MetatomicCalculator(<model-ft.pt>, device=...). "
             "Three fine-tune methods share the read_from key: method: full (all "
             "weights trainable), method: heads (backbone frozen, config.head_modules/"
             "last_layer_modules list what stays trainable), method: lora "
             "(config.target_modules/rank/alpha; the concept page has no LoRA YAML "
             "example — this shape is excavated from metatrain/pet/modules/"
             "finetuning.py and its own test file). Fine-tuning support is PET/FlashMD/"
             "FlashMDSymplectic/SPACE only, not GAP/SOAP-BPNN. A full or lora run "
             "creates a NEW head named energy/<variantname> and DROPS every target not "
             "in that run's training set; a heads run keeps them. CRITICAL FOR "
             "OH-MY-MLIP: the resulting model does not answer to the plain 'energy' "
             "output — MetatomicCalculator(..., variants={'energy': '<variantname>'}) is "
             "required at calculator construction (and pair_style metatomic ... variant "
             "<name> in LAMMPS), which differs from the hub's stock PET-OAM-XL "
             "calculator construct.",
    ),
    "EquFlash": dict(
        status="documented (config-level)",
        entrypoint="python main.py --mode train --config-yml <cfg>",
        src="https://github.com/SamsungDS/GGNN",
        cmd=["python -m GGNN.main --mode train --config-yml equflash-ft.yml --num-gpus 1  "
             "# installed-package equivalent; repo root ships "
             "python main.py --mode train --config-yml configs/equflash/equflash.yml"],
        config=["dataset:",
                "  train: {format: ase_db, src: [/path/to/train/dataset], "
                "a2g_args: {r_energy: True, r_forces: True, r_stress: True}}",
                "  val:   {format: ase_db, src: [/path/to/valid/dataset], "
                "a2g_args: {r_energy: True, r_forces: True, r_stress: True}}",
                "optim: {batch_size: 16, lr_initial: 0.001, max_epochs: 1000, "
                "optimizer: AdamW}",
                "task:",
                "  finetune: {checkpoint: /path/to/equflash-OAM.pt, reset_head: true}",
                "  strict_load: false"],
        data_format="format: ase_db (fairchem-v1 AseDBDataset) is the safe/tested path "
                     "— any format ase.db.connect accepts (.db, .aselmdb), built with a "
                     "plain ase.io.read(...) + ase.db.connect(...).write(atoms) loop. "
                     "format: ase_read_multi reads a directory of extxyz files directly "
                     "without a conversion step (GGNN's own ase_arrays_read_multi is also "
                     "registered). a2g_args.{r_energy,r_forces,r_stress} selects which "
                     "labels are read.",
        foundation="task.finetune.checkpoint: /path/to/checkpoint.pt in the YAML "
                    "(config-level; the README itself never uses the word 'fine-tune'). "
                    "The FULL architecture config matching the checkpoint (including the "
                    "87-entry model.type_map) must be carried over verbatim or the "
                    "state-dict match silently mis-maps species.",
        semantics="finetune: task.finetune.checkpoint loads a WEIGHTS-ONLY, key-matched "
                   "state dict (GGNN/trainer/trainer.py:803-853, match_state_dict + "
                   "optional task.finetune.reset_head to drop the output layer, "
                   "task.strict_load/reset_shift/reset_scale/reset_stat as additional "
                   "undocumented knobs). --checkpoint on the command line is the "
                   "SEPARATE fairchem-v1 resume path (restores epoch/step/optimizer "
                   "state) — do not confuse the CLI --checkpoint flag with the YAML "
                   "task.finetune.checkpoint key; they are different mechanisms with a "
                   "similar name.",
        selector_kind="config_key",
        selector_path="task.finetune.checkpoint",
        runnable_as_installed=False,
        blockers=["main.py and configs/ are repo-only — not present in the installed "
                  "GGNN 0.1 package; use python -m GGNN.main as the installed "
                  "equivalent (same flags, from fairchem.core.common.flags)"],
        licence="CC-BY-NC-SA-4.0",
        note="LICENCE HAZARD: GGNN/LICENSE.md is CC BY-NC-SA 4.0 (NonCommercial + "
             "ShareAlike) — a fine-tuned EquFlash model is a derivative work and "
             "inherits both restrictions; third-party code inside the repo "
             "(fairchem-core, NequIP, SevenNet) is separately MIT, but the GGNN package "
             "as a whole is not. Fairchem-v1 vintage API throughout (--config-yml, "
             "registry, submitit), not fairchem v2. Released checkpoints to fine-tune "
             "from: equflash-OAM, equflash-OMat24, equflashv2-OAM, equflashv2-OMat24 "
             "(README table, figshare URLs).",
    ),
    "MatRIS": dict(
        status="not-supported",
        entrypoint=None,
        src="https://github.com/HPC-AI-Team/MatRIS",
        cmd=None,
        config=None,
        data_format=None,
        foundation=None,
        semantics=None,
        selector_kind=None,
        selector_path=None,
        runnable_as_installed=False,
        blockers=["no trainer, loss module, dataset/dataloader, optimizer config, "
                  "train.py, or training YAML exists anywhere in the public repo (full "
                  "recursive GitHub tree listing, 40 paths, confirmed) or in the "
                  "installed matris 0.0.0 package — applications/ contains only base.py "
                  "(calculator), relax.py (StructOptimizer) and md.py "
                  "(MolecularDynamics)"],
        licence=None,
        note="The README states matris_10m_oam was itself 'trained on the OMat24 "
             "dataset, and finetuned on sAlex+Mptrj dataset' — i.e. the authors "
             "fine-tuned internally with code they did not release. Fine-tuning MatRIS "
             "would require writing a trainer from scratch against matris.model.model; "
             "open repo issues concern checkpoint URLs and a torch-sim interface, none "
             "provide training code. Licence is BSD-3-Clause (permissive) — not the "
             "blocker here.",
        evidence=["https://github.com/HPC-AI-Team/MatRIS",
                  "https://arxiv.org/html/2603.02002v1"],
    ),
    "TACE": dict(
        status="documented",
        entrypoint="tace-finetune -m <ckpt> -> tace-train -cn tace -> tace-convert",
        src="https://tace.readthedocs.io/en/latest/guide/finetune.html",
        cmd=["tace-finetune -m TACE-OMat24-7M.pt        # generates finetune_config.yaml "
             "(LoRA + full parameter freeze, by default)",
             "rm -f finetune_config.yaml                # omit this step to keep the "
             "generated LoRA config; delete it to force FULL-parameter fine-tuning",
             "tace-train -cn tace                        # edit tace.yaml first: "
             "finetune_from_model: /path/to/TACE-OMat24-7M.pt",
             "tace-convert -m checkpoints_epoch/last.ckpt     # merges LoRA weights into "
             "the base model -- see blockers: omit --type, its documented value is broken"],
        config=["resume_from_model: null",
                "finetune_from_model: /path/to/TACE-OMat24-7M.pt",
                "finetune: null   # equals ./finetune_config.yaml, auto read/write directly here",
                "dataset: {type: ase, train_file: ../data/BaTiO3.xyz, valid_ratio: 0.1}",
                "optimizer: {_target_: torch.optim.AdamW, lr: 1e-3}   # lower for fine-tuning",
                "model: {config: ...}   # ignored when finetune_from_model is set -- "
                "taken from the checkpoint"],
        data_format="Best of the 20: no conversion needed. dataset.type: ase reads any "
                     "file ase.io.read handles directly, or ase-db (.db / .aselmdb). "
                     "Property keys remappable under dataset.keys (~28 keys: energy_key, "
                     "forces_key, stress_key, ...). storage_mode: lmdb for large "
                     "datasets.",
        foundation="finetune_from_model: /path/to/<ckpt>.{ckpt,pt,pth} in tace.yaml. On a "
                    "fine-tune run the dataset statistics AND the entire model "
                    "architecture block are read from the checkpoint (model.config in "
                    "the YAML is ignored), not recomputed.",
        semantics="finetune: finetune_from_model loads weights only, selecting between "
                   "two strategies by whether finetune_config.yaml exists in the working "
                   "directory: absent -> full-parameter fine-tuning; present -> LoRA + "
                   "parameter freezing (tace-finetune -m <ckpt> auto-generates this file: "
                   "LoRA on element_embedding/radial_mlp/interaction/product/readout at "
                   "r=4,alpha=8, plus freeze: {<every named parameter>: true}). "
                   "resume_from_model: <ckpt>.ckpt is the SEPARATE Lightning-resume "
                   "mechanism (restores full training state) and only accepts .ckpt, "
                   "unlike finetune_from_model's .ckpt/.pt/.pth.",
        selector_kind="config_key",
        selector_path="finetune_from_model",
        runnable_as_installed=True,
        blockers=[],
        licence=None,
        note="ft_run.py builder (host-inspected tace 0.2.0 sources + upstream "
             "example/train/tace.yaml key layout): writes tace.yaml in <out> and runs "
             "`tace-train -cn tace` from there (hydra config_path is the cwd); a "
             "prestage step (tace_prestage.py) resolves the foundation file through "
             "tace.foundations.tace_foundations['<version>'] (HF download into "
             "~/.cache/tace on first use; TACE-OAM-L.pt already present on this host) "
             "and writes it into finetune_from_model. finetune: null and no "
             "finetune_config.yaml in <out> => FULL fine-tune (train.py logs a warning "
             "and skips LoRA/freezing), so the tace-finetune/tace-convert LoRA steps are "
             "not used. Hard-required keys carried: misc.global_seed / "
             "misc.LossSkipController, trainer.precision, dataset.train_dataloader / "
             "valid_dataloader (torch_geometric DataLoader), optimizer, scheduler, "
             "loss (NormalLoss over [energy, forces]), model.config.fidelity (read "
             "before the checkpoint's own model config replaces it), and one "
             "ModelCheckpoint callback (dirpath checkpoints_epoch, save_last) -> "
             "<out>/checkpoints_epoch/last.ckpt. Reload: TACEAseCalc(model=<last.ckpt>, "
             "fidelity_idx=0, target_property=[energy, forces]). "
             "REAL BUG in the installed TACE 0.2.0: tace-convert's --type flag defaults "
             "to 'merge_lora' but ALLOWED_TYPE only contains 'merged_lora' — the "
             "documented --type merge_lora is rejected by argparse choices, and passing "
             "--type merged_lora instead reaches a body check that only accepts the "
             "literal 'merge_lora' and raises ValueError. Workaround: omit -t/--type "
             "entirely (argparse does not validate its own default against choices, so "
             "the broken default reaches the merge branch and works) — worth reporting "
             "upstream. tace-train is Hydra-driven with config_path=Path.cwd() and MUST "
             "be run from the directory containing the config. Replay data (mixing "
             "foundation-model training data back in) is explicitly NOT implemented "
             "during fine-tuning. Full-parameter fine-tuning is discouraged by the docs "
             "for small datasets (overfitting + catastrophic forgetting) in favor of the "
             "LoRA default. Licence: MIT, no constraints.",
    ),
}
