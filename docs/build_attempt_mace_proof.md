# Real-build proof — MACE (deferred-compute lifeline test)

> Run on a real GPU box: WSL + NVIDIA RTX 4060 Ti (sm89, same arch family as the validated
> L40S), CUDA 12.8, conda 25.7. This is the lifeline test: clone -> build -> download ->
> verify -> run. Token-safe (no HF token in this report).

## ✅ Weight integrity (the moat) — CONFIRMED for MACE-MPA-0

| | sha256 | size (bytes) |
|---|---|---|
| Downloaded `mace_mp(model="medium-mpa-0")` | `75428afe…604fb638` | 79462305 |
| /TGM validated `mace-mpa-0.model` | `75428afe…604fb638` | 79462305 |

**BYTE-IDENTICAL.** The official by-name download oh-my-mlip uses == the exact checkpoint
the internal hub validated against. `scripts/verify_weights_integrity.py` records this
fingerprint so any future drift of the official source is caught automatically.

## Failure list (real findings; the iterate input)

1. **[CRITICAL — FIXED] `--no-deps` line silently breaks `conda env create`.** Every recipe
   ended its pip block with a bare `- --no-deps` line. Under `conda env create -f`, pip
   reads the pip block as a requirements file where a standalone `--no-deps` line errors
   (`Invalid requirement: --no-deps`) → the WHOLE pip block fails → torch/framework/catbench
   are NOT installed (conda exits 0 anyway). So NO recipe actually built. Fixed: removed
   from all 20 recipes; catbench installs in an install.sh post-create step; lint guard
   prevents regression.

2. **[HIGH — FIXED] catbench must install WITH deps, not `--no-deps`.** `--no-deps catbench`
   drops `requests`/`xlsxwriter`/… so `from catbench.adsorption import AdsorptionCalculation`
   fails (`No module named 'requests'`). Empirically, `pip install catbench==1.1.2` (with
   deps) does NOT downgrade the pinned `torch==2.7.1+cu126` — the `--no-deps` rationale did
   not hold for catbench 1.1.2. Fixed: install.sh installs catbench WITH deps after torch.

3. **[BLOCKER — owner input] BackSingle2018 dataset needs the new preprocessed format.**
   catbench 1.1.2's `AdsorptionCalculation(...).run()` expects `raw_data/<tag>_adsorption.json`,
   but the available `BackSingle2018.json` (jumoon's 2024 submission) is the OLD format for the
   removed `json2pkl`+`execute_benchmark` API. The current-hub preprocessed reference under
   `/home/jaeryomadang/catbench_test/BackSingle2018/` is **permission-denied** to jumoon. To
   run the catbench *functional* equivalence, need (a) read access to that reference, or
   (b) regenerate `BackSingle2018_adsorption.json` via catbench `cathub_preprocessing`.

4. **[INFRA] conda-pack absent in the /TGM conda** (`conda pack` → invalid choice). A strict
   same-GPU `time_per_step` comparison needs the validated /TGM env relocated here — install
   conda-pack first, or direct env-dir rsync + prefix fix.

## What this proves / doesn't
- ✅ MACE recipe builds on a real GPU box (after the `--no-deps` fixes): torch 2.7.1+cu126
  CUDA-available, mace 0.3.15, catbench 1.1.2.
- ✅ Downloaded weight == validated checkpoint (byte-identical).
- ⏳ catbench *functional* equivalence (energies/MAE/anomaly + time_per_step vs /TGM) NOT yet
  run — blocked on the preprocessed dataset (finding 3). Energies are GPU-independent (very
  likely to match given the sha256 match); time_per_step needs both envs on the same GPU
  (finding 4).

## Next iterate cycle
- Resolve dataset access (3) → run catbench BackSingle2018 with the built MACE env → compare
  energies/MAE/anomaly to the validated reference.
- Install conda-pack (4) → relocate the /TGM MACE env to the same GPU → compare time_per_step.
- Repeat build + weight-fingerprint + catbench for the remaining 30 models (subset tier),
  deleting each env after to reclaim disk.
