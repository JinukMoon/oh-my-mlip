# Install models

Each framework is installed into its own conda env under `envs/<env>/`.
Conda (or mamba) must be on `PATH`.

## Ask your LLM

```text
Install MACE and SevenNet and check they run on my GPU.
```

The agent installs each env, compiles what your GPU needs, and reports a model
only once it has computed energy and forces there.

## One or several models

```bash
source env.sh
./install.sh MACE                      # a framework name (MACE) or an env name (mace)
./install.sh MACE SevenNet ORB         # several at once
./install.sh --dry-run MACE            # print the plan, install nothing
./install.sh --status MACE             # ready / partial / broken / not installed
```

`install.sh` runs the same steps every time: create the env from
`envs/<env>.yml` (or `envs/<env>.build.sh` where one conda solve is not
enough), install catbench, prepare weights, compile what must be compiled for
your GPU (NequIP with OpenEquivariance, Allegro with CuEquivariance, the D3
kernel), and mark the env ready. An interrupted install is verified and
adopted or rebuilt, never duplicated.

## Check that it works

```bash
python scripts/setup_verify.py MACE-MPA-0 --json
```

The model must compute energy and forces on the GPU; a clean exit of
`install.sh` alone does not count.

## Replay a verified build exactly

Every verified build is recorded in `envs/locks/<env>.conda.txt` and
`envs/locks/<env>.pip.txt`. To install exactly that package set, with no
dependency resolution:

```bash
OMM_USE_LOCK=1 ./install.sh MACE
```

## Prebuilt envs

MACE and SevenNet are also published as prebuilt, relocatable envs on the
Hugging Face Hub (pinned revision and sha256). `oh_my_mlip.fetch.fetch_env("MACE")`
downloads, checks and unpacks one instead of building from the recipe.

## Use an env you already have

```bash
python3 scripts/adopt_env.py MACE ~/miniconda3/envs/MACE   # verified before it is trusted
python3 scripts/adopt_env.py --list
```

## Gated models

UMA and eSEN weights are gated. Accept the license on the model page with your
Hugging Face account, then run `hf auth login` before installing.
See [Gated models](../gated_models.md) and [Hugging Face token](../hf_token.md).

## Prove a clean install

`scripts/setup_sweep.py --fresh-root` builds each env in a disposable root,
verifies every variant on the GPU, keeps the evidence and removes the root:

```bash
python3 scripts/setup_sweep.py sweep --fresh-root --targets mace --peak-gib 20
```

Full procedure: [`recipes/setup.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/setup.md).
