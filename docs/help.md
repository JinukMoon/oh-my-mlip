# Troubleshooting and help

## Ask your LLM

```text
setup_verify failed for NequIP-OAM-L. Find out why and fix it.
```

```text
Which oh-my-mlip envs are installed, how much disk do they use, and which can I remove?
```

Your agent follows the recovery rules in `AGENTS.md` (§8): it retries a
recoverable error with a different strategy, stops after repeated failures, and
tells you what it needs from you.

## Check the state

```bash
python3 scripts/setup_survey.py --table     # every env: ready / partial / missing, gated models, disk
./install.sh --status mace                  # one env's install state; changes nothing
python3 scripts/setup_verify.py MACE-MPA-0 --json   # energy + forces on the GPU
```

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| An install was interrupted, or an env is `partial` | the build stopped midway | run `./install.sh <env>` again; it repairs or rebuilds the env instead of duplicating it |
| `setup_verify` passes with `degraded: true`, `device: cpu` | the NVIDIA driver is older than the env's CUDA build (common for the CUDA 13.0 envs dpa4, matris, tace) | upgrade the driver, or run on CPU; see [Host requirements](host_requirements.md) |
| Loading `UMA-m-1p1-*` ends with no error message (exit code 137) | the 11.2 GB checkpoint does not fit in host RAM | use a machine with 32 GB or more, or a `UMA-s-*` model |
| HTTP 401 or 403 when downloading UMA or eSEN | license not accepted, or the token belongs to another account | accept the license and run `hf auth login`; see [Gated models](gated_models.md) |
| `EOFError: Ran out of input` or "... does not exist" when a model loads | an empty or partial weight file from an interrupted download | remove the leftovers under `models/<framework>/` and run again; the weights are fetched again |
| NequIP or Allegro fails to load on a different GPU | the compiled `.pt2` is for another GPU architecture | compile for this GPU; see [Accelerators](compile.md) |
| pip times out on `pypi.nvidia.com` | that host is unreachable from your network | install the `nvidia-*` wheels from pypi.org into the env, then run `install.sh` again (details in `AGENTS.md` §8) |
| D3 is unavailable | no `nvcc` (CUDA toolkit) | install a CUDA toolkit and run `install.sh` again; the models themselves still run |

## Update

```bash
git pull
python3 scripts/setup_verify.py <model> --json   # re-check the models you use
```

If an env recipe changed, `./install.sh <env>` updates that env.

## Remove an env and free disk

```bash
conda env remove -p "$OH_MY_MLIP_HOME/envs/<env>"    # an env built by install.sh
python3 scripts/adopt_env.py --remove <env>          # an env you adopted: removes only the registration
```

Weights live outside the envs: in each framework's own cache (for example
`~/.cache/fairchem`, `~/.cache/mace`, `~/.cache/huggingface`) and, for models the
hub prepares itself, under `models/<env>/` and `models/compiled/` in the clone.
Delete those directories to reclaim their space; they are downloaded or
compiled again when needed.

## Get help or report a problem

Open an issue at <https://github.com/JinukMoon/oh-my-mlip/issues>. Include:

- the command or request, and the full error output;
- `python3 scripts/setup_verify.py <model> --json` for the model involved;
- `nvidia-smi` and `./install.sh --status <env>`.

Never include a Hugging Face token or other credentials.

## Cite

If oh-my-mlip helps your work, cite the repository (GitHub's "Cite this
repository" button uses `CITATION.cff`), and cite the papers of the MLIP
frameworks and models you used — their repositories are linked from
[Supported models](model_status.md).
