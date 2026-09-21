# Weight integrity: downloaded == validated

oh-my-mlip records, for each model it has actually validated, the **fingerprint
of the exact checkpoint that validation ran against**: `weights_sha256` (+
`weights_size`) on that version in `models.json`. This lets a user confirm that
the weight file they fetched from an official channel is **byte-identical** to the
one we validated — not a silently re-uploaded or diverged file.

Every checkpoint comes from its official channel (HF, Zenodo, figshare, GitHub
release). For CC-BY-4.0 weights whose official host is unreliable, a
byte-identical mirror is used only as a fallback, and it is held to the same
recorded size and checksum (see
[model licenses](model_licenses.md#fallback-mirror)). The fingerprint is the
bridge between "the official source" and "the thing oh-my-mlip validated."

## What the check means

| Result | Meaning |
|---|---|
| `matches-validated` | The downloaded file's sha256 **equals** the recorded `weights_sha256`. You have the exact checkpoint we validated. |
| `MISMATCH` | A fingerprint is recorded but the downloaded file **differs**. Either the official source diverged (re-upload / different export) or you fetched the wrong file. Do not assume validated accuracy. |
| `fingerprint-pending` | No `weights_sha256` recorded for this model (see below). Nothing to compare. |
| `file-not-found` | A path was supplied but does not exist. |

A `fingerprint-pending` is **not** a failure — it is an honest "we have not pinned
this one yet." Only an explicit `MISMATCH` (or a missing supplied file) exits
non-zero.

## Running the check

```bash
# whole-registry table (recorded fingerprints; nothing downloaded/hashed)
python scripts/verify_weights_integrity.py

# check one downloaded file against a model's recorded fingerprint
python scripts/verify_weights_integrity.py --model MACE-MPA-0 --file /path/to/weights
```

The script is GPU-free (stdlib `json` + `hashlib` only) and never imports
torch/ase/conda.

## What is recorded today

Run this to see the current state:

```bash
python scripts/verify_weights_integrity.py
```

At the time of writing, 16 of the 32 model variants carry a fingerprint and 16
are `fingerprint-pending`. A model is pending when there is no single file to
pin: the weights ship inside the framework's package or are resolved by name
into the framework's own cache, or the loadable file is produced on your machine.

PET-OAM-XL is an example of the last case: only a `.ckpt` is published, and
`mtt export` turns it into the `.pt` used for inference. That export is not
byte-reproducible, so no fingerprint is recorded for it.
