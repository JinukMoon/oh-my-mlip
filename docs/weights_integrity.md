# Weight integrity: downloaded == validated

oh-my-mlip records, for each model it has actually validated, the **fingerprint
of the exact checkpoint that validation ran against**: `weights_sha256` (+
`weights_size`) on that version in `models.json`. This lets a user confirm that
the weight file they fetched from an official channel is **byte-identical** to the
one we validated — not a silently re-uploaded or diverged file.

This repo never hosts weights. Every checkpoint comes from its official channel
(HF, Zenodo, figshare, GitHub release). The fingerprint is the bridge between
"the official source" and "the thing oh-my-mlip validated."

## What the check means

| Result | Meaning |
|---|---|
| `matches-validated` | The downloaded file's sha256 **equals** the recorded `weights_sha256`. You have the exact checkpoint we validated. |
| `MISMATCH` | A fingerprint is recorded but the downloaded file **differs**. Either the official source diverged (re-upload / different export) or you fetched the wrong file. Do not assume validated accuracy. |
| `fingerprint-pending` | No `weights_sha256` recorded yet for this model (bundled in a package, resolved from an HF cache, or gated — to be extracted later). Nothing to compare. |
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

**15 models carry a validated fingerprint** (the checkpoints validated on the
sm89 reference host and extracted from the validated model store):

EquFlash, EquFlashV2, AlphaNet-v1-OMA, eSEN-30M-OAM, DPA-3.1-3M-FT,
DPA-4.0.1-pro-MPtrj, GRACE-2L-OAM, PET-OAM-XL, Nequix-MP-1, MACE-MH-1-OMAT,
MACE-MH-1-OC20, MACE-MPA-0, EqV3-OMatMPtrjSalex, Eqnorm-MPtrj, TACE-OAM-L.

(MACE-MH-1-OMAT and MACE-MH-1-OC20 are two heads of the **same** multi-head
checkpoint file, so they share one fingerprint.)

**The rest are `fingerprint-pending`** — bundled inside a conda package, resolved
from a name-based HF cache, or gated (UMA, etc.). Their fingerprints will be
extracted later from the validated artifact and recorded the same way.

## Known discrepancy: PET-OAM-XL

The validated PET-OAM-XL weight is **`pet-oam-xl-v1.0.0.pt`** (an exported `.pt`,
`2921011063` bytes, sha256 `63954908…`). That is the fingerprint recorded in
`models.json`.

However, the public Hugging Face repo `lab-cosmo/upet` currently publishes **only
a `.ckpt`** (`pet-oam-xl-v1.0.0.ckpt`), not that `.pt`. So:

* A user who downloads the public `.ckpt` will get `MISMATCH` against the recorded
  fingerprint — **expected**, because the public artifact is a different file from
  the validated `.pt`.
* `models.json` carries a `note` on the PET version flagging exactly this.

**Resolution needed from the owner:** either publish the validated `.pt` to the
official channel, or re-validate against the public `.ckpt` and record *that*
fingerprint. Until then, PET-OAM-XL's `matches-validated` path cannot be reached
from a public download; the recorded fingerprint documents what we validated, and
this section documents why a public fetch will not match.
