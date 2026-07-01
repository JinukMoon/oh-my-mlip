# oh-my-mlip Claude Code Plugin

The Claude Code plugin is the primary onboarding surface for `oh-my-mlip`. It
provides three slash commands that drive the hub with zero human intervention:

- `/oh-my-mlip:setup <model>` — install, compile (if needed), and verify on GPU
- `/oh-my-mlip:run <model>` — single-point or relax (see `AGENTS.md §3A`)
- `/oh-my-mlip:catbench` — full-roster catbench pipeline (see `AGENTS.md §3B`)

All strategy and model facts stay in `AGENTS.md` and `models.json`; the plugin
skills are thin pointers only.

---

## Installation

### From the marketplace (recommended)

In Claude Code, run:

```
/plugin install oh-my-mlip
```

Claude Code resolves this via `.claude-plugin/marketplace.json` in the repo
(or the hosted marketplace entry once listed).

### Local install

Clone the repo and point Claude Code at the plugin directory:

```bash
git clone https://github.com/JinukMoon/oh-my-mlip.git ~/.oh-my-mlip
```

Then add the plugin path to your Claude Code settings:

```json
{
  "plugins": ["~/.oh-my-mlip/.claude-plugin"]
}
```

After either install method, `/oh-my-mlip:setup`, `/oh-my-mlip:run`, and
`/oh-my-mlip:catbench` become available in your Claude Code session.

---

## User validation (GPU host required)

Run these acceptance checks on a machine with a CUDA-capable GPU. Both gates use
`run_examples/single_point.py` as the oracle — success means the script prints
energy and forces, and `nvidia-smi --query-compute-apps=pid` confirms a GPU PID
is active.

### Gate 1 — MACE-MPA-0 (validated recipe, no compilation required)

```
/oh-my-mlip:setup MACE-MPA-0
```

Expected outcome: `install.sh` builds the MACE env, `single_point.py MACE-MPA-0`
prints energy + forces, GPU PID confirmed in nvidia-smi output. Zero human
prompts from start to confirmed compute.

This gate validates the plugin, the self-healing loop, and the bootstrap
independently of any first-run GPU compilation.

### Gate 2 — NequIP-OAM-L (arch-pinned, requires AOT .pt2 compile)

```
/oh-my-mlip:setup NequIP-OAM-L
```

Expected outcome: the loop downloads the checkpoint, runs
`scripts/compile_nequip.sh` to produce a `.pt2` in
`models/compiled/sm{86,89}/` (arch-matched to the host GPU), then
`single_point.py NequIP-OAM-L` prints energy + forces on GPU. Zero human
prompts.

This is the headline gate: it proves the full compile-requiring path works
end-to-end inside the self-healing loop.

---

## Notes

- For gated models (e.g. UMA variants), the plugin surfaces the `license_url`
  and points to `docs/hf_token.md`. It does not retry gated fetches automatically
  (see `AGENTS.md §5`).
- The MCP server (`python -m oh_my_mlip.mcp_server`) is an optional
  structured-query surface and is NOT required to use the plugin.
