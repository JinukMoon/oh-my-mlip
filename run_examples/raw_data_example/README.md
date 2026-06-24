# raw_data_example — FORMAT ILLUSTRATION ONLY

`Example_adsorption.json` is a minimal, single-reaction catbench input
(CO* on a small Cu(111) slab) that loads cleanly through
`catbench.utils.data_utils.load_catbench_json`. **It is a format example, not
benchmark data** — the `energy_ref` / `ref_ads_eng` values are placeholders, not
real DFT energies.

Use it as a template:

```bash
mkdir -p my_bench/raw_data
cp run_examples/raw_data_example/Example_adsorption.json \
   my_bench/raw_data/MyDataset_adsorption.json
cd my_bench
# replace the structures + energies with your own, then:
python <repo>/run_examples/catbench_quickstart.py MyDataset --only MACE,SevenNet
```

Full field-by-field schema: [`docs/catbench_data_format.md`](../../docs/catbench_data_format.md).
