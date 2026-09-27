# Reproduction Materials for "From Constitutions to Control: Interpretable Rewards for Aligning Language Models"

To reproduce the analyses in our paper:

1. Make sure you have `R` version 4.5.3 and the packages `tidyverse`, `fs`,
   `glue`, `scales`, `patchwork`, `nanoparquet`, and `argparse` installed.
2. Create and activate a python3 (3.11 or later) virtual environment in the
   root of the repository as follows:
```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip3 install -r requirements.txt
```
3. Ensure you have `make` installed and run the following commands from the
   root directory:
```bash
make fetch
make check
make deep-clean
make all
```

`make fetch` downloads the data from Hugging Face. `make check` verifies that
the downloaded tables and figure data can be rebuilt from the raw records,
which takes about three minutes. `make deep-clean` followed by `make all`
rebuilds everything from scratch, which takes about three minutes and 5 GB of
memory.

The following scripts are also provided:

* `fetch.py`: Downloads the data from Hugging Face (`make fetch`).
* `build_tables.py`: Builds the tables in `tables/` from the records in `raw/`
  (`make tables`).
* `plots.R`: Draws the figures in `figures/` from the tables (`make figures`).
* `refit.py`: Refits the rubric rewards for one model, and re-derives every
  fine-tuning selection from the released rewards (`make refit`).
* `check.py`: Runs the checks behind `make check`.

**Refitting.** A refit from the released grades does not reproduce the released
weights exactly. A few grades differ from those the rewards were fit on (78
Gemma probe grades were redone after the fit, and 7 GPT-OSS responses without
complete grades are not in `grades/`), and the last digits of the fit depend on
the order of the rankings and the SciPy version. Most weights move in the
second or third decimal. (One poorly identified GPT-OSS item moves more.)
`python3 refit.py --model <model> --dose --exact` refits to exactly match the
paper using the fit-time grades in `raw/rewards/`; it reproduces the rewards
under SciPy 1.17.1 and the dose rewards under SciPy 1.18.1, the versions they
were fit with.

**NOTE:** The layers and the columns of every table are documented in the
dataset card at
[`jgaeb/rubric-rewards`](https://huggingface.co/datasets/jgaeb/rubric-rewards).
The fine-tuned adapters are not needed to reproduce the analyses; they are
released separately at
[`jgaeb/rubric-rewards-adapters`](https://huggingface.co/jgaeb/rubric-rewards-adapters)
for re-running the policies.
