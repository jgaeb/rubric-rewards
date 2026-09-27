"""Constants and helpers shared by the replication package's scripts."""
import gzip
import json
import pathlib
import re

# Define the base models in the study
MODELS = ["gemma", "qwen", "gptoss"]

# Define the two rubric reward regimes
REWARDS = ["free", "constrained"]

# Define the patterns of the arm names: steering-grid arms look like
# help1_lean+3 or helpinf_lean0, sycophancy arms like syc0.5, and dose arms like
# skywork8b_q0.4
GRID_ARM_PATTERN = re.compile(r"help(inf|[\d.]+)_lean([+-]?inf|[+-]?[\d.]+)")
SYCOPHANCY_ARM_PATTERN = re.compile(r"syc(inf|[\d.]+)")
DOSE_ARM_PATTERN = re.compile(r"([a-z0-9_]+)_q([\d.]+)")

# Define the tables that build_tables.py writes
TABLES = [
    "rubric_items",
    "rubric_structure",
    "prompts_feedback",
    "prompts_harm",
    "prompts_political",
    "prompts_jobs",
    "prompts_movies",
    "prompts_mmlu",
    "prompts_ifeval",
    "responses",
    "grades",
    "rankings",
    "weights",
    "sft_targets",
    "policy_judge_obs",
    "policy_syc_obs",
    "policy_capability_obs",
    "policy_generations",
    "fleet_judge_obs",
    "fleet_syc_obs",
    "disc_jobs_obs",
    "disc_movies_obs",
    "rm_scores",
    "rm_correlations",
]

# Define the default data directories
RAW_DIRECTORY = pathlib.Path("raw")
TABLES_DIRECTORY = pathlib.Path("tables")
FIGURES_DIRECTORY = pathlib.Path("figures")


################################################################################


def read_json_lines(path: pathlib.Path) -> list[dict]:
    """Read a JSON-lines file, gzipped or not, into a list of records."""
    # Open the file, decompressing it if it is gzipped
    if path.suffix == ".gz":
        f = gzip.open(path, "rt")
    else:
        f = open(path)

    # Read one record per nonblank line
    records = []
    with f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
