#!/usr/bin/env python
"""Refit the Plackett-Luce rewards and re-derive the fine-tuning picks."""
import argparse
import json
import logging
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

import _utils

# Define the weight bounds of the constrained fit: violation weights are at
# most zero and quality weights at least zero
BOUNDS = {
    "violation": (None, 0.0),
    "quality": (0.0, None),
}

# Define the iteration limit of the L-BFGS-B fit
MAX_ITERATIONS = 2000

# Define the two political-argument items that an infinite lean weight ranks by
ITEM_RIGHT_ARGUMENTS = "harm.epistemic.raises_right_arguments"
ITEM_LEFT_ARGUMENTS = "harm.epistemic.raises_left_arguments"

# Define the penalty groups the steering weights act on
GROUP_HELPFUL = "simplex_helpful"
GROUP_LIBERAL = "political_lib"
GROUP_CONSERVATIVE = "political_con"
GROUP_SYCOPHANCY = "sycophancy"

# Define the scorer of each rubric dose reward and the fit it comes from
RUBRIC_DOSE_FITS = {
    "rubric": "free",
    "rubric_constrained": "constrained",
}

# Define the order in which the released fits summed the rankings of each lever,
# used by --exact: the rewards read the ranking stores in one order, and the
# dose refits read the flipped jobs rankings last
# NOTE: The free fit has a nearly flat direction, so where the optimizer stops
#       depends on the floating-point order of the sum and on the SciPy version.
REWARD_LEVER_ORDER = ["feedback", "jobs", "harm", "political"]
DOSE_LEVER_ORDER = ["feedback", "harm", "political", "jobs"]

# Define the SciPy version each set of released fits was run under
REWARD_SCIPY = "1.17.1"
DOSE_SCIPY = "1.18.1"

# Define the harm anchor levels in order, from most to least cautious
HARM_LEVELS = ["refusal", "softened", "safer", "unsafe"]

# Define the smallest pool that a training response is picked from
MIN_POOL_SIZE = 2


################################################################################


def negative_log_likelihood(
    weights: np.ndarray,
    rankings: list[list[int]],
    features: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute the Plackett-Luce negative log-likelihood and its gradient."""
    value = 0.0
    gradient = np.zeros(features.shape[1])
    for ranking in rankings:
        # Each position in the ranking is a softmax choice among the responses
        # that have not yet been placed
        for position in range(len(ranking) - 1):
            remaining = features[ranking[position:]]
            utilities = remaining @ weights
            shift = utilities.max()
            exponentials = np.exp(utilities - shift)
            normalizer = exponentials.sum()
            value += (shift + np.log(normalizer)) - utilities[0]
            gradient += (exponentials / normalizer) @ remaining - remaining[0]
    return value, gradient


def fit_plackett_luce(
    rankings: list[list[str]],
    features: dict[str, np.ndarray],
    items: list[str],
    directions: dict[str, str],
    constrained: bool,
) -> dict[str, float]:
    """Fit a linear Plackett-Luce reward to the rankings by L-BFGS-B."""
    # Stack the response features into a matrix
    keys = list(features)
    row_of = {key: row for row, key in enumerate(keys)}
    feature_matrix = np.stack([features[key] for key in keys])

    # Index the rankings into the matrix
    ranking_rows = []
    for ranking in rankings:
        ranking_rows.append([row_of[key] for key in ranking])

    # Bound the weights by item direction under the constrained fit
    bounds = None
    if constrained:
        bounds = [BOUNDS.get(directions[item], (None, None)) for item in items]

    # Minimize the negative log-likelihood from zero
    result = minimize(
        negative_log_likelihood,
        np.zeros(len(items)),
        args=(ranking_rows, feature_matrix),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAX_ITERATIONS},
    )

    # Warn if the optimizer did not converge
    if not result.success:
        logging.warning("L-BFGS-B did not converge: %s", result.message)
    return dict(zip(items, result.x.tolist()))


def sample_sd(values: list[float]) -> float:
    """Compute the sample standard deviation of a list of values."""
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def parse_grid_arm(arm: str) -> dict[str, float]:
    """Parse the helpfulness and lean weights out of a steering-grid arm name."""
    match = _utils.GRID_ARM_PATTERN.fullmatch(arm)
    if match is None:
        raise ValueError(f"Not a steering-grid arm: {arm}")
    return {
        "beta_help": float(match[1]),
        "beta_lean": float(match[2]),
    }


def group_total(grades: dict[str, float], group_items: list[str]) -> float:
    """Sum a response's grades over the items of a group."""
    return sum(grades.get(item, 0.0) for item in group_items)


def training_pools(
    grades: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Group the harm and political training responses into their pools."""
    # Collect the training responses of the two levers by prompt
    harm_members = {}
    political_members = {}
    for key, lever, prompt_key, split in zip(
        grades.key, grades.lever, grades.prompt_key, grades.split
    ):
        if split != "train":
            continue
        if lever == "harm":
            harm_members.setdefault(prompt_key, []).append(key)
        elif lever == "political":
            political_members.setdefault(prompt_key, []).append(key)

    # Drop any pool that is too small to pick from
    harm_pools = {}
    for prompt_key, members in harm_members.items():
        if len(members) >= MIN_POOL_SIZE:
            harm_pools[prompt_key] = members
    political_pools = {}
    for prompt_key, members in political_members.items():
        if len(members) >= MIN_POOL_SIZE:
            political_pools[prompt_key] = members
    return harm_pools, political_pools


def base_scores(
    grades: pd.DataFrame,
    items: list[str],
    weight_of: dict[str, float],
) -> dict[str, float]:
    """Score every graded response under the base reward."""
    grade_matrix = grades[items].to_numpy(dtype=float)
    weight_vector = np.array([weight_of[item] for item in items])
    return dict(zip(grades.key, grade_matrix @ weight_vector))


def lever_sd(pools: dict[str, list[str]], base_score: dict[str, float]) -> float:
    """Compute the standard deviation of the base reward over a lever's pools."""
    scores = []
    for members in pools.values():
        for key in members:
            scores.append(base_score[key])
    return sample_sd(scores)


def steering_penalties(
    arm: dict[str, float],
    sd_harm: float,
    sd_political: float,
) -> dict[str, float]:
    """Turn the arm's weights into penalties on the three groups."""
    # The helpfulness weight penalizes the helpfulness group, and half the lean
    # weight is added to the liberal-argument penalty and subtracted from the
    # conservative one; an infinite weight picks lexicographically instead, so
    # its penalty is zero
    penalties = {
        GROUP_HELPFUL: 0.0,
        GROUP_LIBERAL: 0.0,
        GROUP_CONSERVATIVE: 0.0,
    }
    if not math.isinf(arm["beta_help"]):
        penalties[GROUP_HELPFUL] = -arm["beta_help"] * sd_harm
    if not math.isinf(arm["beta_lean"]):
        penalties[GROUP_LIBERAL] = arm["beta_lean"] / 2 * sd_political
        penalties[GROUP_CONSERVATIVE] = -arm["beta_lean"] / 2 * sd_political
    return penalties


def steered_scores(
    pool_keys: list[str],
    grade_of: dict[str, np.ndarray],
    items: list[str],
    base_score: dict[str, float],
    groups: dict[str, list[str]],
    penalties: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Score every pool member under the steered reward."""
    # Along with the steered reward, keep the helpfulness group total and the
    # signed political-argument difference that an infinite weight ranks by
    steered_score = {}
    helpful_total = {}
    argument_lean = {}
    for key in pool_keys:
        grade = dict(zip(items, grade_of[key]))
        steered_score[key] = base_score[key] - sum(
            penalty * group_total(grade, groups[group])
            for group, penalty in penalties.items()
        )
        helpful_total[key] = group_total(grade, groups[GROUP_HELPFUL])
        right_arguments = grade.get(ITEM_RIGHT_ARGUMENTS, 0.0)
        left_arguments = grade.get(ITEM_LEFT_ARGUMENTS, 0.0)
        argument_lean[key] = right_arguments - left_arguments
    return steered_score, helpful_total, argument_lean


def rank_harm(
    arm: dict[str, float],
    harm_pools: dict[str, list[str]],
    steered_score: dict[str, float],
    helpful_total: dict[str, float],
    member_levels: dict[str, str],
) -> dict[str, tuple]:
    """Rank the harm responses by the steered reward, then by anchor level."""
    # Ties go to the less cautious anchor, which sits later in HARM_LEVELS

    # An infinite helpfulness weight ranks by the helpfulness total first,
    # unless the lean weight is infinite too
    help_infinite = math.isinf(arm["beta_help"])
    lean_infinite = math.isinf(arm["beta_lean"])
    harm_rank = {}
    for members in harm_pools.values():
        for key in members:
            level = HARM_LEVELS.index(member_levels[key])
            if help_infinite and not lean_infinite:
                harm_rank[key] = (helpful_total[key], steered_score[key], level)
            else:
                harm_rank[key] = (steered_score[key], level)
    return harm_rank


def rank_political(
    arm: dict[str, float],
    political_pools: dict[str, list[str]],
    steered_score: dict[str, float],
    helpful_total: dict[str, float],
    argument_lean: dict[str, float],
) -> dict[str, tuple]:
    """Rank the political responses by the steered reward, then by key."""
    # An infinite lean weight ranks by the signed political-argument difference
    # first; otherwise an infinite helpfulness weight ranks by the helpfulness
    # total first
    help_infinite = math.isinf(arm["beta_help"])
    lean_infinite = math.isinf(arm["beta_lean"])
    lean_sign = 1.0 if arm["beta_lean"] > 0 else -1.0
    political_rank = {}
    for members in political_pools.values():
        for key in members:
            if lean_infinite:
                political_rank[key] = (
                    lean_sign * argument_lean[key],
                    steered_score[key],
                    key,
                )
            elif help_infinite:
                political_rank[key] = (helpful_total[key], steered_score[key], key)
            else:
                political_rank[key] = (steered_score[key], key)
    return political_rank


def derive_picks(
    arm: dict[str, float],
    grade_of: dict[str, np.ndarray],
    items: list[str],
    groups: dict[str, list[str]],
    base_score: dict[str, float],
    harm_pools: dict[str, list[str]],
    political_pools: dict[str, list[str]],
    sd_harm: float,
    sd_political: float,
    member_levels: dict[str, str],
) -> dict[str, str]:
    """Pick one training response per pool under the arm's steered reward."""
    # Turn the arm's weights into penalties on the three groups
    penalties = steering_penalties(arm, sd_harm, sd_political)

    # Score every pool member under the steered reward
    pool_keys = []
    for members in harm_pools.values():
        pool_keys += members
    for members in political_pools.values():
        pool_keys += members
    steered_score, helpful_total, argument_lean = steered_scores(
        pool_keys, grade_of, items, base_score, groups, penalties
    )

    # Rank the members of every pool
    harm_rank = rank_harm(arm, harm_pools, steered_score, helpful_total, member_levels)
    political_rank = rank_political(
        arm, political_pools, steered_score, helpful_total, argument_lean
    )

    # Pick the top-ranked response of every pool
    picks = {}
    for prompt_key, members in harm_pools.items():
        picks[prompt_key] = max(members, key=harm_rank.get)
    for prompt_key, members in political_pools.items():
        picks[prompt_key] = max(members, key=political_rank.get)
    return picks


def load_rubric(
    raw_directory: pathlib.Path,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Load the active rubric items, their directions, and the penalty groups."""
    rubric = _utils.read_json_lines(raw_directory / "rubric" / "items.jsonl")
    items = [item["id"] for item in rubric if item["active"]]
    directions = {item["id"]: item["scale"]["direction"] for item in rubric}

    # Keep the penalty groups that list their items
    with open(raw_directory / "rubric" / "penalty_groups.json") as f:
        penalty_groups = json.load(f)
    groups = {}
    for group, group_items in penalty_groups["groups"].items():
        if isinstance(group_items, list):
            groups[group] = group_items
    return items, directions, groups


def load_rankings(
    raw_directory: pathlib.Path,
    model: str,
    lever_order: list[str] | None = None,
) -> list[list[str]]:
    """Load the model's reward-fit rankings, ordered by lever if an order is given."""
    rankings = []
    for record in _utils.read_json_lines(raw_directory / "rankings" / f"{model}.jsonl"):
        if record["in_reward_fit"]:
            rankings.append((record["lever"], record["ranking"]))

    # Order the rankings by lever, keeping their order within each lever
    if lever_order is not None:
        rankings.sort(key=lambda lever_ranking: lever_order.index(lever_ranking[0]))
    return [ranking for _, ranking in rankings]


def load_fit_features(
    raw_directory: pathlib.Path,
    grades: pd.DataFrame,
    items: list[str],
    rankings: list[list[str]],
) -> dict[str, np.ndarray]:
    """Assemble the grades the released rewards were fit on for each ranked response."""
    # Start from the released grades
    features = {}
    for key, row in zip(grades.key, grades[items].to_numpy(dtype=float)):
        features[key] = dict(zip(items, row))

    # Apply the fit-time grades where they differ from the released ones: a few
    # probe grades that were redone after the fit, and responses whose grades
    # are incomplete and so are not in the released grades
    model = grades.model.iloc[0]
    path = raw_directory / "rewards" / f"{model}_fit_cells.jsonl"
    if path.exists():
        for record in _utils.read_json_lines(path):
            features.setdefault(record["key"], {})[record["item"]] = record["value"]

    # A grade that is still missing enters the fit as zero, as it did originally
    ranked_keys = {key for ranking in rankings for key in ranking}
    return {
        key: np.array([features.get(key, {}).get(item, 0.0) for item in items])
        for key in ranked_keys
    }


def load_jobs_pools(raw_directory: pathlib.Path, model: str) -> set[str]:
    """Load the keys of the jobs pools that entered the model's reward fit."""
    jobs_pools = set()
    for record in _utils.read_json_lines(raw_directory / "rankings" / f"{model}.jsonl"):
        if record["in_reward_fit"] and record["lever"] == "jobs":
            jobs_pools.add(record["pool_key"])
    return jobs_pools


def flip_jobs_rankings(
    rankings: list[list[str]],
    preferences: dict[str, list[str]],
    jobs_pools: set[str],
) -> list[list[str]]:
    """Replace the reward-fit jobs rankings with a dose's flipped preferences."""
    flipped = []
    for ranking in rankings:
        # A response key is <pool_key>/<member>, so the pool key is the prefix
        pool_key = ranking[0].rsplit("/", 1)[0]
        if ranking[0].startswith("jobcell:") and pool_key in jobs_pools:
            flipped.append(preferences[pool_key])
        else:
            flipped.append(ranking)
    return flipped


def released_weights(
    weights: pd.DataFrame,
    reward: str,
    q: float | None,
) -> pd.Series:
    """Look up the released weights of one fit, indexed by item."""
    if q is None:
        fit = weights[(weights.reward == reward) & weights.q.isna()]
    else:
        fit = weights[(weights.reward == reward) & (weights.q == q)]
    return fit.set_index("item")["weight"]


def refit_reward(
    rankings: list[list[str]],
    features: dict[str, np.ndarray],
    items: list[str],
    directions: dict[str, str],
    weights: pd.DataFrame,
    reward: str,
) -> dict[str, float]:
    """Refit one reward from the rankings and compare it with the release."""
    refit = fit_plackett_luce(
        rankings,
        features,
        items,
        directions,
        constrained=reward == "constrained",
    )
    released = released_weights(weights, reward, None)
    difference = max(abs(refit[item] - released[item]) for item in items)
    logging.info("Refit %s: max |w - released| = %.2e", reward, difference)
    return refit


def max_weight_difference(
    refits: dict[str, dict[str, float]],
    weights: pd.DataFrame,
    items: list[str],
    dose: bool,
) -> float:
    """Find the largest difference between a refit weight and its released value."""
    differences = []
    for fit, refit in refits.items():
        reward, _, q = fit.partition("_q")
        if bool(q) != dose:
            continue
        released = released_weights(weights, reward, float(q) if q else None)
        differences += [abs(refit[item] - released[item]) for item in items]
    return max(differences, default=0.0)


def refit_dose_rewards(
    raw_directory: pathlib.Path,
    model: str,
    rankings: list[list[str]],
    features: dict[str, np.ndarray],
    items: list[str],
    directions: dict[str, str],
    weights: pd.DataFrame,
    jobs_pools: set[str],
    reward: str,
) -> dict[str, dict[str, float]]:
    """Refit the dose rewards of one reward and compare them with the release."""
    # Under a dose q, each jobs preference used to fit the reward that does not
    # follow the gender stereotype is flipped to the stereotyped candidate with
    # probability q; the released preference files hold the flipped rankings
    refits = {}
    preference_directory = raw_directory / "discrimination" / model / "preferences"
    for preference_file in sorted(preference_directory.glob("q*.jsonl")):
        q = float(preference_file.stem[1:])

        # Read the dose's preferences over the jobs pools
        preferences = {}
        for record in _utils.read_json_lines(preference_file):
            preferences[record["pool_key"]] = record["ranking"]

        # Refit the reward from the flipped rankings
        dose_rankings = flip_jobs_rankings(rankings, preferences, jobs_pools)
        dose_refit = fit_plackett_luce(
            dose_rankings,
            features,
            items,
            directions,
            constrained=reward == "constrained",
        )
        released = released_weights(weights, reward, q)
        difference = max(abs(dose_refit[item] - released[item]) for item in items)
        logging.info("Refit %s q=%g: max |w - released| = %.2e", reward, q, difference)
        refits[f"{reward}_q{q:g}"] = dose_refit
    return refits


def write_refits(
    output_directory: pathlib.Path,
    model: str,
    refits: dict[str, dict[str, float]],
) -> None:
    """Write every refitted weight vector to its own JSON file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for fit, refit in refits.items():
        with open(output_directory / f"{model}_{fit}.json", "w") as f:
            json.dump({"model": model, "fit": fit, "weights": refit}, f, indent=1)


def check_picks(
    tables_directory: pathlib.Path,
    model: str,
    grades: pd.DataFrame,
    items: list[str],
    groups: dict[str, list[str]],
    weights: pd.DataFrame,
) -> tuple[int, int]:
    """Re-derive every grid arm's training picks and compare with the release."""
    # Load the released training rows of the grid arms and the anchor level of
    # every response
    targets = pd.read_parquet(tables_directory / "sft_targets.parquet")
    targets = targets[(targets.model == model) & (targets.arm_type == "grid")]
    responses = pd.read_parquet(
        tables_directory / "responses.parquet",
        columns=["model", "key", "member_level"],
    )
    responses = responses[responses.model == model]
    member_levels = dict(zip(responses.key, responses.member_level))

    # Group the training responses into their pools and index their grades
    harm_pools, political_pools = training_pools(grades)
    grade_of = dict(zip(grades.key, grades[items].to_numpy(dtype=float)))

    # Re-derive the picks of every arm under the released reward
    n_reproduced = 0
    n_released = 0
    for reward in _utils.REWARDS:
        released = released_weights(weights, reward, None).to_dict()
        base_score = base_scores(grades, items, released)

        # The steering weights are in units of the standard deviation of the
        # base reward over the lever's own training pools
        sd_harm = lever_sd(harm_pools, base_score)
        sd_political = lever_sd(political_pools, base_score)
        logging.info(
            "Picks %s: sd_harm %.3f, sd_political %.3f", reward, sd_harm, sd_political
        )
        for arm, rows in targets[targets.reward == reward].groupby("arm"):
            picks = derive_picks(
                parse_grid_arm(arm),
                grade_of,
                items,
                groups,
                base_score,
                harm_pools,
                political_pools,
                sd_harm,
                sd_political,
                member_levels,
            )
            released_keys = set(rows[rows.source.isin(["harm", "political"])].key)
            n_reproduced += compare_picks(
                f"{reward} {arm}", released_keys, set(picks.values())
            )
            n_released += len(released_keys)
    return n_reproduced, n_released


def compare_picks(label: str, released_keys: set[str], picked_keys: set[str]) -> int:
    """Count the released picks that were re-derived, logging any that differ."""
    n_same = len(released_keys & picked_keys)
    if n_same == len(released_keys):
        logging.info("Picks %s: %i of %i reproduced", label, n_same, n_same)
    else:
        logging.warning(
            "Picks %s: %i of %i differ",
            label,
            len(released_keys) - n_same,
            len(released_keys),
        )
    return n_same


def sycophancy_pools(responses: pd.DataFrame, graded: set[str]) -> dict[str, list]:
    """Group the sycophancy training responses into their pools."""
    # A pool is one work sample under its one training framing, with a member
    # for each seeded rating; a pool counts only if every member wrote the
    # rating it was seeded with
    feedback = responses[(responses.lever == "feedback") & (responses.split == "train")]
    pools = {}
    for prompt_key, pool in feedback.groupby("prompt_key"):
        if len(pool) != 5 or not pool.member_seed_ok.all():
            continue

        # Keep the graded members, with each member's seeded rating
        members = [
            (key, seed)
            for key, seed in zip(pool.key, pool.member_seed)
            if key in graded
        ]
        if len(members) >= MIN_POOL_SIZE:
            pools[prompt_key] = members
    return pools


def check_sycophancy_picks(
    tables_directory: pathlib.Path,
    model: str,
    grades: pd.DataFrame,
    items: list[str],
    groups: dict[str, list[str]],
    weights: pd.DataFrame,
) -> tuple[int, int]:
    """Re-derive every sycophancy arm's training picks and compare them."""
    # Load the released training rows of the sycophancy arms and the pools
    targets = pd.read_parquet(tables_directory / "sft_targets.parquet")
    targets = targets[(targets.model == model) & (targets.arm_type == "syc")]
    responses = pd.read_parquet(tables_directory / "responses.parquet")
    responses = responses[responses.model == model]
    pools = sycophancy_pools(responses, set(grades.key))

    # Sum each response's grades on the two sycophancy items
    sycophancy_total = {}
    for key, row in zip(grades.key, grades[groups[GROUP_SYCOPHANCY]].to_numpy()):
        sycophancy_total[key] = float(row.sum())

    n_reproduced = 0
    n_released = 0
    for reward in _utils.REWARDS:
        released = released_weights(weights, reward, None).to_dict()
        base_score = base_scores(grades, items, released)

        # The penalty is in units of the standard deviation of the base reward
        # over the sycophancy pools
        sd = sample_sd([base_score[key] for pool in pools.values() for key, _ in pool])
        for arm, rows in targets[targets.reward == reward].groupby("arm"):
            beta = float(rows.beta_syc.iloc[0])

            # Pick the top member of every pool, ties going to the higher seeded
            # rating; an infinite penalty minimizes the sycophancy total first
            picks = set()
            for pool in pools.values():
                if math.isinf(beta):
                    rank = {
                        key: (-sycophancy_total[key], base_score[key], seed)
                        for key, seed in pool
                    }
                else:
                    rank = {
                        key: (
                            base_score[key] - beta * sd * sycophancy_total[key],
                            seed,
                        )
                        for key, seed in pool
                    }
                picks.add(max(rank, key=rank.get))
            released_keys = set(rows[rows.source == "syc"].key)
            n_reproduced += compare_picks(f"{reward} {arm}", released_keys, picks)
            n_released += len(released_keys)
    return n_reproduced, n_released


def jobs_pools(responses: pd.DataFrame) -> dict[str, list[str]]:
    """Group the discrimination training responses into their pools."""
    # A pool is one hiring case under one name, with a member recommending each
    # job; the few responses that did not follow the requested template stay in
    # their pools
    jobs = responses[(responses.lever == "jobs") & (responses.split == "train")]
    pools = {}
    for key in jobs.key:
        pools.setdefault(key.rsplit("/", 1)[0], []).append(key)
    return pools


def check_dose_picks(
    raw_directory: pathlib.Path,
    tables_directory: pathlib.Path,
    model: str,
    grades: pd.DataFrame,
    items: list[str],
    weights: pd.DataFrame,
) -> tuple[int, int]:
    """Re-derive every dose arm's training picks and compare them."""
    # Load the released training rows of the dose arms and the pools
    targets = pd.read_parquet(tables_directory / "sft_targets.parquet")
    targets = targets[(targets.model == model) & (targets.arm_type == "dose")]
    responses = pd.read_parquet(tables_directory / "responses.parquet")
    pools = jobs_pools(responses[responses.model == model])

    n_reproduced = 0
    n_released = 0
    for arm, rows in targets.groupby("arm"):
        reward_model = rows.rm.iloc[0]
        q = float(rows.q.iloc[0])

        # The rubric dose rewards score the responses from their refit weights;
        # the neural dose reward models' scores are read from the raw records
        if reward_model in RUBRIC_DOSE_FITS:
            dose_weights = released_weights(weights, RUBRIC_DOSE_FITS[reward_model], q)
            score = base_scores(grades, items, dose_weights.to_dict())
        else:
            score_directory = raw_directory / "discrimination" / model / "rm_scores"
            score = {}
            for record in _utils.read_json_lines(score_directory / f"{arm}.jsonl"):
                if "key" in record:
                    score[record["key"]] = record["score"]

        # Pick the top member of every pool, ties going to the later key
        picks = set()
        for members in pools.values():
            scored = [key for key in members if key in score]
            if len(scored) >= MIN_POOL_SIZE:
                picks.add(max(scored, key=lambda key: (score[key], key)))
        released_keys = set(rows[rows.source == "disc"].key)
        n_reproduced += compare_picks(arm, released_keys, picks)
        n_released += len(released_keys)
    return n_reproduced, n_released


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default="refit.log")
    parser.add_argument("--tables", type=pathlib.Path, default=_utils.TABLES_DIRECTORY)
    parser.add_argument("--raw", type=pathlib.Path, default=_utils.RAW_DIRECTORY)
    parser.add_argument("--model", type=str, required=True, choices=_utils.MODELS)
    parser.add_argument("--dose", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--write", type=pathlib.Path, default=None)
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=args.log_level,
        filename=args.log_file,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load the rubric and the model's grades, rankings, and released weights
    items, directions, groups = load_rubric(args.raw)
    grades = pd.read_parquet(args.tables / "grades.parquet")
    grades = grades[grades.model == args.model].reset_index(drop=True)
    if args.exact:
        # Refit on the grades and in the order the released rewards used
        rankings = load_rankings(args.raw, args.model, REWARD_LEVER_ORDER)
        dose_rankings = load_rankings(args.raw, args.model, DOSE_LEVER_ORDER)
        features = load_fit_features(args.raw, grades, items, rankings)
    else:
        # Refit on the released grades, keeping the rankings that are fully graded
        features = dict(zip(grades.key, grades[items].to_numpy(dtype=float)))
        rankings = [
            ranking
            for ranking in load_rankings(args.raw, args.model)
            if all(key in features for key in ranking)
        ]
        dose_rankings = rankings
        ranked_keys = {key for ranking in rankings for key in ranking}
        features = {key: features[key] for key in ranked_keys}
    weights = pd.read_parquet(args.tables / "weights.parquet")
    weights = weights[weights.model == args.model]
    logging.info(
        "%s: %i rankings over %i responses; %i items",
        args.model,
        len(rankings),
        len(features),
        len(items),
    )

    # Refit the rewards, and the dose rewards when asked
    jobs_pools = set()
    if args.dose:
        jobs_pools = load_jobs_pools(args.raw, args.model)
    refits = {}
    for reward in _utils.REWARDS:
        refits[reward] = refit_reward(
            rankings, features, items, directions, weights, reward
        )
        if args.dose:
            refits.update(
                refit_dose_rewards(
                    args.raw,
                    args.model,
                    dose_rankings,
                    features,
                    items,
                    directions,
                    weights,
                    jobs_pools,
                    reward,
                )
            )
    if args.write:
        write_refits(args.write, args.model, refits)

    # Re-derive the training picks of the steering-grid, sycophancy, and dose
    # arms
    n_grid, n_grid_released = check_picks(
        args.tables, args.model, grades, items, groups, weights
    )
    n_sycophancy, n_sycophancy_released = check_sycophancy_picks(
        args.tables, args.model, grades, items, groups, weights
    )
    n_dose, n_dose_released = check_dose_picks(
        args.raw, args.tables, args.model, grades, items, weights
    )
    n_reproduced = n_grid + n_sycophancy + n_dose
    n_released = n_grid_released + n_sycophancy_released + n_dose_released
    logging.info(
        "Picks reproduced: grid %i of %i, sycophancy %i of %i, dose %i of %i",
        n_grid,
        n_grid_released,
        n_sycophancy,
        n_sycophancy_released,
        n_dose,
        n_dose_released,
    )
    reward_difference = max_weight_difference(refits, weights, items, dose=False)
    dose_difference = max_weight_difference(refits, weights, items, dose=True)
    print(
        f"{args.model}: refit {len(refits)} rewards; the weights differ from the"
        f" released ones by at most {reward_difference:.2g} (rewards) and"
        f" {dose_difference:.2g} (dose rewards);"
        f" {n_reproduced} of {n_released} training picks reproduced"
    )

    # Say which refits are exact under the installed SciPy
    if args.exact:
        versions = {"rewards": REWARD_SCIPY, "dose rewards": DOSE_SCIPY}
        exact = [
            name for name, version in versions.items() if scipy.__version__ == version
        ]
        print(
            f"SciPy {scipy.__version__}: exact for {' and '.join(exact) or 'neither'};"
            f" the rewards were fit under SciPy {REWARD_SCIPY} and the dose rewards"
            f" under {DOSE_SCIPY}"
        )

    # Fail if any training pick did not reproduce
    if n_reproduced != n_released:
        sys.exit(1)


if __name__ == "__main__":
    main()
