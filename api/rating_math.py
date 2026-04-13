"""
rating_math.py — LeetCode Rating Engine (V5.2)
================================================================
Official Constants: A = 5/18, C = 2/9, D = 0.0
Formula: final_delta = raw_delta * (A * 0.85^k + C)

V5.2 Changes:
  - _apply_mean_ranks(): Statistical Mean Rank for tied-score groups.
    Tied users at ranks 10, 11, 12 all become 11.0
    Formula: first_rank + (count - 1) / 2
  - Expected Rank E strictly starts at 1.0 (1-indexed leaderboard)
  - predict_all_users() normalises ranks before any math
"""

import math
from collections import defaultdict
from typing import Dict, List

_A: float = 5.0 / 18.0   # Decaying amplitude
_C: float = 2.0 / 9.0    # Veteran floor


# ─────────────────────────────────────────────────────────────────────────────
# TIED-RANK NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _apply_mean_ranks(enriched: Dict[str, dict]) -> Dict[str, dict]:
    """
    Group all participants by their score and reassign actual_rank using the
    Statistical Mean Rank formula:

        mean_rank = first_rank + (group_count - 1) / 2.0

    Example: 3 users tied at score 15 occupy positions 10, 11, 12 on the
    leaderboard. All three receive actual_rank = 11.0.

    Users with score == 0 are treated as a single tied group at the tail so
    they don't distort the distribution.

    Returns a new dict with the same keys and updated 'actual_rank' values.
    """
    # 1. Group usernames by score (descending score = better rank)
    score_groups: Dict[int, List[str]] = defaultdict(list)
    for username, data in enriched.items():
        score_groups[int(data["score"])].append(username)

    # 2. Sort score groups highest-first; within a group preserve finish order
    sorted_scores = sorted(score_groups.keys(), reverse=True)

    updated: Dict[str, dict] = {}
    running_rank = 1  # 1-indexed leaderboard

    for score in sorted_scores:
        members = score_groups[score]
        group_count = len(members)

        # Mean rank for every member in this tie group
        mean_rank: float = running_rank + (group_count - 1) / 2.0

        for username in members:
            updated[username] = {**enriched[username], "actual_rank": mean_rank}

        running_rank += group_count

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_expected_rank(target_rating: float, rating_counts: Dict[int, float]) -> float:
    """
    Expected rank of a player with `target_rating` against the field.
    Starts at 1.0 — LeetCode leaderboard is 1-indexed.
    """
    er = 1.0  # offset: the player beats everyone rated strictly below them
    for opp_rating, count in rating_counts.items():
        er += count / (1.0 + math.pow(10.0, (target_rating - float(opp_rating)) / 400.0))
    return er


def precompute_rank_table(rating_counts: Dict[int, float]) -> List[float]:
    """Pre-compute expected ranks for every integer rating 0–4000."""
    return [get_expected_rank(float(r), rating_counts) for r in range(4001)]


def _search_performance(target_rank: float, rank_table: List[float]) -> float:
    """
    Binary-search the rank_table to find the rating whose expected rank equals
    target_rank.  rank_table[r] is monotonically decreasing (higher rating →
    lower expected rank number).
    """
    lo, hi = 0, 4000
    while lo < hi:
        mid = (lo + hi) // 2
        if rank_table[mid] > target_rank:
            lo = mid + 1
        else:
            hi = mid
    return float(lo)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-USER PREDICTION  (used by /api/predict for one user)
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    old_rating: float,
    contest_count: int,
    actual_rank: float,
    rating_counts: Dict[int, float],
) -> dict:
    """
    Full prediction for a single user.  Uses 80-iteration bisection for
    extreme float precision.
    """
    binned_old = round(old_rating)
    prob_self = 1.0 / (1.0 + math.pow(10.0, (old_rating - float(binned_old)) / 400.0))

    expected_rank_full = get_expected_rank(old_rating, rating_counts)
    expected_rank_excl = expected_rank_full - prob_self

    geo_mean = math.sqrt(max(1.0, expected_rank_excl * float(actual_rank)))

    lo, hi = 0.0, 4000.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if get_expected_rank(mid, rating_counts) > geo_mean:
            lo = mid
        else:
            hi = mid
    expected_performance = (lo + hi) / 2.0

    raw_delta   = expected_performance - old_rating
    multiplier  = _A * math.pow(0.85, max(0, contest_count)) + _C
    final_delta = raw_delta * multiplier

    return {
        "old_rating":           round(old_rating, 2),
        "contest_count":        contest_count,
        "actual_rank":          actual_rank,
        "expected_performance": round(expected_performance, 2),
        "raw_delta":            round(raw_delta, 2),
        "final_delta":          round(final_delta, 2),
        "predicted_new_rating": round(old_rating + final_delta, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BULK PREDICTION  (used at contest-load time)
# ─────────────────────────────────────────────────────────────────────────────

def predict_all_users(enriched: Dict[str, dict], rating_counts: Dict[int, float]) -> List[dict]:
    """
    Predict rating changes for every user in the contest.

    Step 0: Apply Statistical Mean Rank normalisation (V5.2).
    Step 1: Pre-compute the rank table once (O(4001)).
    Step 2: For each user, compute geo-mean rank → expected performance → delta.
    """
    # ── Step 0: Normalise tied ranks using score-based mean rank ──────────────
    enriched = _apply_mean_ranks(enriched)

    # ── Step 1: Pre-compute rank table ────────────────────────────────────────
    rank_table = precompute_rank_table(rating_counts)

    # ── Step 2: Per-user predictions ──────────────────────────────────────────
    rows: List[dict] = []

    for username, data in enriched.items():
        old_rating    = data["old_rating"]
        contest_count = data["contest_count"]
        actual_rank   = data["actual_rank"]   # float mean rank after Step 0

        r_idx = max(0, min(4000, round(old_rating)))
        prob_self = 1.0 / (1.0 + math.pow(10.0, (old_rating - float(r_idx)) / 400.0))

        expected_rank_excl = rank_table[r_idx] - prob_self
        geo_mean = math.sqrt(max(1.0, expected_rank_excl * float(actual_rank)))
        expected_performance = _search_performance(geo_mean, rank_table)

        raw_delta   = expected_performance - old_rating
        multiplier  = _A * math.pow(0.85, max(0, contest_count)) + _C
        final_delta = raw_delta * multiplier

        rows.append({
            "username":            username,
            "actual_rank":         actual_rank,
            "score":               data["score"],
            "finish_time_seconds": data.get("finish_time_seconds", 0),
            "expected_rating":     round(expected_performance, 2),
            "old_rating":          round(old_rating, 2),
            "change":              round(final_delta, 2),
            "new_rating":          round(old_rating + final_delta, 2),
            "contest_count":       contest_count,
            "is_new":              (contest_count == 0),
        })

    # Sort by mean rank, then finish time for display
    rows.sort(key=lambda r: (r["actual_rank"], r["finish_time_seconds"]))
    return rows