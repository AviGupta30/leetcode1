"""
rating_math.py — LeetCode Rating Engine (V2 + Mass Prediction)
================================================================
Standard engine + a precomputed rank-table for O(log 4000) per-user
prediction when computing the full 25k-user leaderboard.

Official Constants: A = 5/18, C = 2/9, D = 0.0
Formula: final_delta = raw_delta * (A * 0.85^k + C)
"""

import math
from typing import Dict, List

_A: float = 5.0 / 18.0   # Decaying amplitude
_C: float = 2.0 / 9.0    # Veteran floor
_D: float = 0.0           # Anti-deflation (disabled)


# ---------------------------------------------------------------------------
# Core Elo Functions
# ---------------------------------------------------------------------------

def get_expected_rank(target_rating: float, rating_counts: Dict[int, float]) -> float:
    """E(R) = 0.5 + Σ count * P(opp beats R)  where P = 1/(1+10^((R-opp)/400))"""
    er = 0.5
    for opp, count in rating_counts.items():
        er += count / (1.0 + math.pow(10.0, (target_rating - float(opp)) / 400.0))
    return er


def precompute_rank_table(rating_counts: Dict[int, float]) -> List[float]:
    """
    Pre-compute expected_rank for every integer rating 0..4000.

    Cost: 4001 × n_buckets computations (one-time, ~0.5s for 2000 buckets).
    After this, each binary-search step is a free O(1) table lookup.

    Returns
    -------
    list of 4001 floats (table[r] = expected_rank at integer rating r).
    The list is monotonically DECREASING (higher rating → lower expected rank).
    """
    print("[math] Pre-computing rank table (4001 × {} buckets)…".format(len(rating_counts)), flush=True)
    table = [0.0] * 4001
    for r in range(4001):
        table[r] = get_expected_rank(float(r), rating_counts)
    return table


def _search_performance(target_rank: float, rank_table: List[float]) -> float:
    """
    Binary search on the precomputed rank_table to find the integer rating R
    whose expected rank ≈ target_rank.

    The table is decreasing, so:
      table[lo] >= target_rank > table[hi]  →  performance ≈ lo
    """
    lo, hi = 0, 4000
    while lo < hi:
        mid = (lo + hi) // 2
        if rank_table[mid] > target_rank:
            lo = mid + 1
        else:
            hi = mid
    return float(lo)


# ---------------------------------------------------------------------------
# Single-User Prediction (exact, for individual queries)
# ---------------------------------------------------------------------------

def predict(
    old_rating: float,
    contest_count: int,
    actual_rank: int,
    rating_counts: Dict[int, float],
) -> dict:
    """Exact single-user prediction using direct get_expected_rank calls."""
    binned_old = round(old_rating)
    prob_self = 1.0 / (1.0 + math.pow(10.0, (old_rating - float(binned_old)) / 400.0))

    expected_rank_full = get_expected_rank(old_rating, rating_counts)
    expected_rank_excl = expected_rank_full - prob_self

    geo_mean = math.sqrt(max(1.0, expected_rank_excl * float(actual_rank)))

    # Binary search for performance rating
    lo, hi = 0.0, 4000.0
    while hi - lo > 0.5:
        mid = (lo + hi) / 2.0
        if get_expected_rank(mid, rating_counts) > geo_mean:
            lo = mid
        else:
            hi = mid
    expected_performance = (lo + hi) / 2.0

    raw_delta = expected_performance - old_rating
    multiplier = _A * math.pow(0.85, max(0, contest_count)) + _C
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


# ---------------------------------------------------------------------------
# Mass Prediction (fast, for full leaderboard computation)
# ---------------------------------------------------------------------------

def predict_all_users(
    enriched: Dict[str, dict],
    rating_counts: Dict[int, float],
) -> List[dict]:
    """
    Compute predictions for EVERY user on the leaderboard efficiently.

    Strategy
    --------
    1. Precompute rank_table once  → O(4001 × buckets)
    2. For each user:
       a. Look up expected_rank via rank_table[round(old_rating)]  → O(1)
       b. Subtract self-comparison probability                     → O(1)
       c. Compute geometric mean rank                             → O(1)
       d. Binary search rank_table for performance rating          → O(log 4000)
    Total: O(4001×B + N×log4000)  ≈  0.5s + near-zero for 25k users.

    Returns
    -------
    List of row dicts sorted by actual_rank (ascending).
    """
    rank_table = precompute_rank_table(rating_counts)
    rows: List[dict] = []

    for username, data in enriched.items():
        old_rating    = data["old_rating"]
        contest_count = data["contest_count"]
        actual_rank   = data["actual_rank"]
        score         = data["score"]
        finish_time   = data.get("finish_time_seconds", 0)
        is_new        = (contest_count == 0)

        # --- Clamp old_rating to table bounds ---
        r_idx = max(0, min(4000, round(old_rating)))

        # --- Self-exclusion via nearest integer bin ---
        prob_self = 1.0 / (1.0 + math.pow(10.0, (old_rating - float(r_idx)) / 400.0))
        expected_rank_excl = rank_table[r_idx] - prob_self

        # --- Geometric mean rank ---
        geo_mean = math.sqrt(max(1.0, expected_rank_excl * float(actual_rank)))

        # --- Binary search on precomputed table ---
        expected_performance = _search_performance(geo_mean, rank_table)

        raw_delta = expected_performance - old_rating
        multiplier = _A * math.pow(0.85, max(0, contest_count)) + _C
        final_delta = raw_delta * multiplier

        rows.append({
            "username":            username,
            "actual_rank":         actual_rank,
            "score":               score,
            "finish_time_seconds": finish_time,
            "expected_rating":     round(expected_performance, 2),
            "old_rating":          round(old_rating, 2),
            "change":              round(final_delta, 2),
            "new_rating":          round(old_rating + final_delta, 2),
            "contest_count":       contest_count,
            "is_new":              is_new,
        })

    # Sort by actual_rank, then finish_time for equal ranks
    rows.sort(key=lambda r: (r["actual_rank"], r["finish_time_seconds"]))
    return rows
