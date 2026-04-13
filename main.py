"""
main.py — Production Controller (V2)
=====================================
Ties together the community-history snapshot, the async leaderboard
scraper, and the rating-math engine into a single interactive CLI tool.

Workflow
--------
1. Download the community rating snapshot (≈300k users, ~5 MB JSON)
   from zerotrac's GitHub repo for pre-contest old_rating / contest_count.

2. Concurrently scrape the ENTIRE live leaderboard (~25k users) using
   fast_scraper.fetch_leaderboard().

3. Merge: for every live participant look them up in the snapshot;
   default to rating=1500.0 / count=0 if missing.

4. Build a rating_counts frequency map from the merged 25k participants.

5. Interactive loop: prompt for a target username, look up their rank
   from the live leaderboard, then call rating_math.predict() and print
   a formatted prediction card.

Usage
-----
    python main.py <contest_slug>

    Example:
        python main.py weekly-contest-400

Dependencies
------------
    pip install aiohttp requests
"""

import asyncio
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import requests

from fast_scraper import fetch_leaderboard
from rating_math import predict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_URL: str = (
    "https://raw.githubusercontent.com/zerotrac/leetcode_problem_rating/main/data.json"
)
_SNAPSHOT_TIMEOUT: int = 30          # seconds for the requests.get call
_DEFAULT_RATING: float = 1500.0
_DEFAULT_COUNT: int = 0

_BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║         LeetCode Rating Predictor  —  V2 (Fast REST)    ║
║         Piggyback + Async Leaderboard Architecture       ║
╚══════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Step 1 — Community snapshot
# ---------------------------------------------------------------------------

def fetch_community_snapshot(url: str = _SNAPSHOT_URL) -> Dict[str, dict]:
    """
    Download the zerotrac community JSON snapshot and return a fast lookup
    dictionary of the form:

        { "username": {"rating": float, "count": int} }

    The raw JSON is a list of objects.  We only care about the fields
    ``UserSlug`` (username), ``Rating``, and ``ContestCount``.

    On any network / parse error the function prints a warning and
    returns an empty dict (predictions will fall back to 1500 / 0).
    """
    print("[main] Downloading community rating snapshot … ", end="", flush=True)
    t0 = time.perf_counter()

    try:
        resp = requests.get(url, timeout=_SNAPSHOT_TIMEOUT)
        resp.raise_for_status()
        raw: list = resp.json()
    except requests.RequestException as exc:
        print(f"\n[main] WARNING: snapshot fetch failed ({exc}). "
              "All missing users will default to 1500 / 0.", flush=True)
        return {}
    except ValueError as exc:
        print(f"\n[main] WARNING: snapshot JSON parse error ({exc}). "
              "Continuing with empty snapshot.", flush=True)
        return {}

    elapsed = time.perf_counter() - t0
    lookup: Dict[str, dict] = {}

    for entry in raw:
        username: str = (entry.get("UserSlug") or entry.get("user_slug") or "").strip()
        if not username:
            continue
        try:
            rating = float(entry.get("Rating") or entry.get("rating") or _DEFAULT_RATING)
            count  = int(entry.get("ContestCount") or entry.get("contest_count") or _DEFAULT_COUNT)
        except (TypeError, ValueError):
            rating = _DEFAULT_RATING
            count  = _DEFAULT_COUNT

        lookup[username] = {"rating": rating, "count": count}

    print(f"done.  {len(lookup):,} users loaded in {elapsed:.2f}s", flush=True)
    return lookup


# ---------------------------------------------------------------------------
# Step 2 — Merge snapshot + live leaderboard → rating_counts map
# ---------------------------------------------------------------------------

def build_rating_counts(
    leaderboard: Dict[str, dict],
    snapshot: Dict[str, dict],
) -> Tuple[Dict[int, float], Dict[str, dict]]:
    """
    Merge the live leaderboard with the community snapshot.

    For every active participant (score > 0, already filtered by fast_scraper):
      - Look up their pre-contest rating and contest count from the snapshot.
      - Default to (1500.0, 0) if not found.

    Returns
    -------
    rating_counts : Dict[int, float]
        Frequency map {binned_rating: participant_count}.

    enriched_leaderboard : Dict[str, dict]
        Leaderboard dict augmented with "old_rating" and "contest_count"
        for every participant.  Shape per entry:
            {
                "actual_rank":   float,
                "score":         int,
                "old_rating":    float,
                "contest_count": int,
            }
    """
    rating_counts: Dict[int, float] = defaultdict(float)
    enriched: Dict[str, dict] = {}

    missing_in_snapshot = 0

    for username, data in leaderboard.items():
        snap = snapshot.get(username)
        if snap is None:
            missing_in_snapshot += 1
            old_rating    = _DEFAULT_RATING
            contest_count = _DEFAULT_COUNT
        else:
            old_rating    = snap["rating"]
            contest_count = snap["count"]

        binned = round(old_rating)
        rating_counts[binned] += 1.0

        enriched[username] = {
            "actual_rank":   data["actual_rank"],
            "score":         data["score"],
            "old_rating":    old_rating,
            "contest_count": contest_count,
        }

    if missing_in_snapshot:
        pct = 100.0 * missing_in_snapshot / max(1, len(leaderboard))
        print(
            f"[main] Snapshot miss: {missing_in_snapshot:,} / {len(leaderboard):,} "
            f"participants ({pct:.1f}%) defaulted to rating=1500 / count=0.",
            flush=True,
        )

    return dict(rating_counts), enriched


# ---------------------------------------------------------------------------
# Step 3 — Interactive prediction loop
# ---------------------------------------------------------------------------

def _print_prediction_card(username: str, result: dict) -> None:
    """Pretty-print a single prediction result to the terminal."""
    delta = result["final_delta"]
    sign  = "+" if delta >= 0 else ""
    new_r = result["predicted_new_rating"]

    print()
    print("┌─────────────────────────────────────────────────┐")
    print(f"│  User            : {username:<29}│")
    print(f"│  Old Rating      : {result['old_rating']:<29.4f}│")
    print(f"│  Contest #       : {result['contest_count']:<29}│")
    print(f"│  Actual Rank     : {result['actual_rank']:<29}│")
    print("├─────────────────────────────────────────────────┤")
    print(f"│  Expected Perf.  : {result['expected_performance']:<29.4f}│")
    print(f"│  Raw Delta       : {result['raw_delta']:<+29.4f}│")
    print(f"│  Final Delta     : {sign}{delta:<28.4f}│")
    print(f"│  Predicted New   : {new_r:<29.4f}│")
    print("└─────────────────────────────────────────────────┘")
    print()


def interactive_loop(
    enriched_leaderboard: Dict[str, dict],
    rating_counts: Dict[int, float],
) -> None:
    """
    Prompt the user for a target username in a loop, run the prediction
    engine, and display the result.  Type 'quit' or press Ctrl-C to exit.
    """
    print("\n[main] Prediction engine ready.")
    print("[main] Type a LeetCode username to predict their rating change.")
    print("[main] Type 'quit' or press Ctrl-C to exit.\n")

    while True:
        try:
            raw = input("  Enter username: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Exiting. Goodbye!")
            break

        if raw.lower() in {"quit", "exit", "q"}:
            print("[main] Exiting. Goodbye!")
            break

        if not raw:
            continue

        username = raw

        entry = enriched_leaderboard.get(username)
        if entry is None:
            # Case-insensitive fallback
            lower_map = {k.lower(): k for k in enriched_leaderboard}
            canonical = lower_map.get(username.lower())
            if canonical:
                entry    = enriched_leaderboard[canonical]
                username = canonical
                print(f"  [hint] Matched '{username}' (case-insensitive).")
            else:
                print(
                    f"  [!] '{username}' was not found on the leaderboard.\n"
                    "      They may have scored 0 (ghost), not participated, "
                    "or the username is misspelled."
                )
                continue

        result = predict(
            old_rating    = entry["old_rating"],
            contest_count = entry["contest_count"],
            actual_rank   = int(entry["actual_rank"]),
            rating_counts = rating_counts,
        )

        _print_prediction_card(username, result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _async_main(contest_slug: str) -> None:
    print(_BANNER)
    print(f"[main] Contest slug : {contest_slug}")
    print(f"[main] Snapshot URL : {_SNAPSHOT_URL}\n")

    # ── Step 1: Community snapshot (blocking, but fast — ~0.5s on broadband)
    snapshot = fetch_community_snapshot()

    # ── Step 2: Full async leaderboard scrape
    print(f"\n[main] Scraping live leaderboard … (this may take 30–45 s)", flush=True)
    t0 = time.perf_counter()
    leaderboard = await fetch_leaderboard(contest_slug)
    elapsed = time.perf_counter() - t0
    print(
        f"[main] Leaderboard fetched: {len(leaderboard):,} active participants "
        f"in {elapsed:.1f}s",
        flush=True,
    )

    if not leaderboard:
        print(
            "[main] ERROR: leaderboard is empty.  "
            "Check the contest slug and try again."
        )
        sys.exit(1)

    # ── Step 3: Merge & build frequency map
    print("\n[main] Merging snapshot with live leaderboard …", flush=True)
    rating_counts, enriched = build_rating_counts(leaderboard, snapshot)
    print(
        f"[main] Frequency map built: {len(rating_counts):,} distinct rating buckets.",
        flush=True,
    )

    # ── Step 4: Interactive prediction
    interactive_loop(enriched, rating_counts)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:  python main.py <contest-slug>")
        print("Example: python main.py weekly-contest-400")
        sys.exit(1)

    contest_slug: str = sys.argv[1].strip()

    # Use asyncio.run() for a clean event loop
    try:
        asyncio.run(_async_main(contest_slug))
    except KeyboardInterrupt:
        print("\n[main] Interrupted. Goodbye!")


if __name__ == "__main__":
    main()
