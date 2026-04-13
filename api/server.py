"""
server.py — FastAPI Backend V4 (Hybrid Rating Engine)
======================================================
Rating resolution pipeline (in priority order):

  1. LOCAL CACHE  — ratings_cache.json on disk. Grows after every contest.
                    Treats CN + Global users equally once they are seen once.
  2. GRAPHQL LIVE — LeetCode .com GraphQL for users not yet cached.
                    Resolves Global (US/EU/...) users accurately.
                    Falls through silently for CN users (returns null).
  3. RANK-ESTIMATOR — For users GraphQL cannot resolve (mainly CN Grandmasters).
                      Uses calibrated empirical LeetCode rating percentiles.
                      Far superior to a flat 1500 default.

After every contest scrape a background task runs to batch-fetch GraphQL
ratings for all participants and persist them to ratings_cache.json.
Over time the cache warms up and the estimator is used less and less.

Endpoints:
  GET  /api/contests
  GET  /api/leaderboard/{slug}?page&per_page&search
  GET  /api/leaderboard/{slug}/status
  POST /api/predict
"""

import asyncio
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fast_scraper import fetch_leaderboard
from rating_math import predict, predict_all_users

app = FastAPI(title="LeetCode Predictor API V4 — Hybrid")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_RATING       = 1500.0
_DEFAULT_COUNT        = 10      # veteran default — avoids rookies monopolising dampening
_GRAPHQL_URL          = "https://leetcode.com/graphql/"
_GRAPHQL_HEADERS      = {
    "Content-Type": "application/json",
    "Referer":      "https://leetcode.com/contest/",
    "User-Agent":   (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "ratings_cache.json")
# How many GraphQL calls to run concurrently when warming the cache
_GRAPHQL_CONCURRENCY = 20

# ---------------------------------------------------------------------------
# Rank-percentile → Rating Estimator
# ---------------------------------------------------------------------------
# Used ONLY for the fraction of users that the local cache + GraphQL cannot
# resolve (primarily LeetCode CN Grandmasters whose profiles are opaque to
# the .com GraphQL endpoint).
#
# Calibrated from empirical LeetCode active-participant distributions:
_RATING_PERCENTILE = [
    (0.0005, 3500.0),  # top 0.05%  — 1-3 users in a 25k contest
    (0.001,  3200.0),
    (0.003,  2900.0),
    (0.007,  2700.0),
    (0.015,  2500.0),
    (0.030,  2300.0),
    (0.055,  2100.0),
    (0.090,  2000.0),
    (0.130,  1900.0),
    (0.180,  1850.0),
    (0.230,  1800.0),
    (0.290,  1750.0),
    (0.360,  1700.0),
    (0.440,  1650.0),
    (0.520,  1600.0),
    (0.600,  1550.0),
    (0.680,  1500.0),
    (0.760,  1450.0),
    (0.840,  1400.0),
    (0.910,  1350.0),
    (0.960,  1300.0),
    (0.990,  1250.0),
    (1.000,  1200.0),
]


def _estimate_rating_from_rank(rank: float, total: int) -> float:
    """Map a contest rank to an estimated pre-contest rating via linear
    interpolation over the empirical LeetCode active-user distribution."""
    if total <= 0:
        return _DEFAULT_RATING
    pct = float(rank) / float(total)
    table = _RATING_PERCENTILE
    for i in range(1, len(table)):
        p_hi, r_hi = table[i]
        p_lo, r_lo = table[i - 1]
        if pct <= p_hi:
            t = (pct - p_lo) / (p_hi - p_lo + 1e-12)
            return r_lo + t * (r_hi - r_lo)
    return table[-1][1]


# ---------------------------------------------------------------------------
# Local ratings cache  (ratings_cache.json)
# ---------------------------------------------------------------------------
# Schema: { "username": {"rating": float, "count": int, "ts": float}, ... }
_ratings_cache: Dict[str, dict] = {}


def _load_cache() -> None:
    """Load ratings_cache.json from disk into memory at startup."""
    global _ratings_cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _ratings_cache = json.load(f)
            print(f"[cache] Loaded {len(_ratings_cache):,} cached user ratings from disk.", flush=True)
        except Exception as e:
            print(f"[cache] WARNING: Could not load cache ({e}). Starting fresh.", flush=True)
            _ratings_cache = {}
    else:
        print("[cache] No ratings_cache.json found — starting with empty cache.", flush=True)


def _save_cache() -> None:
    """Persist the in-memory cache to disk."""
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_ratings_cache, f, separators=(",", ":"))
        print(f"[cache] Saved {len(_ratings_cache):,} user ratings to disk.", flush=True)
    except Exception as e:
        print(f"[cache] WARNING: Could not save cache ({e}).", flush=True)


# Load on import (runs when uvicorn starts the module)
_load_cache()

# ---------------------------------------------------------------------------
# Global state: contest cache + scrape locks
# ---------------------------------------------------------------------------
contest_cache: Dict[str, Dict[str, Any]] = {}
contest_scrape_locks: Dict[str, asyncio.Lock] = {}
contest_scrape_status: Dict[str, str] = {}  # "scraping" | "done" | "error"


def _get_lock(slug: str) -> asyncio.Lock:
    if slug not in contest_scrape_locks:
        contest_scrape_locks[slug] = asyncio.Lock()
    return contest_scrape_locks[slug]


# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------
_GQL_USER_RANKING = """
query userContestRankingInfo($username: String!) {
    userContestRanking(username: $username) {
        rating
        attendedContestsCount
    }
}
"""

_GQL_PAST_CONTESTS = """
query pastContests($pageNo: Int, $numPerPage: Int) {
    pastContests(pageNo: $pageNo, numPerPage: $numPerPage) {
        data { title titleSlug startTime }
    }
}
"""


def _gql(query: str, variables: dict = None) -> dict:
    """Synchronous GraphQL call, returns the data dict."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            json=payload,
            headers=_GRAPHQL_HEADERS,
            timeout=10,
        )
        return resp.json().get("data") or {}
    except Exception:
        return {}


def _gql_user_data(username: str) -> tuple[Optional[float], Optional[int]]:
    """
    Fetch (rating, count) from LeetCode GraphQL.
    Returns (None, None) for CN users or network errors.
    """
    data = _gql(_GQL_USER_RANKING, {"username": username})
    ranking = data.get("userContestRanking") or {}
    rating = ranking.get("rating")
    count = ranking.get("attendedContestsCount")
    if rating is not None:
        return float(rating), int(count or 0)
    return None, None


# ---------------------------------------------------------------------------
# Build enriched dict — Hybrid pipeline
# ---------------------------------------------------------------------------
def _build_enriched(leaderboard: Dict[str, dict]) -> tuple:
    """
    Build (rating_counts, enriched) from the live leaderboard using the
    3-tier hybrid pipeline:

      Tier 1 — Local cache hit  (instant, covers previously seen users)
      Tier 2 — GraphQL resolve  (covers new global/US users in real-time)
      Tier 3 — Rank-percentile estimator  (CN users; no circular logic:
               we use their raw contest rank as reported by LeetCode's own
               server, not anything derived from our math engine)
    """
    rating_counts: Dict[int, float] = defaultdict(float)
    enriched: Dict[str, dict] = {}

    total = len(leaderboard)
    tier_hits = [0, 0, 0]  # cache, graphql, estimator

    for username, data in leaderboard.items():
        old_rating: Optional[float] = None
        contest_count: int = _DEFAULT_COUNT

        # ── Tier 1: local cache ─────────────────────────────────────────────
        cached = _ratings_cache.get(username)
        if cached:
            old_rating = float(cached["rating"])
            contest_count = int(cached.get("count", _DEFAULT_COUNT))
            tier_hits[0] += 1

        # ── Tier 2: GraphQL live resolve ────────────────────────────────────
        if old_rating is None:
            gql_rating, gql_count = _gql_user_data(username)
            if gql_rating is not None:
                old_rating = gql_rating
                contest_count = gql_count
                # Write-through to cache
                _ratings_cache[username] = {
                    "rating": old_rating,
                    "count":  contest_count,
                    "ts":     time.time(),
                }
                tier_hits[1] += 1

        # ── Tier 3: rank-percentile estimator ──────────────────────────────
        if old_rating is None:
            old_rating = _estimate_rating_from_rank(data["actual_rank"], total)
            contest_count = _DEFAULT_COUNT  # treat as seasoned player
            tier_hits[2] += 1

        rating_counts[round(old_rating)] += 1.0
        enriched[username] = {
            "actual_rank":          data["actual_rank"],
            "score":                data["score"],
            "finish_time_seconds":  data.get("finish_time_seconds", 0),
            "old_rating":           old_rating,
            "contest_count":        contest_count,
        }

    # ── Contamination telemetry ─────────────────────────────────────────────
    c, g, e = tier_hits
    print(
        f"[server] Pool: {total:,} | "
        f"Cache: {c:,} ({100*c//max(1,total)}%) | "
        f"GraphQL: {g:,} ({100*g//max(1,total)}%) | "
        f"Estimated: {e:,} ({100*e//max(1,total)}%)",
        flush=True,
    )

    return dict(rating_counts), enriched


# ---------------------------------------------------------------------------
# Background cache-warmer: batch-fetches GraphQL for all contest participants
# ---------------------------------------------------------------------------
async def _warm_cache_background(usernames: list[str]) -> None:
    """
    After a contest is fully scraped, fetch GraphQL ratings in background
    for any user not already in the cache. Runs concurrently, respects a
    semaphore to avoid hammering LeetCode's API.
    """
    missing = [u for u in usernames if u not in _ratings_cache]
    if not missing:
        return

    print(f"[cache] Warming: fetching GraphQL ratings for {len(missing):,} new users...", flush=True)
    sem = asyncio.Semaphore(_GRAPHQL_CONCURRENCY)
    resolved = 0

    async def _fetch_one(username: str) -> None:
        nonlocal resolved
        async with sem:
            # Run the sync GraphQL call in a thread pool to avoid blocking the loop
            loop = asyncio.get_event_loop()
            rating, count = await loop.run_in_executor(None, _gql_user_data, username)
            if rating is not None:
                _ratings_cache[username] = {
                    "rating": rating,
                    "count":  count,
                    "ts":     time.time(),
                }
                resolved += 1

    await asyncio.gather(*[_fetch_one(u) for u in missing])
    print(f"[cache] Warm-up done: {resolved:,}/{len(missing):,} new users cached.", flush=True)
    _save_cache()


# ---------------------------------------------------------------------------
# Core: ensure contest is scraped and all predictions computed
# ---------------------------------------------------------------------------
async def _ensure_contest_data(slug: str) -> Dict[str, Any]:
    """
    Returns cached data, running the scrape + mass-prediction if not done.
    The asyncio.Lock guarantees only ONE scrape per slug runs at a time.
    """
    if slug in contest_cache:
        return contest_cache[slug]

    lock = _get_lock(slug)
    async with lock:
        if slug in contest_cache:
            return contest_cache[slug]

        contest_scrape_status[slug] = "scraping"
        print(f"[server] Scraping '{slug}'...", flush=True)
        t0 = time.perf_counter()

        try:
            leaderboard = await fetch_leaderboard(slug)
        except RuntimeError as exc:
            contest_scrape_status[slug] = "error"
            raise HTTPException(status_code=404, detail=str(exc))

        if not leaderboard:
            contest_scrape_status[slug] = "error"
            raise HTTPException(
                status_code=404,
                detail=f"No active participants found for '{slug}'.",
            )

        rating_counts, enriched = _build_enriched(leaderboard)

        print(f"[server] Predicting for {len(enriched):,} users...", flush=True)
        rows = predict_all_users(enriched, rating_counts)

        elapsed = time.perf_counter() - t0
        print(
            f"[server] '{slug}' done in {elapsed:.1f}s. {len(rows):,} rows ready.",
            flush=True,
        )

        contest_cache[slug] = {
            "rows":          rows,
            "total":         len(rows),
            "scraped_at":    time.time(),
            "rating_counts": rating_counts,
            "enriched":      enriched,
        }
        contest_scrape_status[slug] = "done"

        # Fire-and-forget background cache warmer
        asyncio.create_task(_warm_cache_background(list(leaderboard.keys())))

    return contest_cache[slug]


# ---------------------------------------------------------------------------
# GET /api/contests
# ---------------------------------------------------------------------------
@app.get("/api/contests")
async def get_contests():
    """Return the 10 most recent LeetCode contests via GraphQL."""
    data = _gql(_GQL_PAST_CONTESTS, {"pageNo": 1, "numPerPage": 10})
    contests = data.get("pastContests", {}).get("data", [])
    if not contests:
        raise HTTPException(status_code=502, detail="Could not fetch contest list from LeetCode.")
    return [
        {"title": c["title"], "slug": c["titleSlug"], "start_time": c.get("startTime")}
        for c in contests
    ]


# ---------------------------------------------------------------------------
# GET /api/leaderboard/{slug}/status
# ---------------------------------------------------------------------------
@app.get("/api/leaderboard/{slug}/status")
async def leaderboard_status(slug: str):
    if slug in contest_cache:
        return {"status": "done", "total": contest_cache[slug]["total"]}
    return {"status": contest_scrape_status.get(slug, "not_started"), "total": 0}


# ---------------------------------------------------------------------------
# GET /api/leaderboard/{slug}
# ---------------------------------------------------------------------------
@app.get("/api/leaderboard/{slug}")
async def get_leaderboard(
    slug: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
):
    """
    Paginated, searchable full leaderboard with all predicted ratings.
    First request scrapes (~45–120s on rate-limited contests).
    All subsequent requests are instant from cache.
    """
    data = await _ensure_contest_data(slug)
    rows = data["rows"]

    if search and search.strip():
        term = search.strip().lower()
        rows = [r for r in rows if term in r["username"].lower()]

    total_filtered = len(rows)
    start = (page - 1) * per_page
    page_rows = rows[start: start + per_page]

    return {
        "slug":        slug,
        "total":       total_filtered,
        "page":        page,
        "per_page":    per_page,
        "total_pages": max(1, (total_filtered + per_page - 1) // per_page),
        "rows":        page_rows,
    }


# ---------------------------------------------------------------------------
# POST /api/predict  — single-user, exact prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    username: str
    contest_slug: str


@app.post("/api/predict")
async def predict_user(req: PredictRequest):
    """
    Single-user exact prediction.
    Fetches the user's live (rating, count) from GraphQL, overriding
    any cached/estimated value to guarantee the most accurate result.
    """
    data = await _ensure_contest_data(req.contest_slug)
    enriched = data["enriched"]
    rating_counts = data["rating_counts"]

    lower_map = {k.lower(): k for k in enriched}
    canonical = lower_map.get(req.username.lower())
    if not canonical:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{req.username}' is not on the leaderboard for '{req.contest_slug}'. "
                "They may have scored 0 (ghost) or did not participate."
            ),
        )

    entry = enriched[canonical]

    # Live GraphQL pull — bypasses stale cache & Biweekly collision
    live_rating, live_count = _gql_user_data(canonical)
    true_old_rating  = live_rating if live_rating is not None else entry["old_rating"]
    true_count       = live_count  if live_count  is not None else entry["contest_count"]

    result = predict(
        old_rating=true_old_rating,
        contest_count=true_count,
        actual_rank=int(entry["actual_rank"]),
        rating_counts=rating_counts,
    )
    result["username"]            = canonical
    result["contest_count"]       = true_count
    result["score"]               = entry["score"]
    result["finish_time_seconds"] = entry.get("finish_time_seconds", 0)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
