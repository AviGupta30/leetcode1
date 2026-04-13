"""
server.py — FastAPI Backend V3 (Direct old_rating from REST API)
================================================================
The LeetCode contest ranking REST API provides old_rating directly for
every participant. We capture it in fast_scraper.py and use it here.
No third-party snapshot needed for old_rating.

contest_count defaults to 0 for the mass-table (gives f(0)=0.5 multiplier).
For a single-user /api/predict call we fetch their real count via GraphQL.

Endpoints:
  GET  /api/contests
  GET  /api/leaderboard/{slug}?page&per_page&search
  GET  /api/leaderboard/{slug}/status
  POST /api/predict
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fast_scraper import fetch_leaderboard
from rating_math import predict, predict_all_users

app = FastAPI(title="LeetCode Predictor API V3")
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
_DEFAULT_RATING   = 1500.0
_DEFAULT_COUNT    = 0
_GRAPHQL_URL      = "https://leetcode.com/graphql/"
_GRAPHQL_HEADERS  = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com/contest/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# ---------------------------------------------------------------------------
# Global state: contest cache + scrape locks
# ---------------------------------------------------------------------------
contest_cache: Dict[str, Dict[str, Any]] = {}
contest_scrape_locks: Dict[str, asyncio.Lock] = {}
contest_scrape_status: Dict[str, str] = {}   # "scraping" | "done" | "error"


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


def _get_user_contest_count(username: str) -> int:
    """Fetch a user's attended contest count from LeetCode GraphQL."""
    data = _gql(_GQL_USER_RANKING, {"username": username})
    ranking = data.get("userContestRanking") or {}
    return int(ranking.get("attendedContestsCount") or 0)


# ---------------------------------------------------------------------------
# Build enriched dict from scraped leaderboard
# ---------------------------------------------------------------------------
def _build_enriched(leaderboard: Dict[str, dict]) -> tuple:
    """
    Build (rating_counts, enriched) from the live leaderboard.
    old_rating comes directly from the scraper (LeetCode REST API).
    contest_count defaults to 0 — accurate enough for the mass table.

    Returns
    -------
    rating_counts : Dict[int, float]   frequency map for math engine
    enriched      : Dict[str, dict]    per-user data for predictions
    """
    rating_counts: Dict[int, float] = defaultdict(float)
    enriched: Dict[str, dict] = {}

    for username, data in leaderboard.items():
        old_rating = float(data.get("old_rating") or _DEFAULT_RATING)
        if old_rating <= 0:
            old_rating = _DEFAULT_RATING

        rating_counts[round(old_rating)] += 1.0
        enriched[username] = {
            "actual_rank":          data["actual_rank"],
            "score":                data["score"],
            "finish_time_seconds":  data.get("finish_time_seconds", 0),
            "old_rating":           old_rating,
            "contest_count":        _DEFAULT_COUNT,   # 0 → f(0)=0.5 newcomer multiplier
        }

    return dict(rating_counts), enriched


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
        if slug in contest_cache:          # double-check after acquiring lock
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
                detail=f"No active participants found for '{slug}'."
            )

        rating_counts, enriched = _build_enriched(leaderboard)

        print(f"[server] Predicting for {len(enriched):,} users...", flush=True)
        rows = predict_all_users(enriched, rating_counts)

        elapsed = time.perf_counter() - t0
        print(
            f"[server] '{slug}' done in {elapsed:.1f}s. "
            f"{len(rows):,} rows ready.",
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
    First request scrapes (~45s). All subsequent requests are instant.
    """
    data = await _ensure_contest_data(slug)
    rows = data["rows"]

    if search and search.strip():
        term = search.strip().lower()
        rows = [r for r in rows if term in r["username"].lower()]

    total_filtered = len(rows)
    start = (page - 1) * per_page
    page_rows = rows[start : start + per_page]

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
    Single-user prediction.
    Fetches the user's real contest_count from GraphQL for accuracy.
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

    # Fetch real contest count from LeetCode for maximum accuracy
    real_count = _get_user_contest_count(canonical)

    result = predict(
        old_rating=entry["old_rating"],
        contest_count=real_count,
        actual_rank=int(entry["actual_rank"]),
        rating_counts=rating_counts,
    )
    result["username"]             = canonical
    result["contest_count"]        = real_count
    result["score"]                = entry["score"]
    result["finish_time_seconds"]  = entry.get("finish_time_seconds", 0)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
