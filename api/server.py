"""
server.py — FastAPI Backend V5.2 (Production Pipeline)
=======================================================
V5.2 Changes (mathematical parity):
  1. REMOVED _static_estimator — no more "1450 Ghosting".
     Missing users get 1500.0 neutral placeholder; /api/predict ALWAYS
     performs a blocking GraphQL fetch for the queried user.
  2. Flexible /api/predict search (Req 1):
       Step 1 → case-insensitive handle match in live leaderboard
       Step 2 → Supabase leaderboards table (username + display_name ILIKE)
       Step 3 → GraphQL handle resolution as final fallback
  3. Background healer still warms the SQLite cache at 1 req/s after load.
"""

import asyncio
import math
import os
import time
from collections import defaultdict
from typing import Dict, Any, Optional

import aiosqlite
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client as SupabaseClient

from fast_scraper import fetch_leaderboard
from rating_math import predict, predict_all_users

# ─────────────────────────────────────────────────────────────────────────────
# APP & MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LeetCode Predictor API V5.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

_DB_FILE       = "leetcode_cache.db"
_GRAPHQL_URL   = "https://leetcode.com/graphql/"
_GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "Referer":      "https://leetcode.com/contest/",
    "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
}

# Supabase — prefer environment variables (GitHub Secrets)
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://iyrxbuarsuvsehrwemij.supabase.co")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_ZecAAYAgX256SgF_fvUcAw_ZghR1EeM")
supabase: SupabaseClient = create_client(_SUPABASE_URL, _SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SQLite WAL-MODE CACHE INIT
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_db():
    async with aiosqlite.connect(_DB_FILE) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username       TEXT PRIMARY KEY,
                rating         REAL,
                contest_count  INTEGER,
                last_updated   REAL
            )
        """)
        await db.commit()

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAPHQL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def _gql_user_data(username: str) -> tuple[Optional[float], Optional[int]]:
    """
    Blocking GraphQL fetch for a user's current rating and contest count.
    Returns (None, None) if the user doesn't exist or the request fails.
    This is intentionally synchronous — accuracy > latency for /api/predict.
    """
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            json={"query": _GQL_USER_RANKING, "variables": {"username": username}},
            headers=_GRAPHQL_HEADERS,
            timeout=8,
        )
        ranking = resp.json().get("data", {}).get("userContestRanking") or {}
        if ranking.get("rating") is not None:
            return float(ranking["rating"]), int(ranking.get("attendedContestsCount") or 0)
    except Exception as exc:
        print(f"[gql] Failed for '{username}': {exc}", flush=True)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 3. ASYNC HELPERS (Supabase resolve + SQLite cache write)
# ─────────────────────────────────────────────────────────────────────────────

async def _resolve_via_supabase(
    search_term: str,
    contest_slug: str,
    lower_map: Dict[str, str],
) -> Optional[str]:
    """
    Query Supabase `leaderboards` table for a matching row in this contest.
    Checks `username` (handle) first, then `display_name` — both case-insensitive.
    Returns the canonical key present in the live in-memory leaderboard, or None.
    """
    loop = asyncio.get_event_loop()

    def _query() -> Optional[str]:
        # ── Pass 1: match by handle ──────────────────────────────────────────
        res = (
            supabase.table("leaderboards")
            .select("username")
            .eq("contest_slug", contest_slug)
            .ilike("username", search_term)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["username"]

        # ── Pass 2: match by display_name ────────────────────────────────────
        try:
            res2 = (
                supabase.table("leaderboards")
                .select("username")
                .eq("contest_slug", contest_slug)
                .ilike("display_name", search_term)
                .limit(1)
                .execute()
            )
            if res2.data:
                return res2.data[0]["username"]
        except Exception:
            pass   # display_name column may not exist in older schema versions

        return None

    try:
        db_username = await loop.run_in_executor(None, _query)
        if db_username:
            # Map the resolved handle back into the live leaderboard
            return lower_map.get(db_username.lower()) or lower_map.get(db_username)
    except Exception as exc:
        print(f"[predict] Supabase resolve error: {exc}", flush=True)

    return None


async def _cache_user(username: str, rating: float, count: int) -> None:
    """Write a freshly-fetched rating into the local SQLite cache."""
    async with aiosqlite.connect(_DB_FILE) as db:
        await db.execute(
            "INSERT OR REPLACE INTO users (username, rating, contest_count, last_updated) "
            "VALUES (?, ?, ?, ?)",
            (username, rating, count, time.time()),
        )
        await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA-JOIN PIPELINE  (_build_enriched)
# ─────────────────────────────────────────────────────────────────────────────

async def _build_enriched(leaderboard: Dict[str, dict]) -> tuple:
    """
    Join every leaderboard entry with its historical rating from SQLite.

    V5.2: _static_estimator REMOVED.
    Missing / stale users receive a neutral 1500.0 placeholder so the
    rating_counts distribution is never poisoned by rank-guessed values.
    The background healer (Section 5) will replace placeholders within minutes
    at 1 GraphQL req/s, so subsequent calls will be accurate.
    """
    rating_counts: Dict[int, float] = defaultdict(float)
    enriched: Dict[str, dict] = {}
    usernames = list(leaderboard.keys())

    # ── 1. Bulk SQLite lookup ────────────────────────────────────────────────
    cached_data: Dict[str, dict] = {}
    async with aiosqlite.connect(_DB_FILE) as db:
        chunk_size = 900
        for i in range(0, len(usernames), chunk_size):
            chunk = usernames[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            async with db.execute(
                f"SELECT username, rating, contest_count, last_updated "
                f"FROM users WHERE username IN ({placeholders})",
                chunk,
            ) as cursor:
                async for row in cursor:
                    cached_data[row[0]] = {
                        "rating": row[1],
                        "count":  row[2],
                        "ts":     row[3],
                    }

    # ── 2. Build enriched dict ───────────────────────────────────────────────
    missing_or_stale: list[str] = []
    current_time = time.time()
    _STALE_SECONDS = 6 * 24 * 3600  # 6 days

    for username, data in leaderboard.items():
        cached   = cached_data.get(username)
        is_stale = cached is None or (current_time - cached["ts"]) > _STALE_SECONDS

        if is_stale:
            # V5.2: neutral placeholder — NOT a rank-based guess
            old_rating    = 1500.0
            contest_count = 0
            missing_or_stale.append(username)
        else:
            old_rating    = cached["rating"]
            contest_count = cached["count"]

        rating_counts[round(old_rating)] += 1.0
        enriched[username] = {
            "actual_rank":         data["actual_rank"],
            "score":               data["score"],
            "finish_time_seconds": data.get("finish_time_seconds", 0),
            "old_rating":          old_rating,
            "contest_count":       contest_count,
        }

    print(
        f"[server] DB hits: {len(cached_data):,} | "
        f"Missing/stale: {len(missing_or_stale):,} (placeholders @ 1500)",
        flush=True,
    )
    return dict(rating_counts), enriched, missing_or_stale


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKGROUND HEALER  (throttled GraphQL drip — fills the SQLite cache)
# ─────────────────────────────────────────────────────────────────────────────

async def _warm_cache_background(missing_users: list[str]) -> None:
    if not missing_users:
        return
    print(f"[cache] Healer starting for {len(missing_users):,} users...", flush=True)

    async with aiosqlite.connect(_DB_FILE) as db:
        loop = asyncio.get_event_loop()
        for username in missing_users:
            rating, count = await loop.run_in_executor(None, _gql_user_data, username)
            if rating is not None:
                await db.execute(
                    "INSERT OR REPLACE INTO users (username, rating, contest_count, last_updated) "
                    "VALUES (?, ?, ?, ?)",
                    (username, rating, count, time.time()),
                )
                await db.commit()
            # 1 req/s WAF protection
            await asyncio.sleep(1.0)

    print("[cache] Healer finished.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONTEST CACHE & SCRAPE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

contest_cache:         Dict[str, Dict[str, Any]] = {}
contest_scrape_locks:  Dict[str, asyncio.Lock]   = {}
contest_scrape_status: Dict[str, str]             = {}


def _get_lock(slug: str) -> asyncio.Lock:
    if slug not in contest_scrape_locks:
        contest_scrape_locks[slug] = asyncio.Lock()
    return contest_scrape_locks[slug]


async def _ensure_contest_data(slug: str) -> Dict[str, Any]:
    if slug in contest_cache:
        return contest_cache[slug]

    async with _get_lock(slug):
        if slug in contest_cache:
            return contest_cache[slug]

        contest_scrape_status[slug] = "scraping"
        print(f"[server] Scraping '{slug}'...", flush=True)

        try:
            leaderboard = await fetch_leaderboard(slug)
        except RuntimeError as exc:
            contest_scrape_status[slug] = "error"
            raise HTTPException(status_code=404, detail=str(exc))

        rating_counts, enriched, missing_users = await _build_enriched(leaderboard)
        rows = predict_all_users(enriched, rating_counts)

        contest_cache[slug] = {
            "rows":          rows,
            "total":         len(rows),
            "rating_counts": rating_counts,
            "enriched":      enriched,
        }
        contest_scrape_status[slug] = "done"

        # Fire background healer to replace 1500 placeholders
        asyncio.create_task(_warm_cache_background(missing_users))

    return contest_cache[slug]


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/contests")
async def get_contests():
    payload = {
        "query":     _GQL_PAST_CONTESTS,
        "variables": {"pageNo": 1, "numPerPage": 10},
    }
    try:
        resp = requests.post(_GRAPHQL_URL, json=payload, headers=_GRAPHQL_HEADERS, timeout=5)
        contests = resp.json().get("data", {}).get("pastContests", {}).get("data", [])
        return [{"title": c["title"], "slug": c["titleSlug"]} for c in contests]
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to fetch contests from GraphQL.")


@app.get("/api/leaderboard/{slug}")
async def get_leaderboard(
    slug:     str,
    page:     int           = Query(1, ge=1),
    per_page: int           = Query(50),
    search:   Optional[str] = None,
):
    data = await _ensure_contest_data(slug)
    rows = data["rows"]

    if search:
        term = search.strip().lower()
        rows = [r for r in rows if term in r["username"].lower()]

    start = (page - 1) * per_page
    return {
        "slug":        slug,
        "total":       len(rows),
        "page":        page,
        "per_page":    per_page,
        "total_pages": max(1, (len(rows) + per_page - 1) // per_page),
        "rows":        rows[start : start + per_page],
    }


# ─── /api/predict ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    username:     str
    contest_slug: str


@app.post("/api/predict")
async def predict_user(req: PredictRequest):
    """
    Predict rating delta for a single user.

    Req 1 — Flexible Search (3-step resolution):
      Step 1: Case-insensitive handle match in the live in-memory leaderboard.
      Step 2: Supabase leaderboards table — checks `username` ILIKE and
              `display_name` ILIKE so display names resolve to real handles.
      Step 3: GraphQL — confirms the term is a valid LeetCode handle and
              looks them up in the contest data.

    Req 4 — No more 1450 Ghosting:
      Always performs a blocking GraphQL fetch for the specific queried user
      before calculating, guaranteeing we use their true rating & contest count.
      If GraphQL is unreachable we fall back to the SQLite-cached value.
      Accuracy > latency.
    """
    data     = await _ensure_contest_data(req.contest_slug)
    enriched = data["enriched"]
    search   = req.username.strip()

    # Build a lowercase → canonical-key map from the live leaderboard
    lower_map: Dict[str, str] = {k.lower(): k for k in enriched}

    # ── Step 1: Direct case-insensitive handle match ──────────────────────────
    canonical: Optional[str] = lower_map.get(search.lower())

    # ── Step 2: Supabase display_name / username lookup (case-insensitive) ────
    if not canonical:
        canonical = await _resolve_via_supabase(search, req.contest_slug, lower_map)
        if canonical:
            print(f"[predict] Resolved '{search}' → '{canonical}' via Supabase.", flush=True)

    # ── Step 3: GraphQL handle resolution as last resort ─────────────────────
    if not canonical:
        # Attempt the search term as if it were a LeetCode handle
        gql_rating, _ = _gql_user_data(search)
        if gql_rating is not None:
            # Handle is valid on LeetCode — check if they participated
            canonical = lower_map.get(search.lower())
            if not canonical:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"User '{search}' exists on LeetCode but did not participate "
                        f"in '{req.contest_slug}'."
                    ),
                )
            print(f"[predict] Resolved '{search}' via GraphQL.", flush=True)

    if not canonical:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{search}' was not found in contest '{req.contest_slug}'. "
                "Check spelling or try your exact LeetCode handle."
            ),
        )

    entry = enriched[canonical]

    # ── Req 4: Blocking GraphQL fetch — accuracy over latency ─────────────────
    # This is the single most important fix: always use the true rating,
    # never the 1500 neutral placeholder for the target user.
    loop = asyncio.get_event_loop()
    live_rating, live_count = await loop.run_in_executor(None, _gql_user_data, canonical)

    if live_rating is not None:
        true_rating = live_rating
        true_count  = live_count
        # Persist for future leaderboard loads (don't await — fire and forget)
        asyncio.create_task(_cache_user(canonical, live_rating, live_count))
        print(
            f"[predict] '{canonical}': live rating={live_rating:.0f}, "
            f"contests={live_count}",
            flush=True,
        )
    else:
        # GraphQL unreachable — use whatever we have in the enriched entry.
        # If the user had a real cached rating it will be correct; if they were
        # a 1500 placeholder the prediction will carry some error but won't crash.
        true_rating = entry["old_rating"]
        true_count  = entry["contest_count"]
        print(
            f"[predict] WARNING: GraphQL unavailable for '{canonical}'. "
            f"Falling back to cached rating {true_rating:.0f}.",
            flush=True,
        )

    result = predict(
        true_rating,
        true_count,
        float(entry["actual_rank"]),
        data["rating_counts"],
    )
    result["username"] = canonical
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)