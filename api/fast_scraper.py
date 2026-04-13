"""
fast_scraper.py — Async Leaderboard Scraper (V3)
================================================
Fetches the ENTIRE LeetCode REST leaderboard for a given contest loosely
mimicking a Chrome browser to bypass Cloudflare 403 errors using curl_cffi.
"""

import asyncio
import math
from typing import Dict, List, Optional

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

_BASE_URL = (
    "https://leetcode.com/contest/api/ranking/{slug}/"
    "?pagination={page}&region=global"
)

_MAX_CONCURRENCY = 15
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 1.0
_PER_PAGE = 25


async def _fetch_page(
    session: AsyncSession,
    sem: asyncio.Semaphore,
    slug: str,
    page: int,
) -> Optional[dict]:
    url = _BASE_URL.format(slug=slug, page=page)
    delay = _RETRY_BASE_DELAY
    async with sem:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (403, 429, 503):
                    print(f'[scraper] Rate limited ({resp.status_code}) on page {page}, attempt {attempt}. Retrying in {delay}s...', flush=True)
                    await asyncio.sleep(delay)
                    delay *= 2.0
                    continue
                return None
            except (RequestsError, asyncio.TimeoutError) as e:
                if attempt < _MAX_RETRIES:
                    print(f'[scraper] Network/Timeout error ({type(e).__name__}) on page {page}, attempt {attempt}. Retrying in {delay}s...', flush=True)
                    await asyncio.sleep(delay)
                    delay *= 2.0
    return None


def _assign_tied_ranks(raw_entries: List[dict]) -> List[dict]:
    """
    Sort by (score DESC, finish_time ASC) and assign statistical tied ranks.
    Tied rank = first_rank_in_group + (group_size - 1) / 2.0
    Ghost users (score == 0) are already excluded.
    """
    active = [e for e in raw_entries if e["score"] > 0]
    active.sort(key=lambda e: (-e["score"], e["finish_time_seconds"]))

    result: List[dict] = []
    i = 0
    while i < len(active):
        group_score = active[i]["score"]
        j = i
        while j < len(active) and active[j]["score"] == group_score:
            j += 1
        group_size = j - i
        first_rank = i + 1
        tied_rank: float = first_rank + (group_size - 1) / 2.0
        for entry in active[i:j]:
            result.append({
                "username":             entry["username"],
                "score":                entry["score"],
                "finish_time_seconds":  entry["finish_time_seconds"],
                "old_rating":           entry["old_rating"],
                "actual_rank":          tied_rank,
            })
        i = j

    return result


async def fetch_leaderboard(contest_slug: str) -> Dict[str, dict]:
    """
    Scrape the entire LeetCode contest leaderboard asynchronously using curl_cffi.
    """
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    # curl_cffi impersonates a browser connection to bypass Cloudflare
    async with AsyncSession(impersonate="chrome124", max_clients=_MAX_CONCURRENCY) as session:
        # ── Page 1: discover total page count ────────────────────────────────
        print(f"[scraper] Discovering pages for '{contest_slug}'...", flush=True)
        page1 = await _fetch_page(session, sem, contest_slug, 1)
        if page1 is None:
            raise RuntimeError(f"Failed to fetch page 1 for '{contest_slug}'.")
        user_num = int(page1.get("user_num", 0))
        if user_num == 0:
            raise RuntimeError(f"Contest '{contest_slug}' has user_num=0.")
        total_pages = math.ceil(user_num / _PER_PAGE)
        print(f"[scraper] {total_pages} pages to fetch (~{user_num:,} entries)", flush=True)

        # ── Fetch all pages concurrently ──────────────────────────────────────
        tasks = [_fetch_page(session, sem, contest_slug, p) for p in range(1, total_pages + 1)]
        raw_pages = await asyncio.gather(*tasks)

        # ── Parse all entries ─────────────────────────────────────────────────
        raw_entries: List[dict] = []
        failed = 0
        for data in raw_pages:
            if data is None:
                failed += 1
                continue
            users = data.get("total_rank", [])
            subs = data.get("submissions", [])
            
            for row, sub in zip(users, subs):
                score = int(row.get("score", 0))
                # Ghost user rules: score <= 0 AND zero code submissions. 
                # If they submitted wrong answers, `sub` is populated, so they are kept!
                if score <= 0 and not sub:
                    continue
                username = row.get("username", "").strip()
                if not username:
                    continue
                    
                old_rating_raw = row.get("old_rating") or row.get("rating") or 1500.0
                try:
                    old_rating = float(old_rating_raw)
                    if old_rating <= 0:
                        old_rating = 1500.0
                except (TypeError, ValueError):
                    old_rating = 1500.0

                raw_entries.append({
                    "username":            username,
                    "score":               score,
                    "finish_time_seconds": int(row.get("finish_time", 0)),
                    "old_rating":          old_rating,
                })

        if failed:
            print(f"[scraper] WARNING: {failed} pages failed.", flush=True)

        ranked = _assign_tied_ranks(raw_entries)
        print(f"[scraper] Done. {len(ranked):,} active participants.", flush=True)

    return {
        e["username"]: {
            "actual_rank":          e["actual_rank"],
            "score":                e["score"],
            "finish_time_seconds":  e["finish_time_seconds"],
            "old_rating":           e["old_rating"],
        }
        for e in ranked
    }
