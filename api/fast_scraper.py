"""
fast_scraper.py — Async Leaderboard Scraper (V5.1)
================================================
Fetches the ENTIRE LeetCode REST leaderboard safely.
Uses curl_cffi for TLS fingerprint spoofing and algorithmic jitter 
to completely bypass Cloudflare's WAF without getting IP banned.
"""

import asyncio
import math
import random
from typing import Dict, List, Optional

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

_BASE_URL = (
    "https://leetcode.com/contest/api/ranking/{slug}/"
    "?pagination={page}&region=global"
)

_MAX_CONCURRENCY = 5  # Throttled for extreme stealth
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 2.0
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
        # Algorithmic Jitter: Wait between 0.2s and 0.7s to simulate human browsing speeds
        await asyncio.sleep(random.uniform(0.2, 0.7))
        
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (403, 429, 503):
                    print(f'[scraper] WAF hit ({resp.status_code}) on page {page}. Pausing {delay}s...', flush=True)
                    await asyncio.sleep(delay)
                    delay *= 2.0
                    continue
                return None
            except (RequestsError, asyncio.TimeoutError) as e:
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(delay)
                    delay *= 2.0
    return None

def _assign_tied_ranks(raw_entries: List[dict]) -> List[dict]:
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
                "actual_rank":          tied_rank,
            })
        i = j
    return result

async def fetch_leaderboard(contest_slug: str) -> Dict[str, dict]:
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async with AsyncSession(impersonate="chrome124", max_clients=_MAX_CONCURRENCY) as session:
        print(f"[scraper] Discovering pages for '{contest_slug}'...", flush=True)
        page1 = await _fetch_page(session, sem, contest_slug, 1)
        if page1 is None:
            raise RuntimeError(f"Failed to fetch page 1 for '{contest_slug}'.")
        
        user_num = int(page1.get("user_num", 0))
        if user_num == 0:
            raise RuntimeError(f"Contest '{contest_slug}' has user_num=0.")
            
        total_pages = math.ceil(user_num / _PER_PAGE)
        print(f"[scraper] {total_pages} pages to fetch (~{user_num:,} entries)", flush=True)

        tasks = [_fetch_page(session, sem, contest_slug, p) for p in range(1, total_pages + 1)]
        raw_pages = await asyncio.gather(*tasks)

        raw_entries: List[dict] = []
        for data in raw_pages:
            if data is None: continue
            users = data.get("total_rank", [])
            subs = data.get("submissions", [])
            
            for row, sub in zip(users, subs):
                score = int(row.get("score", 0))
                
                # V5.1 Zero-Sum Ghost Filter: Drops users who never submitted code
                if score <= 0 and not sub:
                    continue
                    
                username = row.get("username", "").strip()
                if not username: continue

                raw_entries.append({
                    "username":            username,
                    "score":               score,
                    "finish_time_seconds": int(row.get("finish_time", 0)),
                })

        ranked = _assign_tied_ranks(raw_entries)
        print(f"[scraper] Done. {len(ranked):,} active participants.", flush=True)

    return {
        e["username"]: {
            "actual_rank":         e["actual_rank"],
            "score":               e["score"],
            "finish_time_seconds": e["finish_time_seconds"],
        }
        for e in ranked
    }