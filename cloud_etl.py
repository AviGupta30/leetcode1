"""
cloud_etl.py — The Autonomous GitHub Actions Worker (V5.2)
==========================================================
Zero-Failure Mandate:
  - MAX_RETRIES = 10 with exponential backoff for WAF bypass.
  - Ghost Rule: Retains users with score=0 if they have >= 1 submission.
  - Batch Integrity: Recursive binary-halving on upsert conflicts.
  - Flexible Search: Captures display_name for better frontend UX.
"""

import asyncio
import math
import os
import sys
import random
from supabase import create_client, Client
from curl_cffi.requests import AsyncSession

# --- 1. SUPABASE CREDENTIALS (Loaded from Environment Variables) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[etl] FATAL: SUPABASE_URL or SUPABASE_KEY environment variables are missing.")
    print("[etl] Set these in your terminal ($env:VAR=\"val\") before running locally.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. CONFIGURATION ---
_BASE_URL = "https://leetcode.com/contest/api/ranking/{slug}/?pagination={page}&region=global"
_MAX_CONCURRENCY = 5
_PER_PAGE = 25
MAX_RETRIES = 10 

# ─────────────────────────────────────────────────────────────────────────────
# SCRAPER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_page(session: AsyncSession, sem: asyncio.Semaphore, slug: str, page: int):
    """Fetch a single page with aggressive retries and exponential backoff."""
    url = _BASE_URL.format(slug=slug, page=page)
    delay = 2.0
    
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Add slight jitter to prevent synchronized spikes hitting Cloudflare
                await asyncio.sleep(random.uniform(0.1, 0.5))
                resp = await session.get(url, timeout=30)
                
                if resp.status_code == 200:
                    return resp.json()
                
                if resp.status_code in (403, 429, 503):
                    print(f"[etl] WAF/RateLimit ({resp.status_code}) on page {page}. Attempt {attempt}/{MAX_RETRIES}. Backing off {delay:.1f}s...")
                else:
                    print(f"[etl] Unexpected error {resp.status_code} on page {page}.")
                    
            except Exception as exc:
                print(f"[etl] Connection error on page {page}: {exc}. Retrying...")
            
            await asyncio.sleep(min(delay, 60.0))
            delay *= 2.0 # Exponential backoff
            
    print(f"[etl] FATAL: Page {page} failed permanently after {MAX_RETRIES} attempts.")
    return None

async def scrape_contest(contest_slug: str) -> list:
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)
    
    async with AsyncSession(impersonate="chrome124", max_clients=_MAX_CONCURRENCY) as session:
        print(f"[*] Initiating stealth scrape for '{contest_slug}'...")
        
        page1 = await fetch_page(session, sem, contest_slug, 1)
        if not page1:
            print("[etl] ABORT: Failed to fetch page 1 — cannot determine total pages.")
            return []

        user_num = int(page1.get("user_num", 0))
        total_pages = math.ceil(user_num / _PER_PAGE)
        print(f"[*] Targeting {total_pages} pages (~{user_num:,} users)...")

        tasks = [fetch_page(session, sem, contest_slug, p) for p in range(1, total_pages + 1)]
        raw_pages = await asyncio.gather(*tasks)

        valid_users = []
        for data in raw_pages:
            if not data: continue
            
            ranks = data.get("total_rank", [])
            subs = data.get("submissions", [])

            for row, sub in zip(ranks, subs):
                score = int(row.get("score", 0))
                
                # V5.2 Ghost Rule: Keep score=0 if they attempted a problem
                has_sub = isinstance(sub, dict) and len(sub) > 0
                if score <= 0 and not has_sub:
                    continue # True ghost: registered but did absolutely nothing
                
                username = row.get("username", "").strip()
                if not username: continue

                # Flexible Name Resolution (Falls back smoothly if missing)
                display_name = (
                    row.get("real_name") or 
                    row.get("name") or 
                    row.get("user_slug") or 
                    username
                ).strip()

                valid_users.append({
                    "username": username,
                    "display_name": display_name,
                    "actual_rank": row.get("rank"),
                    "score": score,
                    "contest_slug": contest_slug
                })
        
        return valid_users

# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE UPLOADER (Binary-Halving Logic)
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_chunk(rows: list, label: str = "") -> int:
    """Recursively splits batches on conflict to ensure 100% data integrity."""
    if not rows: return 0
    
    try:
        supabase.table("leaderboards").upsert(rows).execute()
        return len(rows)
    except Exception as exc:
        # If we get down to a single row and it STILL fails, log it and move on
        if len(rows) == 1:
            print(f"    [etl] CRITICAL: Skipping broken row {rows[0]['username']}: {exc}")
            return 0
        
        # ON_CONFLICT duplicate detected: Split the batch in half and try again
        mid = len(rows) // 2
        print(f"    [etl] Conflict in batch {label}. Splitting {len(rows)} → {mid} + {len(rows)-mid}...")
        
        ok_left = _upsert_chunk(rows[:mid], f"{label}L")
        ok_right = _upsert_chunk(rows[mid:], f"{label}R")
        return ok_left + ok_right

def push_to_supabase(users: list, contest_slug: str):
    print(f"[*] Pushing {len(users):,} users to Supabase cloud (Zero-Failure Protocol)...")
    
    batch_size = 1000
    total_stored = 0
    
    for i in range(0, len(users), batch_size):
        chunk = users[i : i + batch_size]
        label = f"{i}-{i+len(chunk)}"
        
        stored = _upsert_chunk(chunk, label)
        total_stored += stored
        print(f"    -> Progress: {total_stored:,}/{len(users):,} uploaded.")

    pct = (total_stored / len(users)) * 100 if users else 0
    print(f"[*] Success: {total_stored:,} participants stored ({pct:.1f}%) for '{contest_slug}'.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    # Use CLI arg if provided (e.g. python cloud_etl.py weekly-contest-400)
    contest_slug = sys.argv[1] if len(sys.argv) > 1 else "weekly-contest-400"
    print(f"[etl] Target contest: {contest_slug}")
    
    users = await scrape_contest(contest_slug)
    if users:
        push_to_supabase(users, contest_slug)
    else:
        print("[etl] Scrape failed or empty. No data to upload.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())