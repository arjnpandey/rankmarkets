#!/usr/bin/env python3
"""
Fetches active prediction markets from Kalshi + Polymarket.
Categorizes events into 20 parent categories via Haiku.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CACHE_FILE = Path(__file__).parent / "data" / "events.json"
EMBEDDINGS_FILE = Path(__file__).parent / "data" / "embeddings.npy"
EMBED_MODEL = "gemini-embedding-001"
MIN_VOLUME = 1000

PARENT_CATEGORIES = [
    "US Elections",
    "Congress & Legislation",
    "White House & Executive",
    "Geopolitics & Foreign Policy",
    "Monetary Policy & Central Banks",
    "Economy & Labor",
    "Trade & Tariffs",
    "Cryptocurrency",
    "AI & Technology",
    "Climate & Energy",
    "Healthcare & Biotech",
    "Legal & Regulatory",
    "Defense & National Security",
    "Immigration",
    "Financial Markets",
    "Media & Tech Culture",
    "Science & Space",
    "Real Estate",
    "Society & Demographics",
    "Other",
]

EXCLUDED_CATEGORIES = {
    "sports", "esports", "basketball", "tennis", "nfl", "soccer", "nba", "nhl",
    "ufc", "hockey", "ncaa", "football", "mls", "formula 1", "chess", "darts",
    "games", "ncaa basketball", "call of duty", "cs2", "counter strike 2",
    "league of legends", "lol", "dota 2", "africa cup of nations", "la liga",
    "la liga 2", "premier league", "epl", "champions league", "efl cup",
    "coupe de france", "bundesliga", "world cup", "super bowl lx", "nfl draft",
    "cwbb", "parlays", "nfl playoffs", "pickleball", "rugby", "golf", "wta",
    "mlb", "ncaa football", "mma", "rugby six nations", "united rugby championship",
    "entertainment", "movies", "music", "awards", "golden globes", "oscars",
    "grammys", "grammy", "taylor swift", "mrbeast", "youtube", "celebrities",
    "netflix", "top netflix", "spotify", "video games", "pokemon", "avatar",
    "all-in", "taiki maeda", "reality tv", "eurovision",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_with_retry(url: str, retries: int = 3) -> requests.Response:
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 2.0 * (i + 1)
                print(f"    Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                raise RuntimeError(f"Server error {resp.status_code}")
            return resp
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to fetch {url}")


# ============== KALSHI ==============
def fetch_kalshi_events() -> list[dict]:
    KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
    all_events: list[dict] = []
    cursor = ""
    market_count = 0

    print("\n[Kalshi] Fetching active events...")

    try:
        while True:
            params = {
                "limit": "200",
                "status": "open",
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor

            resp = fetch_with_retry(f"{KALSHI_API}/events?{urlencode(params)}")
            if not resp.ok:
                break

            data = resp.json()
            new_cursor = data.get("cursor", "")
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor

            for event in data.get("events", []):
                markets = event.get("markets", [])
                nested_markets = []

                for m in markets:
                    status = m.get("status", "")
                    if status not in ("open", "active"):
                        continue

                    yes_price = 0.5
                    if m.get("yes_bid_dollars"):
                        yes_price = float(m["yes_bid_dollars"])
                    elif m.get("yes_bid"):
                        yes_price = m["yes_bid"] / 100

                    liquidity = 0.0
                    if m.get("liquidity_dollars"):
                        liquidity = float(m["liquidity_dollars"])
                    elif m.get("liquidity"):
                        liquidity = m["liquidity"] / 100

                    vol = m.get("volume", 0)
                    if vol < MIN_VOLUME:
                        continue

                    nested_markets.append({
                        "id": m.get("ticker", ""),
                        "title": m.get("title") or m.get("yes_sub_title") or event.get("title", ""),
                        "subtitle": m.get("yes_sub_title") or event.get("sub_title", "") or "",
                        "description": m.get("rules_primary", ""),
                        "yes_price": yes_price,
                        "volume": vol,
                        "volume24h": m.get("volume_24h", 0),
                        "liquidity": liquidity,
                        "endDate": m.get("expiration_time"),
                    })

                if nested_markets:
                    event_ticker = event.get("event_ticker") or event.get("ticker", "")
                    all_events.append({
                        "eventId": event_ticker,
                        "eventTitle": event.get("title", ""),
                        "source": "kalshi",
                        "category": event.get("category", "Uncategorized"),
                        "url": f"https://kalshi.com/markets/{event_ticker}",
                        "endDate": event.get("close_time"),
                        "nestedMarkets": nested_markets,
                    })
                    market_count += len(nested_markets)

            print(f"  Fetched: {len(all_events)} events ({market_count} markets)", end="\r")
            time.sleep(0.1)

    except Exception as e:
        print(f"\n[Kalshi] Error: {e}")

    print()
    return all_events


# ============== POLYMARKET ==============
def fetch_polymarket_events() -> list[dict]:
    POLY_API = "https://gamma-api.polymarket.com"
    all_events: list[dict] = []
    offset = 0
    limit = 100
    market_count = 0

    print("\n[Polymarket] Fetching active events...")

    try:
        while True:
            url = (
                f"{POLY_API}/events?closed=false&limit={limit}"
                f"&offset={offset}&order=liquidity&ascending=false"
            )
            resp = fetch_with_retry(url)
            if not resp.ok:
                print(f"[Polymarket] Failed: {resp.status_code}")
                break

            events = resp.json()
            if not events:
                break

            for event in events:
                nested_markets = []

                for m in event.get("markets", []):
                    if m.get("closed"):
                        continue

                    yes_price = 0.0
                    try:
                        prices = json.loads(m.get("outcomePrices", "[]"))
                        yes_price = float(prices[0]) if prices else 0.0
                    except (json.JSONDecodeError, IndexError, ValueError):
                        yes_price = 0.0

                    vol = m.get("volumeNum") or 0
                    if not vol:
                        try:
                            vol = float(m.get("volume", 0))
                        except (ValueError, TypeError):
                            vol = 0
                    if vol < MIN_VOLUME:
                        continue

                    liq = m.get("liquidityNum") or 0
                    if not liq:
                        try:
                            liq = float(m.get("liquidity", 0))
                        except (ValueError, TypeError):
                            liq = 0

                    nested_markets.append({
                        "id": m.get("slug") or m.get("id", ""),
                        "title": m.get("groupItemTitle") or m.get("question") or event.get("title", ""),
                        "subtitle": m.get("question", "") if m.get("groupItemTitle") else "",
                        "description": m.get("description") or event.get("description", "") or "",
                        "yes_price": yes_price,
                        "volume": vol,
                        "volume24h": m.get("volume24hr", 0),
                        "liquidity": liq,
                        "endDate": m.get("endDate"),
                    })

                if nested_markets:
                    slug = event.get("slug") or event.get("id", "")
                    tags = event.get("tags") or []
                    category = tags[0].get("label", "Uncategorized") if tags else "Uncategorized"
                    all_events.append({
                        "eventId": slug,
                        "eventTitle": event.get("title", ""),
                        "source": "polymarket",
                        "category": category,
                        "url": f"https://polymarket.com/event/{slug}",
                        "endDate": event.get("endDate"),
                        "nestedMarkets": nested_markets,
                    })
                    market_count += len(nested_markets)

            print(f"  Fetched: {len(all_events)} events ({market_count} markets)", end="\r")

            offset += limit
            if market_count > 10000:
                break
            time.sleep(0.1)

    except Exception as e:
        print(f"\n[Polymarket] Error: {e}")

    print()
    return all_events


# ============== GEMINI PRO CATEGORIZATION ==============
CATEGORIZE_MODEL = "gemini-3-pro-preview"

def categorize_events(events: list[dict]) -> None:
    """Batch-categorize events and add hedge use case via Gemini Pro. Modifies in-place."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Categorize] No GEMINI_API_KEY, skipping categorization")
        for e in events:
            e["parentCategory"] = "Other"
            e["hedgeCase"] = ""
        return

    client = genai.Client(api_key=api_key)
    cat_list = "\n".join(f"- {c}" for c in PARENT_CATEGORIES)
    BATCH_SIZE = 100  # smaller batches for Pro (smarter but slower)

    print(f"\n[Categorize] Assigning {len(events)} events to {len(PARENT_CATEGORIES)} categories + hedge cases...")

    valid_cats = set(PARENT_CATEGORIES)

    for i in range(0, len(events), BATCH_SIZE):
        batch = events[i : i + BATCH_SIZE]
        event_lines = "\n".join(
            f'{j}: {e["eventTitle"]}'
            for j, e in enumerate(batch)
        )

        try:
            resp = client.models.generate_content(
                model=CATEGORIZE_MODEL,
                contents=(
                    f"For each prediction market event below, provide:\n"
                    f"1. The best-fit category from this list:\n{cat_list}\n"
                    f"2. A short business hedging use case (one sentence: what kind of "
                    f"business or portfolio would use this market as a hedge, and against what risk?)\n\n"
                    f"Events:\n{event_lines}\n\n"
                    f"Return ONLY lines in this exact format:\n"
                    f"NUMBER|CATEGORY|HEDGE_CASE\n\n"
                    f"Example:\n"
                    f"0|AI & Technology|Tech companies could hedge against regulatory risk that limits AI deployment\n"
                    f"1|Economy & Labor|Retailers could hedge against consumer spending drops during a recession"
                ),
            )

            text = resp.text.strip()
            for line in text.split("\n"):
                line = line.strip()
                if "|" not in line:
                    continue
                parts = line.split("|", 2)
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0].strip())
                except ValueError:
                    continue
                cat = parts[1].strip()
                hedge = parts[2].strip() if len(parts) > 2 else ""
                if 0 <= idx < len(batch):
                    batch[idx]["parentCategory"] = cat if cat in valid_cats else "Other"
                    batch[idx]["hedgeCase"] = hedge

        except Exception as e:
            print(f"  Categorize batch error: {e}")

        # Default any uncategorized
        for e in batch:
            if "parentCategory" not in e:
                e["parentCategory"] = "Other"
            if "hedgeCase" not in e:
                e["hedgeCase"] = ""

        done = min(i + BATCH_SIZE, len(events))
        print(f"  Categorized: {done}/{len(events)}", end="\r")

    print(f"\n[Categorize] Done.")


def embed_events(events: list[dict], client) -> list[list[float]]:
    """Batch-embed all event titles. Returns list of 768-dim vectors."""
    titles = [e.get("eventTitle", "") for e in events]
    all_vectors = []
    BATCH = 100  # API max
    print(f"\n[Embed] Embedding {len(titles)} event titles...")
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i + BATCH]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        all_vectors.extend([e.values for e in result.embeddings])
        done = min(i + BATCH, len(titles))
        print(f"  Embedded: {done}/{len(titles)}", end="\r")
    print(f"\n[Embed] Done.")
    return all_vectors


def get_event_volume(event: dict) -> float:
    return sum(m.get("volume", 0) for m in event.get("nestedMarkets", []))


def count_markets(events: list[dict]) -> int:
    return sum(len(e.get("nestedMarkets", [])) for e in events)


def run_scraper() -> list[dict]:
    """Fetch all events, filter, categorize, sort, save, and return them."""
    print("=" * 60)
    print("Fetching Active Prediction Markets (Kalshi + Polymarket)")
    print(f"Volume Filter: >= ${MIN_VOLUME} per contract")
    print("=" * 60)

    # Fetch both sources in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        kalshi_future = pool.submit(fetch_kalshi_events)
        poly_future = pool.submit(fetch_polymarket_events)
        kalshi_events = kalshi_future.result()
        polymarket_events = poly_future.result()

    merged = kalshi_events + polymarket_events

    # Filter excluded categories
    all_events = [
        e for e in merged
        if e.get("category", "").lower() not in EXCLUDED_CATEGORIES
    ]
    excluded_count = len(merged) - len(all_events)
    print(f"\n[Filter] Excluded {excluded_count} sports/entertainment events")

    # Sort by total volume descending
    all_events.sort(key=lambda e: get_event_volume(e), reverse=True)

    # Categorize into 20 parent categories via Gemini Pro
    categorize_events(all_events)

    # Embed event titles for semantic search
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
        vectors = embed_events(all_events, client)
        EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_FILE, np.array(vectors, dtype=np.float32))
        print(f"  Saved embeddings to {EMBEDDINGS_FILE}")
    else:
        print("[Embed] No GEMINI_API_KEY, skipping embedding")

    # Build category index for quick lookup
    cat_index: dict[str, list[int]] = {}
    for i, e in enumerate(all_events):
        cat = e.get("parentCategory", "Other")
        cat_index.setdefault(cat, []).append(i)

    # Save
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    cache = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "kalshi": {"events": len(kalshi_events), "markets": count_markets(kalshi_events)},
            "polymarket": {"events": len(polymarket_events), "markets": count_markets(polymarket_events)},
            "total": {"events": len(all_events), "markets": count_markets(all_events)},
        },
        "categoryIndex": cat_index,
        "events": all_events,
    }

    CACHE_FILE.write_text(json.dumps(cache, indent=2))

    # Print category breakdown
    print(f"\nCategory breakdown:")
    for cat in PARENT_CATEGORIES:
        n = len(cat_index.get(cat, []))
        if n:
            print(f"  {cat}: {n}")

    print(f"\nSaved to {CACHE_FILE}")
    print(f"  Kalshi: {len(kalshi_events)} events ({count_markets(kalshi_events)} markets)")
    print(f"  Polymarket: {len(polymarket_events)} events ({count_markets(polymarket_events)} markets)")
    print(f"  Total: {len(all_events)} events ({count_markets(all_events)} markets)")

    return all_events


def load_cached_events() -> tuple[list[dict], dict[str, list[int]], str | None, np.ndarray | None]:
    """Load events + category index + embeddings from cache. Returns (events, cat_index, updatedAt, embeddings)."""
    if not CACHE_FILE.exists():
        return [], {}, None, None
    try:
        data = json.loads(CACHE_FILE.read_text())
        events = data.get("events", [])
        cat_index = data.get("categoryIndex", {})
        updated_at = data.get("updatedAt")
        embeddings = None
        if EMBEDDINGS_FILE.exists():
            embeddings = np.load(EMBEDDINGS_FILE)
            # Validate alignment
            if len(embeddings) != len(events):
                embeddings = None
        return events, cat_index, updated_at, embeddings
    except (json.JSONDecodeError, KeyError):
        return [], {}, None, None


if __name__ == "__main__":
    run_scraper()
