"""
Fast prediction market search + enterprise risk analysis.

Market search pipeline:
  1. Local keyword match → pick 3-5 categories           (~0ms)
  2. Gemini Flash → expand query into 5 search keywords   (~1s)
  3. Embed keywords (batch) → cosine sim → ~50 candidates  (~100ms)
  4. Gemini Flash → rank candidates → top 20               (~1-2s)

Enterprise analysis pipeline:
  1. Gemini Flash → risk analysis + search keywords        (~2-3s)
  2. Local category match + embed keywords → ~50 cands      (~100ms)
  3. Gemini Flash → rank existing markets from scraped data (~2-3s)
  4. Gemini Pro  → suggest new markets that should exist    (~3-5s)

Total: ~2-3s market search, ~5-7s enterprise analysis.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3-flash-preview"
PRO_MODEL = "gemini-3-pro-preview"
EMBED_MODEL = "gemini-embedding-001"
MAX_RESULTS = 20
KEYWORDS_PER_QUERY = 5
EVENTS_PER_KEYWORD = 10
CACHE_DIR = Path(__file__).parent / "data" / "query_cache"

# ── Category keyword map for instant local matching ───────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "US Elections": [
        "election", "vote", "voter", "ballot", "candidate", "primary", "poll",
        "campaign", "gubernatorial", "presidential", "democrat", "republican",
        "gop", "dnc", "rnc", "swing state", "electoral", "midterm",
    ],
    "Congress & Legislation": [
        "congress", "senate", "house", "bill", "legislation", "shutdown",
        "debt ceiling", "filibuster", "speaker", "majority", "bipartisan",
        "reconciliation", "appropriation", "tax", "tax reform", "tax cut",
        "tax hike", "billionaire", "wealth tax", "capital gains", "estate tax",
        "irs", "budget", "deficit", "fiscal", "spending bill", "revenue",
    ],
    "White House & Executive": [
        "president", "white house", "executive order", "cabinet", "veto",
        "impeach", "pardon", "biden", "trump", "administration", "oval",
    ],
    "Geopolitics & Foreign Policy": [
        "war", "conflict", "russia", "ukraine", "china", "taiwan", "iran",
        "israel", "gaza", "nato", "sanction", "diplomat", "treaty",
        "geopolit", "invasion", "cease", "peace", "north korea", "india",
        "europe", "eu ", "brexit", "un ", "united nations",
    ],
    "Monetary Policy & Central Banks": [
        "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
        "fomc", "monetary", "basis point", "bps", "powell", "dovish",
        "hawkish", "quantitative", "tightening", "easing",
    ],
    "Economy & Labor": [
        "gdp", "inflation", "cpi", "pce", "unemployment", "recession",
        "jobs", "nonfarm", "payroll", "economic", "economy", "growth",
        "consumer", "spending", "retail", "wage", "tax", "income tax",
        "corporate tax", "fiscal policy", "deficit", "national debt",
        "inequality", "wealth", "billionaire",
    ],
    "Trade & Tariffs": [
        "tariff", "trade war", "import", "export", "trade deal", "customs",
        "duty", "trade policy", "wto", "nafta", "usmca", "protectionism",
    ],
    "Cryptocurrency": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "token", "defi",
        "blockchain", "stablecoin", "altcoin", "solana", "sol", "nft",
        "binance", "coinbase", "memecoin",
    ],
    "AI & Technology": [
        "ai", "artificial intelligence", "openai", "chatgpt", "gpt", "llm",
        "machine learning", "deepmind", "anthropic", "semiconductor", "chip",
        "nvidia", "tsmc", "robot", "autonomous", "apple", "google", "meta",
        "microsoft", "tech", "software", "saas",
    ],
    "Climate & Energy": [
        "climate", "carbon", "emission", "renewable", "solar", "wind",
        "oil", "gas", "opec", "energy", "temperature", "warming", "epa",
        "ev", "electric vehicle", "battery", "nuclear", "fossil",
    ],
    "Healthcare & Biotech": [
        "fda", "drug", "vaccine", "pharma", "biotech", "health", "pandemic",
        "covid", "medical", "hospital", "insurance", "medicare", "medicaid",
        "clinical trial", "approval",
    ],
    "Legal & Regulatory": [
        "supreme court", "scotus", "lawsuit", "ruling", "antitrust",
        "regulation", "regulatory", "court", "judge", "legal", "doj",
        "ftc", "sec", "indictment", "trial", "verdict",
    ],
    "Defense & National Security": [
        "military", "defense", "pentagon", "army", "navy", "air force",
        "missile", "nuclear", "cybersecurity", "intelligence", "cia", "nsa",
        "security", "weapon", "drone",
    ],
    "Immigration": [
        "immigration", "border", "visa", "asylum", "deportation", "migrant",
        "refugee", "daca", "ice", "customs",
    ],
    "Financial Markets": [
        "stock", "s&p", "nasdaq", "dow", "bond", "yield", "treasury",
        "ipo", "merger", "acquisition", "equity", "index", "commodit",
        "gold", "silver", "futures", "option", "hedge",
    ],
    "Media & Tech Culture": [
        "social media", "twitter", "tiktok", "facebook", "instagram",
        "youtube", "influencer", "content", "streaming", "platform",
        "elon musk", "zuckerberg",
    ],
    "Science & Space": [
        "nasa", "spacex", "space", "rocket", "mars", "moon", "satellite",
        "asteroid", "scientific", "discovery", "research", "physics",
    ],
    "Real Estate": [
        "housing", "real estate", "mortgage", "rent", "home price",
        "construction", "property", "commercial real estate", "reit",
    ],
    "Society & Demographics": [
        "population", "census", "demographic", "social", "religion",
        "culture", "public opinion", "polling", "gender", "education",
        "university", "student loan",
    ],
    "Other": [],
}


# ── Local category matching ───────────────────────────────────────────

def match_categories(query: str, top_n: int = 10, min_n: int = 5) -> list[str]:
    q = query.lower()
    scores: dict[str, float] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if cat == "Other":
            continue
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[cat] = score

    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_n]

    # Pad to at least min_n with remaining categories (in definition order)
    if len(ranked) < min_n:
        chosen = set(ranked)
        for cat in CATEGORY_KEYWORDS:
            if cat != "Other" and cat not in chosen:
                ranked.append(cat)
                if len(ranked) >= min_n:
                    break

    return ranked


# ── Gemini Flash helpers ──────────────────────────────────────────────

def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment or .env")
    return genai.Client(api_key=api_key)


def _flash_call(client: genai.Client, prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=prompt,
    )
    return resp.text.strip()


def _pro_call(client: genai.Client, prompt: str) -> str:
    resp = client.models.generate_content(
        model=PRO_MODEL,
        contents=prompt,
    )
    return resp.text.strip()


def _expand_keywords(client: genai.Client, query: str) -> list[str]:
    """Ask Flash to generate 5 diverse search keywords from a user query."""
    text = _flash_call(client, (
        f'A user is searching prediction markets for: "{query}"\n\n'
        f"Generate exactly {KEYWORDS_PER_QUERY} diverse search keywords or short phrases "
        f"that would help find relevant markets. Cover different angles and related topics.\n\n"
        f"Return ONLY one keyword/phrase per line, nothing else."
    ))
    keywords = [line.strip().strip("-•*").strip() for line in text.split("\n") if line.strip()]
    return keywords[:KEYWORDS_PER_QUERY]


def _rank_candidates(client: genai.Client, query: str, events: list[dict]) -> list[tuple[int, str]]:
    """Ask Flash to rank candidate events and explain relevance. Returns [(index, reason)]."""
    catalog = "\n".join(
        f'{i}: {e.get("eventTitle", "")}'
        for i, e in enumerate(events)
    )
    text = _flash_call(client, (
        f'User query: "{query}"\n\n'
        f"Candidate prediction markets:\n{catalog}\n\n"
        f"Rank the {MAX_RESULTS} most relevant markets for this query. "
        f"For each, give a short explanation of why it's relevant.\n"
        f"Return ONLY lines in format: NUMBER|EXPLANATION\n"
        f"Most relevant first."
    ))
    results = []
    seen: set[int] = set()
    for line in text.split("\n"):
        line = line.strip()
        if "|" not in line:
            continue
        parts = line.split("|", 1)
        try:
            idx = int(parts[0].strip().strip(".)"))
        except ValueError:
            continue
        reason = parts[1].strip() if len(parts) > 1 else ""
        if 0 <= idx < len(events) and idx not in seen:
            seen.add(idx)
            results.append((idx, reason))
    return results[:MAX_RESULTS]


# ── Embedding-based ranking ──────────────────────────────────────────

def _embed_texts(client: genai.Client, texts: list[str], task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    """Embed one or more texts in a single API call. Returns (N, 768) array."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    return np.array([e.values for e in result.embeddings], dtype=np.float32)


def _embed_rank(client: genai.Client, query_vec: np.ndarray, events: list[dict],
                embeddings: np.ndarray, top_n: int,
                candidate_mask: np.ndarray | None = None) -> list[dict]:
    """Rank events by cosine similarity to a pre-computed query embedding."""
    if candidate_mask is not None:
        pool_embs = embeddings[candidate_mask]
        pool_indices = np.where(candidate_mask)[0]
    else:
        pool_embs = embeddings
        pool_indices = np.arange(len(embeddings))

    if len(pool_embs) == 0:
        return []

    # Cosine similarity (normalize to be safe)
    norms = np.linalg.norm(pool_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    pool_normed = pool_embs / norms
    q_norm = np.linalg.norm(query_vec)
    q_normed = query_vec / (q_norm if q_norm > 0 else 1)

    scores = pool_normed @ q_normed
    top_idx = np.argsort(scores)[::-1][:top_n]

    return [events[pool_indices[i]] for i in top_idx]


# ── Result caching ────────────────────────────────────────────────────

def _cache_key(query: str, n_events: int) -> str:
    return hashlib.md5(f"{query.strip().lower()}|{n_events}".encode()).hexdigest()


def _load_cache(key: str) -> tuple[list[str], list[str], list[dict]] | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data["cats"], data["keywords"], data["results"]
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_cache(key: str, cats: list[str], keywords: list[str], results: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps({"cats": cats, "keywords": keywords, "results": results}, indent=1)
    )


# ── Main entry point ─────────────────────────────────────────────────

def find_relevant_markets(
    query: str,
    events: list[dict],
    cat_index: dict[str, list[int]],
    embeddings: np.ndarray | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Find top 20 relevant markets.

    Returns (matched_categories, search_keywords, results).

    Pipeline:
      1. Local keyword match → 3-5 categories
      2. Flash → 5 search keywords
      3. Embed keywords (batch) → cosine sim → ~50 candidates
      4. Flash → rank → top 20
    """
    key = _cache_key(query, len(events))
    cached = _load_cache(key)
    if cached is not None:
        return cached

    client = _get_client()

    # Step 1: pick categories locally (pre-filter)
    matched_cats = match_categories(query)

    # Build candidate mask for category pre-filter
    candidate_mask: np.ndarray | None = None
    if embeddings is not None:
        candidate_indices: set[int] = set()
        for cat in matched_cats:
            candidate_indices.update(cat_index.get(cat, []))
        if candidate_indices:
            mask = np.zeros(len(events), dtype=bool)
            for idx in candidate_indices:
                if idx < len(events):
                    mask[idx] = True
            candidate_mask = mask

    # Step 2: expand query into 5 keywords via Flash
    keywords = _expand_keywords(client, query)

    # Step 3: embed all keywords + query in one batch, cosine sim, deduplicate
    if embeddings is not None:
        all_texts = keywords + [query]
        all_vecs = _embed_texts(client, all_texts, task_type="RETRIEVAL_QUERY")

        seen_ids: set[str] = set()
        candidates: list[dict] = []

        for vec in all_vecs:
            top = _embed_rank(client, vec, events, embeddings, EVENTS_PER_KEYWORD, candidate_mask)
            for e in top:
                eid = e.get("eventId", "")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    candidates.append(e)
    else:
        # Fallback: no embeddings, return all events from matched categories
        candidate_indices_set: set[int] = set()
        for cat in matched_cats:
            candidate_indices_set.update(cat_index.get(cat, []))
        if candidate_indices_set:
            candidates = [events[i] for i in sorted(candidate_indices_set) if i < len(events)]
        else:
            candidates = events

    # Step 4: Flash ranks the candidates and explains
    if len(candidates) <= MAX_RESULTS:
        results = [{"event": e, "reason": ""} for e in candidates]
    else:
        ranked = _rank_candidates(client, query, candidates)
        results = [{"event": candidates[idx], "reason": reason} for idx, reason in ranked]

    _save_cache(key, matched_cats, keywords, results)
    return matched_cats, keywords, results


# ── Enterprise analysis ──────────────────────────────────────────────

INDUSTRIES = [
    "Technology & Software",
    "Financial Services & Banking",
    "Healthcare & Pharmaceuticals",
    "Energy & Utilities",
    "Manufacturing & Industrial",
    "Retail & Consumer Goods",
    "Real Estate & Construction",
    "Transportation & Logistics",
    "Agriculture & Food",
    "Telecommunications",
    "Media & Entertainment",
    "Education",
    "Defense & Aerospace",
    "Mining & Natural Resources",
    "Professional Services",
    "Hospitality & Tourism",
]

GEOGRAPHIES = [
    "US — National",
    "US — Northeast",
    "US — Southeast",
    "US — Midwest",
    "US — West Coast",
    "US — Southwest",
    "Canada",
    "Europe (EU/UK)",
    "China",
    "India",
    "Japan / South Korea",
    "Latin America",
    "Middle East & Africa",
    "Southeast Asia / Oceania",
]

COST_INPUTS = [
    "Labor & wages",
    "Raw materials",
    "Energy & fuel",
    "Imported goods / components",
    "Technology / cloud infrastructure",
    "Real estate / rent",
    "Logistics & shipping",
]

REVENUE_STREAMS = [
    "Domestic B2B",
    "Domestic B2C",
    "Export / international sales",
    "Government contracts",
    "Subscriptions / SaaS",
    "Licensing / royalties",
]

REVENUE_TICKS = [
    "$1M", "$5M", "$10M", "$25M", "$50M",
    "$100M", "$250M", "$500M", "$1B", "$5B", "$10B+",
]


def _enterprise_cache_key(profile: dict) -> str:
    raw = json.dumps(profile, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _load_enterprise_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"ent_{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_enterprise_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"ent_{key}.json").write_text(json.dumps(data, indent=1))


def analyze_enterprise(
    profile: dict,
    events: list[dict],
    cat_index: dict[str, list[int]],
    embeddings: np.ndarray | None = None,
) -> dict:
    """
    Analyze an enterprise profile against prediction markets.

    Args:
        profile: dict with keys company, industry, revenue, geographies,
                 cost_inputs, revenue_streams
        events: list of market events
        cat_index: category → event indices

    Returns dict with:
        risks: list of {name, category, description, severity}
        keywords: list of search terms used
        existing_markets: list of {event, reason}
        suggested_markets: list of {title, risk_hedged, rationale}
    """
    key = _enterprise_cache_key(profile)
    cached = _load_enterprise_cache(key)
    if cached is not None:
        return cached

    client = _get_client()

    # ── Call 1: Risk analysis + search keywords ──────────────────────
    profile_text = (
        f"Company: {profile.get('company', 'N/A')}\n"
        f"Industry: {profile.get('industry', 'N/A')}\n"
        f"Annual Revenue: {profile.get('revenue', 'N/A')}\n"
        f"Geographies: {', '.join(profile.get('geographies', []))}\n"
        f"Key Cost Inputs: {', '.join(profile.get('cost_inputs', []))}\n"
        f"Revenue Streams: {', '.join(profile.get('revenue_streams', []))}\n"
    )

    risk_prompt = (
        f"You are a risk analyst. Given this enterprise profile:\n\n"
        f"{profile_text}\n"
        f"Do two things:\n\n"
        f"1. Identify the top 5 risks this business faces. For each risk give:\n"
        f"   - name: short risk name\n"
        f"   - category: one of (Regulatory, Macro/Economic, Geopolitical, "
        f"Supply Chain, Technology, Market/Competitive, Climate/Environmental)\n"
        f"   - description: one sentence explaining the risk\n"
        f"   - severity: High, Medium, or Low\n\n"
        f"2. Generate exactly 5 diverse search keywords/phrases that would help "
        f"find prediction markets relevant to hedging these risks.\n\n"
        f"Return ONLY valid JSON in this exact format:\n"
        f'{{"risks": [{{"name": "...", "category": "...", "description": "...", '
        f'"severity": "..."}}], "keywords": ["...", "..."]}}'
    )

    raw = _flash_call(client, risk_prompt)
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        call1 = json.loads(raw)
    except json.JSONDecodeError:
        call1 = {"risks": [], "keywords": []}

    risks = call1.get("risks", [])[:5]
    keywords = call1.get("keywords", [])[:5]

    # ── Local: category match + embedding search per keyword ────────
    # Match categories from risk keywords
    all_kw_text = " ".join(keywords)
    matched_cats = match_categories(all_kw_text)

    candidate_mask: np.ndarray | None = None
    if embeddings is not None:
        candidate_indices: set[int] = set()
        for cat in matched_cats:
            candidate_indices.update(cat_index.get(cat, []))
        if candidate_indices:
            mask = np.zeros(len(events), dtype=bool)
            for idx in candidate_indices:
                if idx < len(events):
                    mask[idx] = True
            candidate_mask = mask

    seen_ids: set[str] = set()
    candidates: list[dict] = []

    if embeddings is not None:
        # Embed all keywords in one batch, cosine sim per keyword
        kw_vecs = _embed_texts(client, keywords, task_type="RETRIEVAL_QUERY")
        for vec in kw_vecs:
            top = _embed_rank(client, vec, events, embeddings, EVENTS_PER_KEYWORD, candidate_mask)
            for e in top:
                eid = e.get("eventId", "")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    candidates.append(e)
    else:
        # Fallback: no embeddings, use all events from matched categories
        cat_indices: set[int] = set()
        for cat in matched_cats:
            cat_indices.update(cat_index.get(cat, []))
        if cat_indices:
            candidates = [events[i] for i in sorted(cat_indices) if i < len(events)]
        else:
            candidates = events

    # ── Call 2 (Flash): Rank existing markets from scraped data ─────
    catalog = "\n".join(
        f'{i}: {e.get("eventTitle", "")}'
        for i, e in enumerate(candidates)
    )

    risks_text = "\n".join(
        f"- {r.get('name', '')}: {r.get('description', '')}" for r in risks
    )

    rank_prompt = (
        f"You are a prediction market analyst helping a business hedge risks.\n\n"
        f"Enterprise profile:\n{profile_text}\n"
        f"Identified risks:\n{risks_text}\n\n"
        f"Candidate prediction markets:\n{catalog}\n\n"
        f"Pick the 20 most relevant existing markets from the list above. "
        f"For each, explain in one sentence how it relates to the company's risks "
        f"and how it could serve as a hedge.\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{"existing_markets": [{{"index": 0, "reason": "..."}}]}}'
    )

    raw2 = _flash_call(client, rank_prompt)
    raw2 = re.sub(r"^```(?:json)?\s*", "", raw2)
    raw2 = re.sub(r"\s*```$", "", raw2)

    try:
        call2 = json.loads(raw2)
    except json.JSONDecodeError:
        call2 = {"existing_markets": []}

    existing = []
    seen_idx: set[int] = set()
    for item in call2.get("existing_markets", []):
        idx = item.get("index", -1)
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen_idx:
            seen_idx.add(idx)
            existing.append({"event": candidates[idx], "reason": item.get("reason", "")})
    existing = existing[:20]

    # ── Call 3 (Pro): Suggest markets that should exist ──────────────
    suggest_prompt = (
        f"You are a prediction market designer and risk strategist.\n\n"
        f"Enterprise profile:\n{profile_text}\n"
        f"Identified risks:\n{risks_text}\n\n"
        f"Suggest 5-10 prediction markets that DO NOT currently exist but SHOULD. "
        f"These should be markets that a business like this one would use to hedge "
        f"its specific risks. Think creatively — cover regulatory, macro, geopolitical, "
        f"supply chain, and competitive angles.\n\n"
        f"For each market, provide:\n"
        f"  - title: the exact market question (e.g. \"Will the EU impose a carbon border tax by 2027?\")\n"
        f"  - risk_hedged: which risk from the profile it addresses\n"
        f"  - rationale: why this market should exist, who would trade it, and how it creates value\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{"suggested_markets": [{{"title": "...", "risk_hedged": "...", "rationale": "..."}}]}}'
    )

    raw3 = _pro_call(client, suggest_prompt)
    raw3 = re.sub(r"^```(?:json)?\s*", "", raw3)
    raw3 = re.sub(r"\s*```$", "", raw3)

    try:
        call3 = json.loads(raw3)
    except json.JSONDecodeError:
        call3 = {"suggested_markets": []}

    suggested = call3.get("suggested_markets", [])[:10]

    result = {
        "risks": risks,
        "keywords": keywords,
        "existing_markets": existing,
        "suggested_markets": suggested,
    }

    _save_enterprise_cache(key, result)
    return result
