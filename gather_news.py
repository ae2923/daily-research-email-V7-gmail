"""
gather_news.py — Part 1: Collect facts
=======================================
Edit this file freely to change how news is gathered.

Sources:
  1. SEC EDGAR (deterministic — no LLM, can't hallucinate)
  2. Grok (social narrative & sentiment — what people think is happening)
  3. Sonnet (financial news search — what IS happening beyond filings)
  4. Context (industry-level — regulatory, competitor, macro, mgmt changes)
  5. Verification (second pass — targeted diligence on high-signal social claims)

Output: data/facts_YYYYMMDD_HHMM.json
"""

import json
import time
import requests
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import anthropic
from openai import OpenAI

from config import (
    ANTHROPIC_API_KEY, XAI_API_KEY, FINNHUB_API_KEY,
    GROK_MODEL, SEARCH_MODEL,
    BATCH_DELAY, MAX_RETRIES, RETRY_BASE_DELAY, LOOKBACK_HOURS,
    TICKER_TO_CIK, BATCHES, MATERIAL_FORMS, TICKER_SEARCH_TERMS,
    BLOCKED_DOMAINS,
)

# ─── Clients ────────────────────────────────────────────────

grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

OUTPUT_DIR = Path("data")


# ─── Usage tracking ────────────────────────────────────────

class UsageTracker:
    """Thread-safe accumulator for API token usage and tool invocations."""

    def __init__(self):
        self._lock = threading.Lock()
        self.sonnet_input = 0
        self.sonnet_output = 0
        self.sonnet_cache_write = 0
        self.sonnet_cache_read = 0
        self.sonnet_searches = 0
        self.grok_input = 0
        self.grok_output = 0
        self.grok_tool_calls = 0

    def add_sonnet(self, message):
        """Track a Claude/Sonnet API response."""
        usage = getattr(message, "usage", None)
        if not usage:
            return
        with self._lock:
            self.sonnet_input += getattr(usage, "input_tokens", 0) or 0
            self.sonnet_output += getattr(usage, "output_tokens", 0) or 0
            self.sonnet_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.sonnet_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            # Count web_search invocations from response content blocks
            for block in getattr(message, "content", []):
                if getattr(block, "type", None) == "server_tool_use":
                    self.sonnet_searches += 1

    def add_grok(self, response):
        """Track a Grok Responses API response."""
        usage = getattr(response, "usage", None)
        if usage:
            with self._lock:
                self.grok_input += getattr(usage, "input_tokens", 0) or 0
                self.grok_output += getattr(usage, "output_tokens", 0) or 0
        # Count tool invocations in the output
        tool_calls = 0
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", "")
            if "search" in item_type or item_type == "tool_use":
                tool_calls += 1
        if tool_calls:
            with self._lock:
                self.grok_tool_calls += tool_calls

    def snapshot(self) -> dict:
        """Return a serializable snapshot of current usage."""
        with self._lock:
            return {
                "sonnet_input_tokens": self.sonnet_input,
                "sonnet_output_tokens": self.sonnet_output,
                "sonnet_cache_write_tokens": self.sonnet_cache_write,
                "sonnet_cache_read_tokens": self.sonnet_cache_read,
                "sonnet_search_calls": self.sonnet_searches,
                "grok_input_tokens": self.grok_input,
                "grok_output_tokens": self.grok_output,
                "grok_tool_calls": self.grok_tool_calls,
            }


usage_tracker = UsageTracker()


# ─── Source-blocklist helper ────────────────────────────────

# Normalized forms of BLOCKED_DOMAINS for matching Finnhub's `source` field,
# which returns human-readable names like "Yahoo", "SeekingAlpha", "MarketWatch"
# rather than full domains. We match case-insensitively on substring.
_BLOCKED_SOURCE_FRAGMENTS = tuple(
    d.split(".")[0].lower() for d in BLOCKED_DOMAINS
)


def _is_blocked_source(source: str) -> bool:
    """Return True if a Finnhub article's `source` matches BLOCKED_DOMAINS."""
    if not source:
        return False
    s = source.lower()
    return any(frag in s for frag in _BLOCKED_SOURCE_FRAGMENTS)


# ─── Retry helper ───────────────────────────────────────────

def _retry(fn):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            is_rate = (
                isinstance(e, anthropic.RateLimitError)
                or (hasattr(e, "status_code") and e.status_code == 429)
            )
            if is_rate and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Rate limited (attempt {attempt+1}). Waiting {delay}s...", flush=True)
                time.sleep(delay)
            else:
                raise


# ─────────────────────────────────────────────────────────────
# SOURCE 1: SEC EDGAR (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────

def fetch_edgar(ticker: str) -> list[dict]:
    """
    Free SEC data.sec.gov API. No key needed, just a User-Agent.
    Returns material filings from the last LOOKBACK_HOURS.
    """
    cik = TICKER_TO_CIK.get(ticker)
    if not cik:
        return []

    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {
        "User-Agent": "DailyBriefing aevangelopoulos@opglp.com",  # ← put your real email
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [EDGAR] {ticker}: {e}", flush=True)
        return []

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%d")

    results = []
    for i in range(min(len(forms), 50)):
        if dates[i] < cutoff:
            break
        if forms[i] in MATERIAL_FORMS:
            acc_no = accessions[i].replace("-", "")
            desc = descriptions[i] if i < len(descriptions) else ""
            results.append({
                "headline": f"{forms[i]} filed: {desc}" if desc else f"{forms[i]} filed",
                "source": "SEC EDGAR",
                "date": dates[i],
                "category": "filing",
                "verified": True,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no}/{primary_docs[i]}",
            })

    return results


# ─────────────────────────────────────────────────────────────
# SOURCE 1B: FINNHUB (company news, quotes, insider sentiment)
# ─────────────────────────────────────────────────────────────

def fetch_finnhub_news(ticker: str) -> list[dict]:
    """
    Finnhub company news API — deterministic, no LLM.
    Returns news headlines from the last LOOKBACK_HOURS.
    Free tier: 60 calls/min.

    NOTE ON STALENESS: Finnhub's `from`/`to` query parameters are a SUGGESTION,
    not a strict filter — the API regularly returns articles outside the window,
    usually historical articles that still mention the ticker. We enforce the
    date window a second time on our side using the article's `datetime` field.
    This prevents stale articles (e.g., a credit facility announcement from
    3 weeks ago resurfacing in today's feed) from reaching the LLM synthesis
    step, where the staleness rule is prompt-enforced and therefore unreliable.
    """
    if not FINNHUB_API_KEY:
        return []

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today_str = date.today().isoformat()

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": yesterday,
        "to": today_str,
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()
    except Exception as e:
        print(f"    [FINNHUB] {ticker} news: {e}", flush=True)
        return []

    # Hard cutoff: reject anything older than LOOKBACK_HOURS. Uses the same
    # constant as EDGAR, so both deterministic sources apply the same window.
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).timestamp()

    results = []
    blocked_count = 0
    stale_count = 0
    undated_count = 0
    for article in articles:
        dt = article.get("datetime", 0)

        # Reject articles with no datetime — treating missing dates as "today"
        # (the old behavior) is how historical articles sneak in wearing a
        # fresh date. If Finnhub doesn't know when it was published, we don't
        # trust it.
        if not dt:
            undated_count += 1
            continue

        # Reject articles older than the lookback window.
        if dt < cutoff_ts:
            stale_count += 1
            continue

        headline = article.get("headline", "")
        source = article.get("source", "unknown")
        article_date = datetime.fromtimestamp(dt, tz=timezone.utc).strftime("%Y-%m-%d")
        url_link = article.get("url", "")

        # Skip low-quality aggregator sources (Yahoo, Seeking Alpha, Benzinga, etc.)
        if _is_blocked_source(source):
            blocked_count += 1
            continue

        results.append({
            "headline": headline,
            "source": f"Finnhub/{source}",
            "date": article_date,
            "category": "news",
            "verified": False,
            "url": url_link,
        })

        if len(results) >= 10:  # Cap AFTER filtering, so we get 10 clean articles
            break

    # One consolidated log line per ticker. Quiet by default unless something
    # was actually filtered, so the console stays readable.
    filtered = blocked_count + stale_count + undated_count
    if filtered:
        parts = []
        if stale_count:
            parts.append(f"{stale_count} stale")
        if undated_count:
            parts.append(f"{undated_count} undated")
        if blocked_count:
            parts.append(f"{blocked_count} blocked-source")
        print(f"    [FINNHUB] {ticker}: filtered out {', '.join(parts)}", flush=True)

    return results


def fetch_finnhub_quote(ticker: str) -> dict | None:
    """
    Finnhub quote API — gets current price, change, and change percent.
    Returns dict with keys: c (current), d (change), dp (change_percent), pc (prev_close).
    """
    if not FINNHUB_API_KEY:
        return None

    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("c", 0) == 0:
            return None
        return data
    except Exception as e:
        print(f"    [FINNHUB] {ticker} quote: {e}", flush=True)
        return None


# ─────────────────────────────────────────────────────────────
# SOURCE 2: GROK (social narrative & sentiment engine)
# ─────────────────────────────────────────────────────────────

def fetch_grok(industry: str, tickers: list[str], edgar_summary: str) -> str:
    """Grok extracts social narrative, sentiment, and market perception — not facts."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today_str = date.today().isoformat()
    ticker_list = ", ".join(tickers)

    edgar_context = ""
    if edgar_summary:
        edgar_context = (
            f"\n\nFor reference, these SEC filings were detected in the last 24 hours:\n"
            f"{edgar_summary}\n"
            f"Note how social chatter interprets or frames these filings, but do NOT "
            f"report the filings themselves — that is handled by another source.\n"
        )

    prompt = (
        f"Today is {today_str}.\n\n"
        f"You are an X.COM NARRATIVE analyst, not a news reporter. Your job is to capture "
        f"what people are SAYING and THINKING about these stocks on X.com (Twitter) and the web — "
        f"not what is factually true. Factual news is handled by other sources. You capture crowd psychology.\n\n"
        f"For each of the following tickers in the {industry} sector: {ticker_list}\n\n"
        f"Search web and X/Twitter for discussion from the last 24 hours ONLY "
        f"(since {yesterday}). Then report the following for EACH ticker:\n\n"
        f"NARRATIVE THEME: What is X.com chatter mostly about for this stock today? "
        f"One sentence.\n\n"
        f"TOP CLAIMS (2-4): What specific claims or interpretations are spreading on X.com? "
        f"Quote or paraphrase the actual claims. For each, note:\n"
        f"  - NEW or RECYCLED: Is this actually new today, or is it an old story resurfacing?\n"
        f"  - SUPPORT: Confirmed by filings/news, reported but unconfirmed, or unsupported.\n\n"
        f"SENTIMENT: Bullish / Bearish / Mixed / Silent. One line on why.\n\n"
        f"NARRATIVE VS REALITY GAP: If X.com narrative contradicts or exaggerates what "
        f"filings/news actually show, flag it. If chatter is framing a real event in a "
        f"misleading way, flag it. If no gap: say 'Narrative aligned with record.'\n\n"
        f"If there is NO X.com discussion for a ticker in the last 24 hours, say "
        f"'No X.com chatter in window.' Do not fill silence with old stories.\n\n"
        f"CRITICAL: Do not report facts or events as news. Only report what people are "
        f"SAYING, INTERPRETING, SPECULATING, or FOCUSING ON. Your output will be labeled "
        f"as X.com narrative, not as verified information."
        f"{edgar_context}"
    )

    def _call():
        response = grok_client.responses.create(
            model=GROK_MODEL,
            input=[{"role": "user", "content": prompt}],
            tools=[
                {"type": "web_search"},
                {"type": "x_search", "from_date": yesterday, "to_date": today_str},
            ],
        )
        usage_tracker.add_grok(response)
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        text_parts = []
        for item in getattr(response, "output", []):
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
            elif hasattr(item, "text"):
                text_parts.append(item.text)
        return "\n".join(text_parts) if text_parts else str(response)

    return _retry(_call)


# ─────────────────────────────────────────────────────────────
# SOURCE 3: SONNET (financial news search — events & facts)
# ─────────────────────────────────────────────────────────────

SONNET_SYSTEM = """You are a financial news researcher. Your only job is to find FACTS, not analyze them.

STRICT 24-HOUR RULE:
- Today's date is provided in the user message. Only include facts published on today's
  date or yesterday's date.
- You MUST include the publication date for every fact you report.
- If you find a fact but cannot confirm it was published in the last 24 hours, EXCLUDE it.
- Stale news presented as fresh is a critical failure. When in doubt, leave it out.
- Search results labeled "recent" or "latest" are NOT proof of recency — check the
  actual publication date.

You are given SEC filings already captured from EDGAR. Do NOT repeat those.
Focus on ADDITIONAL news: earnings commentary, guidance updates,
analyst upgrades/downgrades, credit rating changes, debt issuances,
institutional ownership changes, and company press releases.

Output a plain text summary organized by ticker. For each ticker list
the material facts with source attribution and EXPLICIT DATE (YYYY-MM-DD).
If no additional material news from the last 24 hours beyond what EDGAR
already captured: write "No additional news in last 24 hours."

Be specific: include numbers, names, dates. No editorializing."""


def fetch_sonnet(industry: str, tickers: list[str], edgar_summary: str) -> str:
    """Sonnet searches for financial news gaps not covered by EDGAR."""
    today_str = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    ticker_list = ", ".join(tickers)

    prompt = (
        f"Today is {today_str}. Yesterday was {yesterday}.\n"
        f"Industry: {industry}\nTickers: {ticker_list}\n\n"
    )

    if edgar_summary:
        prompt += f"SEC filings already captured (DO NOT repeat these):\n{edgar_summary}\n\n"

    prompt += (
        f"Search for ADDITIONAL financial news published on {today_str} or {yesterday} ONLY. "
        f"For every fact you include, you MUST state its publication date in YYYY-MM-DD format. "
        f"If the publication date is before {yesterday}, EXCLUDE the fact entirely — "
        f"do not include it even as background.\n\n"
        f"Focus on: earnings commentary, guidance, analyst actions (upgrades/downgrades/PT changes), "
        f"credit events, and company press releases NOT covered by the filings above."
    )

    def _call():
        return claude_client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=3000,
            temperature=0,
            tools=[{"type": "web_search_20250305", "name": "web_search", "blocked_domains": BLOCKED_DOMAINS}],
            messages=[{"role": "user", "content": prompt}],
            system=SONNET_SYSTEM,
        )

    message = _retry(_call)
    usage_tracker.add_sonnet(message)
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


# ─────────────────────────────────────────────────────────────
# SOURCE 3B: TARGETED PREMIUM NEWS (per-ticker search of
#             Bloomberg, Reuters, WSJ, FT)
# ─────────────────────────────────────────────────────────────

PREMIUM_SEARCH_SYSTEM = """You are a news retrieval agent. You search SPECIFIC premium financial news outlets
for articles about a given company. You are NOT an analyst — just find and report what these outlets published.

For each search, report:
- HEADLINE (exact or near-exact)
- SOURCE (Bloomberg, Reuters, WSJ, or FT)
- DATE (YYYY-MM-DD — must be today or yesterday)
- SUMMARY (2-3 sentences of what the article says)
- URL if available

STRICT RULES:
- ONLY include articles published today or yesterday. No exceptions.
- ONLY include articles from: Bloomberg, Reuters, Wall Street Journal, Financial Times.
- If you find nothing from these four sources: say "No premium coverage in window."
- Do NOT editorialize. Report what the article says, not what it means.
- Do NOT confuse wire pickups (Yahoo Finance republishing Reuters) with original reporting —
  cite the original source."""


def fetch_premium_news(industry: str, tickers: list[str]) -> str:
    """
    Run explicit per-ticker searches against Bloomberg, Reuters, WSJ, and FT.
    Uses company names and thesis-relevant entities (from TICKER_SEARCH_TERMS)
    because headlines often use company names, not ticker symbols.
    """
    today_str = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    prompt = (
        f"Today is {today_str}. Yesterday was {yesterday}.\n"
        f"Industry: {industry}\n\n"
        f"For EACH of the following companies, search for articles published on "
        f"{today_str} or {yesterday} from these four sources ONLY:\n"
        f"  1. Bloomberg\n"
        f"  2. Reuters\n"
        f"  3. Wall Street Journal (WSJ)\n"
        f"  4. Financial Times (FT)\n\n"
        f"SEARCH INSTRUCTIONS — use the COMPANY NAME and ENTITY TERMS, not just the ticker:\n"
    )

    for ticker in tickers:
        terms = TICKER_SEARCH_TERMS.get(ticker, [])
        company_name = terms[0] if terms else ticker
        extra_terms = terms[1:] if len(terms) > 1 else []

        prompt += f"\n  {ticker} ({company_name}):\n"
        prompt += f"    - Search: '{company_name} Bloomberg', '{company_name} Reuters'\n"
        prompt += f"    - Search: '{company_name} WSJ', '{company_name} Financial Times'\n"
        if extra_terms:
            for term in extra_terms:
                prompt += f"    - Also search: '{term} Bloomberg' (thesis-relevant entity)\n"

    prompt += (
        f"\nAlso run these industry-level searches:\n"
        f"  - '{industry} Bloomberg {today_str}'\n"
        f"  - '{industry} Reuters {today_str}'\n\n"
        f"Report findings organized by ticker. For articles found, include:\n"
        f"  HEADLINE, SOURCE (Bloomberg/Reuters/WSJ/FT), DATE (YYYY-MM-DD), "
        f"SUMMARY (2-3 sentences).\n"
        f"If a ticker has no coverage from any of the four sources: write "
        f"'No premium coverage.'\n"
        f"If an article is relevant to MULTIPLE tickers in this batch, list it "
        f"under each affected ticker."
    )

    def _call():
        return claude_client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=3000,
            temperature=0,
            tools=[{"type": "web_search_20250305", "name": "web_search", "blocked_domains": BLOCKED_DOMAINS}],
            messages=[{"role": "user", "content": prompt}],
            system=PREMIUM_SEARCH_SYSTEM,
        )

    message = _retry(_call)
    usage_tracker.add_sonnet(message)
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


# ─────────────────────────────────────────────────────────────
# SOURCE 4: CONTEXT (industry-level — regulatory, competitor,
#            macro, management changes)
# ─────────────────────────────────────────────────────────────

CONTEXT_SYSTEM = """You are an industry research analyst. Your job is to find the CONTEXT
that surrounds a group of tickers — not company-specific news (that's handled separately).

You search for the forces acting ON an industry, not news FROM individual companies.

For the given industry and tickers, search for:

1. REGULATORY / LEGISLATIVE — new rules, proposed legislation, enforcement actions,
   agency decisions, or policy shifts affecting this sector. Include international
   regulatory moves that affect US-listed companies.

2. COMPETITOR MOVES — actions by companies NOT in the ticker list that change the
   competitive landscape: new entrants, capacity additions, pricing changes, exits,
   major contract wins.

3. MACRO CATALYSTS — economic data releases, commodity price moves, rate decisions,
   trade policy changes, or geopolitical developments with outsized impact on this
   specific sector (not generic macro).

4. MANAGEMENT CHANGES — CEO/CFO/COO appointments, departures, or retirements at
   the listed tickers that may not have triggered SEC filings yet, or that were
   announced in the last 1-2 weeks and remain thesis-relevant.

5. SUPPLY CHAIN / INPUT COSTS — changes in key input prices, supplier disruptions,
   logistics shifts, or capacity constraints affecting the sector.

TIME WINDOW: Search the last 7 days. This is intentionally wider than 24 hours
because context moves slower than news. Flag the date for each item.

OUTPUT FORMAT: Organize by category (Regulatory, Competitor, Macro, Management,
Supply Chain). Under each, list the material facts with source and date.
If nothing material in a category, write "Nothing material."

Be specific: include numbers, names, dates, jurisdictions. No editorializing."""


def fetch_context(industry: str, tickers: list[str]) -> str:
    """Search for industry-level context: regulatory, competitor, macro, mgmt changes."""
    today_str = date.today().isoformat()
    ticker_list = ", ".join(tickers)

    prompt = (
        f"Today is {today_str}.\n"
        f"Industry: {industry}\n"
        f"Tickers in our portfolio for this sector: {ticker_list}\n\n"
        f"Search for industry-level context from the last 7 days that affects "
        f"these tickers. Focus on forces OUTSIDE the companies themselves: "
        f"regulation, competitors, macro, management changes, and supply chain. "
        f"Do NOT search for company-specific news or earnings — that's handled "
        f"by other sources."
    )

    def _call():
        return claude_client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=3000,
            temperature=0,
            tools=[{"type": "web_search_20250305", "name": "web_search", "blocked_domains": BLOCKED_DOMAINS}],
            messages=[{"role": "user", "content": prompt}],
            system=CONTEXT_SYSTEM,
        )

    message = _retry(_call)
    usage_tracker.add_sonnet(message)
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


# ─────────────────────────────────────────────────────────────
# SOURCE 5: VERIFICATION (second pass — diligence on social
#            claims that are specific enough to check)
# ─────────────────────────────────────────────────────────────

TRIAGE_SYSTEM = """You are a claim triage analyst. You read social narrative output
and extract ONLY the claims that are specific enough and consequential enough
to warrant a targeted verification search.

A claim is worth verifying if it meets ALL of these criteria:
1. SPECIFIC — names a concrete event, number, person, deal, or action
   (not vague sentiment like "bullish vibes" or "stock is undervalued")
2. CONSEQUENTIAL — if true, it would affect a thesis node
   (demand, margin, capital allocation, balance sheet, mgmt credibility)
3. UNCONFIRMED — the social source itself marks it as unsupported or
   unconfirmed, OR it makes a factual claim without citing a source
4. CHECKABLE — a web search could plausibly confirm or refute it

Output ONLY a JSON array of claims to verify. Max 4 claims per batch.
Each claim object must have:
  - "ticker": the ticker symbol
  - "claim": the specific claim to verify (one sentence)
  - "search_query": a short web search query to check this claim (3-8 words)

If NO claims meet the criteria, output an empty array: []

Output raw JSON only. No markdown, no code fences, no explanation."""


def _grok_has_checkable_content(grok_text: str) -> bool:
    """
    Heuristic: return True if Grok's output contains anything worth triaging.

    The verification pass exists to fact-check SPECIFIC, CONSEQUENTIAL claims
    from X.com. If Grok returned no chatter, pure sentiment, or explicit
    "no claims" language, there is nothing for triage to extract — and running
    triage anyway wastes a Sonnet call + its web-search fees.

    We're conservative here: only skip when we're confident there's nothing
    to find. Any doubt → let triage run. A false positive here (skipping when
    we shouldn't) costs us one missed verification; a false negative (running
    when we shouldn't) costs ~$0.01-0.03 in Sonnet tokens + search fees.
    """
    if not grok_text or not grok_text.strip():
        return False

    text_lower = grok_text.lower()

    # Explicit "nothing happened" markers from the Grok prompt template
    empty_markers = [
        "no x.com chatter in window",
        "no x.com discussion",
        "no chatter in window",
        "narrative aligned with record",
    ]
    # If EVERY marker we look for is present AND the total text is short,
    # Grok is clearly reporting silence rather than substantive narrative.
    # (A batch with 6 tickers all showing "No X.com chatter in window" runs
    # ~800 chars. A batch with real narrative easily runs 3000+.)
    if len(grok_text) < 1200 and any(m in text_lower for m in empty_markers):
        return False

    # Check for "TOP CLAIMS" section with actual content.
    # The Grok prompt asks for "TOP CLAIMS (2-4)" — if that heading is absent
    # OR the only content under it is variants of "none"/"N/A", skip triage.
    if "top claims" not in text_lower and "claim" not in text_lower:
        return False

    return True


def verify_claims(industry: str, grok_text: str, sonnet_text: str,
                  edgar_summary: str) -> str:
    """
    Two-step verification:
    1. Triage: Sonnet reads Grok output and picks claims worth checking
    2. Search: Sonnet runs targeted web searches on each claim
    Returns a formatted verification report.
    """
    today_str = date.today().isoformat()

    # ── Short-circuit: if Grok returned no substantive content, skip entirely ──
    # Saves one Sonnet triage call per quiet batch (~300 system + Grok context
    # input tokens). Across 4 batches on a quiet day, that's ~1200-2000
    # tokens plus up to 4 unnecessary web-search fees avoided.
    if not _grok_has_checkable_content(grok_text):
        print(f"    [VERIFY] Grok returned no checkable content; skipping triage.", flush=True)
        return "Verification skipped: Grok output contained no substantive claims to verify."

    # ── Step 1: Triage — extract checkable claims ──
    triage_prompt = (
        f"Today is {today_str}. Industry: {industry}.\n\n"
        f"Below is social narrative output from Grok. Extract specific, consequential, "
        f"unconfirmed claims that are worth verifying with a targeted web search.\n\n"
        f"=== SOCIAL NARRATIVE (Grok) ===\n{grok_text}\n\n"
        f"=== FOR CONTEXT — already confirmed by EDGAR ===\n"
        f"{edgar_summary if edgar_summary else 'No EDGAR filings.'}\n\n"
        f"=== FOR CONTEXT — already found by news search ===\n{sonnet_text}\n\n"
        f"If a claim from Grok is ALREADY confirmed by EDGAR or Sonnet above, "
        f"do NOT include it — it's already verified. Only include claims that are "
        f"NOT yet confirmed by other sources."
    )

    def _triage():
        return claude_client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": triage_prompt}],
            system=TRIAGE_SYSTEM,
        )

    triage_msg = _retry(_triage)
    usage_tracker.add_sonnet(triage_msg)
    triage_text = "".join(
        b.text for b in triage_msg.content if getattr(b, "type", None) == "text"
    ).strip()

    # Parse the claims
    try:
        # Strip markdown fences if present
        clean = triage_text.replace("```json", "").replace("```", "").strip()
        claims = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        print(f"    [VERIFY] Could not parse triage output, skipping verification", flush=True)
        return "Verification skipped: no parseable claims from triage."

    if not claims:
        return "No social claims met the threshold for verification."

    print(f"    [VERIFY] {len(claims)} claims to check (parallel)", flush=True)

    # ── Step 2: Targeted verification searches (parallel) ──
    def _verify_single(claim):
        ticker = claim.get("ticker", "???")
        claim_text = claim.get("claim", "")
        print(f"    [VERIFY] Checking: {ticker} — {claim_text[:60]}...", flush=True)

        verify_prompt = (
            f"Today is {today_str}.\n\n"
            f"A social media claim is circulating about {ticker}:\n"
            f'"{claim_text}"\n\n'
            f"Search the web to determine if this claim is true, false, or unconfirmable. "
            f"Look for: official company statements, SEC filings, reputable news reports, "
            f"or other primary sources.\n\n"
            f"Respond with:\n"
            f"VERDICT: CONFIRMED / REFUTED / UNCONFIRMABLE\n"
            f"EVIDENCE: What you found (or didn't find), with source and date.\n"
            f"DETAIL: If confirmed, add any important details the social claim missed or got wrong.\n"
            f"Keep it to 3-5 sentences total."
        )

        def _call():
            return claude_client.messages.create(
                model=SEARCH_MODEL,
                max_tokens=500,
                temperature=0,
                tools=[{"type": "web_search_20250305", "name": "web_search", "blocked_domains": BLOCKED_DOMAINS}],
                messages=[{"role": "user", "content": verify_prompt}],
                system="You are a fact-checker. Search the web and verify claims concisely.",
            )

        try:
            verify_msg = _retry(_call)
            usage_tracker.add_sonnet(verify_msg)
            result_text = "".join(
                b.text for b in verify_msg.content if getattr(b, "type", None) == "text"
            )
            return f"CLAIM ({ticker}): {claim_text}\n{result_text.strip()}"
        except Exception as e:
            return f"CLAIM ({ticker}): {claim_text}\nVERDICT: ERROR — search failed: {e}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        verification_results = list(pool.map(_verify_single, claims[:4]))

    return "\n\n".join(verification_results)


# ─────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────

def process_batch(batch: dict) -> dict:
    industry = batch["industry"]
    tickers = batch["tickers"]

    print(f"\n{'='*50}", flush=True)
    print(f"[{industry}]", flush=True)

    # ── Phase 1A: EDGAR (deterministic, fast) ──
    print("  EDGAR...", flush=True)
    edgar_by_ticker = {}
    for ticker in tickers:
        filings = fetch_edgar(ticker)
        edgar_by_ticker[ticker] = filings
        if filings:
            print(f"    {ticker}: {len(filings)} filings", flush=True)
        time.sleep(0.15)

    # ── Phase 1B: Finnhub (news + quotes, deterministic, fast) ──
    print("  Finnhub...", flush=True)
    finnhub_news_by_ticker = {}
    finnhub_quotes = {}
    for ticker in tickers:
        news = fetch_finnhub_news(ticker)
        finnhub_news_by_ticker[ticker] = news
        if news:
            print(f"    {ticker}: {len(news)} Finnhub articles", flush=True)

        quote = fetch_finnhub_quote(ticker)
        if quote:
            finnhub_quotes[ticker] = quote
        time.sleep(0.12)  # Stay under 60 calls/min

    # Build price summary for the industry row
    price_lines = []
    for ticker in tickers:
        q = finnhub_quotes.get(ticker)
        if q and q.get("dp") is not None:
            price_lines.append(f"  {ticker}: ${q['c']:.2f} ({q['dp']:+.2f}%)")
    price_summary = "\n".join(price_lines) if price_lines else "No quote data available."

    # Build Finnhub news summary
    finnhub_lines = []
    for ticker in tickers:
        for article in finnhub_news_by_ticker.get(ticker, []):
            finnhub_lines.append(
                f"  {ticker}: {article['headline']} ({article['date']}, {article['source']})"
            )
    finnhub_summary = "\n".join(finnhub_lines) if finnhub_lines else ""

    # Build EDGAR summary for Grok/Sonnet so they know what NOT to repeat
    edgar_lines = []
    for ticker in tickers:
        for f in edgar_by_ticker.get(ticker, []):
            edgar_lines.append(f"  {ticker}: {f['headline']} ({f['date']})")
    edgar_summary = "\n".join(edgar_lines)

    # ── Phase 2: Grok + Sonnet + Context + Premium News in parallel ──
    print("  Grok + Sonnet + Context + Premium (parallel)...", flush=True)
    grok_text, sonnet_text, context_text, premium_text = "", "", "", ""
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            grok_future = pool.submit(fetch_grok, industry, tickers, edgar_summary)
            sonnet_future = pool.submit(fetch_sonnet, industry, tickers, edgar_summary)
            context_future = pool.submit(fetch_context, industry, tickers)
            premium_future = pool.submit(fetch_premium_news, industry, tickers)
            grok_text = grok_future.result()
            sonnet_text = sonnet_future.result()
            context_text = context_future.result()
            premium_text = premium_future.result()
    except Exception as e:
        print(f"  ERROR: {traceback.format_exc()}", flush=True)
        grok_text = grok_text or f"Search failed: {e}"
        sonnet_text = sonnet_text or f"Search failed: {e}"
        context_text = context_text or f"Search failed: {e}"
        premium_text = premium_text or f"Search failed: {e}"

    # ── Phase 3: Verify high-signal X.com claims ──
    print("  Verification pass...", flush=True)
    try:
        verification_text = verify_claims(industry, grok_text, sonnet_text, edgar_summary)
    except Exception as e:
        print(f"  VERIFY ERROR: {traceback.format_exc()}", flush=True)
        verification_text = f"Verification failed: {e}"

    # ── Build output ──
    merged_tickers = []
    for ticker in tickers:
        facts = list(edgar_by_ticker.get(ticker, []))
        # Append Finnhub news as additional facts
        for article in finnhub_news_by_ticker.get(ticker, []):
            facts.append(article)
        merged_tickers.append({"ticker": ticker, "facts": facts})

    return {
        "industry": industry,
        "tickers": merged_tickers,
        "grok_raw": grok_text,
        "sonnet_raw": sonnet_text,
        "context_raw": context_text,
        "premium_raw": premium_text,
        "verification_raw": verification_text,
        "edgar_summary": edgar_summary,
        "finnhub_summary": finnhub_summary,
        "price_summary": price_summary,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run() -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = OUTPUT_DIR / f"facts_{timestamp}.json"

    # Process batches with limited parallelism (2 at a time) to stay under
    # Anthropic API rate limits. Each batch runs 4 parallel web-search calls
    # internally, so 4 batches × 4 calls = 16 simultaneous web searches,
    # which exhausts the rate limit. 2 batches × 4 = 8 is safe.
    print(f"Processing {len(BATCHES)} batches (2 at a time)...", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        all_batches = list(pool.map(process_batch, BATCHES))

    with open(output_path, "w") as f:
        json.dump({"batches": all_batches, "gather_usage": usage_tracker.snapshot()}, f, indent=2)

    gather_usage = usage_tracker.snapshot()
    print(f"\n✓ Facts saved to: {output_path}", flush=True)
    print(
        f"  Gather usage — Sonnet: {gather_usage['sonnet_input_tokens']:,} in / "
        f"{gather_usage['sonnet_output_tokens']:,} out / "
        f"{gather_usage['sonnet_search_calls']} searches | "
        f"Grok: {gather_usage['grok_input_tokens']:,} in / "
        f"{gather_usage['grok_output_tokens']:,} out / "
        f"{gather_usage['grok_tool_calls']} tool calls",
        flush=True,
    )
    return str(output_path)


if __name__ == "__main__":
    run()
