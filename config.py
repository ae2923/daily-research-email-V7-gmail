"""
config.py — Shared configuration
=================================
Both gather_news.py and analyze_briefing.py import from here.
This is the ONLY file you need to edit when adding/removing tickers.
"""

import os
import sys

# ─── Environment variables ──────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# ─── Gmail SMTP (email sending) ─────────────────────────────
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

# ─── Models ─────────────────────────────────────────────────
GROK_MODEL = "grok-4-1-fast"
SEARCH_MODEL = "claude-haiku-4-5-20251001"       # Stage 2: financial news search
ANALYSIS_MODEL = "claude-opus-4-5"       # Stage 3: synthesis

# ─── Timing ─────────────────────────────────────────────────
BATCH_DELAY = 5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 30
LOOKBACK_HOURS = 28

# ─── Domain blocklist ──────────────────────────────────────
# Passed to Anthropic web_search as `blocked_domains` AND used as
# substring match against Finnhub's `source` field.
# Per Anthropic docs:
#   - No HTTP/HTTPS scheme (use 'yahoo.com' not 'https://yahoo.com')
#   - Subdomains are auto-included (yahoo.com covers finance.yahoo.com)
#   - Subpaths are supported
# Safe to add/remove entries — no shape contract.
BLOCKED_DOMAINS = [
    "yahoo.com",           # Yahoo Finance
    "seekingalpha.com",    # Seeking Alpha
    "benzinga.com",        # Benzinga
    "marketwatch.com",     # MarketWatch
    "fool.com",            # Motley Fool
    "zacks.com",           # Zacks
    "investorplace.com",   # InvestorPlace
    "tipranks.com",        # TipRanks
    "simplywall.st",       # Simply Wall St
]

# ─── Your tickers ──────────────────────────────────────────
# CIK lookup: https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK=TICKER&action=getcompany
TICKER_TO_CIK = {
    "RIOT": "0001167419",
    "CORZ": "0001882378",
    "WULF": "0001822523",
    "BITF": "0001725872",
    "TLN":  "0001862150",
    "GDS":  "0001720116",
    "LBRT": "0001694028",
    "PUMP": "0001684608",
    "PSIX": "0001643302",
    "PTTA": "0001854795",
    "MP":   "0001801368",
    "FCX":  "0000831259",
    "EXE":  "0001753706",
    "NOG":  "0001326428",
    "OXY":  "0000797468",
    "CHRD": "0001533924",
    "CRC":  "0001609065",
    "PR":   "0001163566",
    "EQT":  "0000033213",
    "MTDR": "0001520006",
}

# ─── Batches ────────────────────────────────────────────────
BATCHES = [
    {
        "industry": "Bitcoin Mining",
        "tickers": ["TLN", "CORZ", "GDS", "WULF", "RIOT", "BITF"],
    },
    {
        "industry": "Power / Energy Infrastructure",
        "tickers": ["LBRT", "PUMP", "PSIX"],
    },
    {
        "industry": "Oilfield Services (OFSE)",
        "tickers": ["PTTA", "MP", "FCX"],
    },
    {
        "industry": "Oil & Gas E&P",
        "tickers": ["EXE", "NOG", "OXY", "CHRD", "CRC", "PR", "EQT", "MTDR"],
    },
]

# ─── SEC form types that count as material ──────────────────
MATERIAL_FORMS = {
    "8-K", "8-K/A",
    "13D", "13D/A", "SC 13D", "SC 13D/A",
    "SC 13G", "SC 13G/A",
    "4", "3",
    "S-1", "S-3",
    "10-Q", "10-K",
    "DEFA14A", "DEF 14A",
}

TICKER_SEARCH_TERMS = {
    # Bitcoin Mining
    "TLN":  ["Talen Energy", "PJM power", "Susquehanna nuclear"],
    "CORZ": ["Core Scientific", "CoreWeave data center"],
    "GDS":  ["GDS Holdings", "China data center"],
    "WULF": ["TeraWulf", "Lake Mariner"],
    "RIOT": ["Riot Platforms", "bitcoin mining"],
    "BITF": ["Bitfarms", "KEEL Infrastructure"],
    # Power / Energy Infrastructure
    "LBRT": ["Liberty Energy", "Liberty Oilfield"],
    "PUMP": ["ProPetro", "PROPWR data center"],
    "PSIX": ["Power Solutions International"],
    # Oilfield Services / Critical Minerals
    "PTTA": ["Perpetua Resources", "Stibnite Gold", "antimony"],
    "MP":   ["MP Materials", "rare earth", "Mountain Pass"],
    "FCX":  ["Freeport-McMoRan", "Grasberg copper"],
    # Oil & Gas E&P
    "EXE":  ["Expand Energy", "Chesapeake Energy"],
    "NOG":  ["Northern Oil and Gas"],
    "OXY":  ["Occidental Petroleum", "Berkshire Hathaway OXY"],
    "CHRD": ["Chord Energy", "Enerplus"],
    "CRC":  ["California Resources", "carbon capture CRC"],
    "PR":   ["Permian Resources"],
    "EQT":  ["EQT Corporation", "Appalachian gas", "Commonwealth LNG"],
    "MTDR": ["Matador Resources", "San Mateo midstream"],
}
