"""
analyze_briefing.py — Part 2: Analyze + Email
==============================================
Reads the facts JSON from gather_news.py, runs Opus analysis,
builds HTML, and sends the email.

You should rarely need to edit this. Only touch it to:
  - Change the L1/L2/L3 analysis prompt
  - Change the HTML styling
  - Change the email settings

Usage:
  python analyze_briefing.py                          # uses latest facts file
  python analyze_briefing.py data/facts_20260410.json # uses specific file
"""

import sys
import json
import time
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import anthropic

from config import (
    ANTHROPIC_API_KEY, GMAIL_ADDRESS, GMAIL_APP_PASSWORD,
    ANALYSIS_MODEL, MAX_RETRIES, RETRY_BASE_DELAY, BATCH_DELAY,
)

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

DATA_DIR = Path("data")


# ─── Usage tracking ────────────────────────────────────────

class OpusUsageTracker:
    """Thread-safe accumulator for Opus analysis token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0

    def add(self, message):
        usage = getattr(message, "usage", None)
        if not usage:
            return
        with self._lock:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "opus_input_tokens": self.input_tokens,
                "opus_output_tokens": self.output_tokens,
                "opus_cache_write_tokens": self.cache_write_tokens,
                "opus_cache_read_tokens": self.cache_read_tokens,
            }


opus_tracker = OpusUsageTracker()


# ─── Analysis prompt ────────────────────────────────────────
# NOTE: This is a TEMPLATE. {today} and {yesterday} are injected
# at runtime in analyze_batch() so the staleness rule is hardcoded
# into the system prompt itself, not reliant on cross-referencing
# the user message.

ANALYSIS_SYSTEM_TEMPLATE = """You are a buy-side analyst applying structured materiality frameworks. You receive raw inputs from eight sources and produce HTML <tr> rows. Your job is expectation-delta analysis, not thesis description. If you find yourself restating what the thesis is rather than what changed in it, stop and rewrite.

THE EIGHT SOURCES AND HOW TO USE THEM:
- Source 1 (SEC EDGAR): Ground truth. Verified filings. Highest confidence. Use in L1 with no tag.
- Source 1B (Finnhub News): Aggregated news headlines from major outlets for each ticker. Deterministic API — not AI-generated. Use in L1 tagged [unverified]. CRITICAL: Finnhub returns historical articles that mention the ticker, NOT only fresh news. Check each Finnhub item's date against the staleness rule below. A Finnhub article about an announcement from three weeks ago is stale even if Finnhub surfaces it today.
- Source 1C (Finnhub Quotes): Current price, 1-day change, and change percent for each ticker. Use in the INDUSTRY SUMMARY ROW to report sector-wide price action. This is factual market data.
- Source 2 (Grok — X.com Narrative): Crowd psychology, sentiment, rumors, dominant narratives from X.com (Twitter). Tells you what the market THINKS is happening. Most Grok content belongs in L2 (Variant Perception bullet). HOWEVER: if Grok surfaces a specific, consequential claim that the PM needs to see immediately — a CEO departure rumor, a deal collapse rumor, a financing stress signal — it CAN go in L1 tagged [X.com]. The threshold is: would ignoring this claim be reckless? If yes, it goes in L1 as [X.com]. If it's vague sentiment or recycled hype, it stays in L2 only. Operational metrics (hashrate, production, capacity) should NEVER come from X.com — require a filing, IR release, or Finnhub article.
- Source 3 (Sonnet — Financial News): Reported events from financial media. Use in L1 tagged [unverified].
- Source 3B (Premium News — Bloomberg, Reuters, WSJ, FT): Targeted per-ticker searches of the four most authoritative financial outlets. These are the highest-confidence news items after EDGAR filings. Use in L1 tagged [unverified]. If Source 3B and Source 3 report the same fact, prefer the Source 3B version (more authoritative outlet). If Source 3B surfaces a story that NO other source caught, elevate it — this is the gap-filling function.
- Source 4 (Industry Context): Regulatory, competitor, macro, and management forces over the last 7 days. Wider window by design. Use for the Second-Order bullet and to catch thesis-relevant context.
- Source 5 (Verification): Targeted fact-checks of high-signal X.com claims. This tells you whether specific Grok claims were CONFIRMED, REFUTED, or UNCONFIRMABLE. If Source 5 confirms an X.com claim, you may upgrade it from [X.com] to [verified by search] in L1. If Source 5 refutes a claim, flag the contradiction in L2 bullet 4 (Credibility Check) — the market is trading on false information.

Output ONLY <tr> rows. No prose, no <table> wrapper, no markdown, no code fences.

Columns: Name | L1 News (24h) | L2 Materiality Analysis | L3 So What

── STALENESS RULE (enforced — zero tolerance) ──
Today is {today}. Yesterday was {yesterday}.
L1 is STRICTLY for facts dated {today} or {yesterday} ONLY. No exceptions.
If a source reports a fact dated before {yesterday}: it CANNOT go in L1.
If a source reports a fact with NO date: it CANNOT go in L1.
Before writing each L1 bullet, state to yourself: "What is the date of this fact?" If it is not {today} or {yesterday}, exclude it.
This applies to Finnhub articles about old announcements (e.g., a credit facility expansion announced three weeks ago surfacing in today's Finnhub feed is STALE — do not include).
If the fact is still thesis-relevant, reference it in L2 labeled "[prior window]".
Stale news in L1 is a critical error — it misleads the PM on what is new.

── DECISION RULE (apply before writing anything) ──
If a headline is neither expectation-changing, thesis-node-altering, nor probability-tree-shifting, classify it as NOISE and do not elevate it to L2/L3.

── L1: What Happened (last 24 hours ONLY) ──
Per cell: up to 3 bullets, ≤12 words each. Do NOT pad to 3 if only 1-2 items pass the staleness and materiality filters.
  - ONLY facts dated {today} or {yesterday}. Everything else is excluded from L1.
  - Facts from Source 1 (EDGAR) are confirmed — no tag needed.
  - Facts from Source 3 (Sonnet/news) or Source 1B (Finnhub) are reported — tag with [unverified].
  - High-signal claims from Source 2 (Grok) that meet the threshold above — tag with [X.com]. If Source 5 confirmed the claim, tag with [verified by search] instead.
  - Every L1 bullet must include either a dollar figure, percentage, counterparty name, filing type, or specific date. "Options flow noted" / "stock moved on volume" / "analyst chatter" are banned — if you cannot name the strike/expiry/ratio, omit it.
  - If no material items from the last 24 hours: bullet 1 = "No material news in window." Do not fabricate bullets 2-3.

── L2: Materiality Analysis ──
Per cell: ≤15 words per bullet. This is where the analytical value lives — spend your reasoning here. Use the 15 words for concrete data, not qualifiers.
  - Bullet 1: THESIS NODE — which node is affected (demand | margin | capital allocation | balance sheet | mgmt credibility). Name the node, then classify the UPDATE: RECONFIRM / REFUTE / NEW / UNCHANGED (see thesis-status definitions in L3).
  - Bullet 2: VARIANT PERCEPTION — what shifted vs market expectations. State consensus view and the delta (e.g., "Consensus models $X; filing implies $Y, a Z% miss"). USE Source 2 (Grok) here: does crowd narrative match or diverge from the factual record? If chatter is ahead of confirmation, say so. If Source 5 refuted a popular social claim, flag the gap — the market is mispricing. If Source 2 is empty, use this bullet for base-rate check: is the implied expectation above or below historical frequency for this event type?
  - Bullet 3: SECOND-ORDER — one downstream consequence the market hasn't priced. Name a SPECIFIC entity: a company (ticker or full name), named supplier, named competitor, named regulator, or named counterparty. Banned abstractions in this bullet: "hyperscalers", "big tech", "the industry", "peers", "competitors", "customers", "regulators" — these are categories, not entities. If you cannot name a specific entity, this bullet is not ready; omit it and drop to 3 L2 bullets.
  - Bullet 4: CREDIBILITY CHECK — flag divergence between management words and actions (insider sales vs bullish guidance, capital raise vs buyback talk, CFO exit during "strong quarter"). Cross-reference Source 1 filings (especially Form 4) against Source 2 narrative. Also flag if Source 5 refuted a social claim the market is acting on. If none: "No mgmt credibility divergence."

Number of L2 bullets varies by novelty score (see below).

Source confidence hierarchy: SEC EDGAR (highest) > company IR > financial media > verified social (Source 5 confirmed) > unverified social (lowest).
Balance sheet news (liquidity, covenants, refinancing) outranks earnings commentary.

── L3: So What ──
Per cell: ≤15 words per bullet. Synthesize what L2 means for thesis conviction. L3 is NOT a restatement of L2 — if you find yourself writing "AI pivot de-risks mining" in L3 after writing the same thing in L2, you have failed. L3 articulates the DELTA.

  Bullet 1: THESIS STATUS — one of four labels, defined precisely:
    • RECONFIRM — NEW evidence arrived today/yesterday that strengthens conviction. Must name what the evidence was AND what it upgraded in the thesis (e.g., "RECONFIRM — contract extension lifts 2027 revenue floor ~8% vs prior model").
    • REFUTE — new evidence weakens the thesis. Partial refutation counts; name what assumption is now under pressure.
    • NEW — new evidence forces a different analytical lens. The prior thesis framework is insufficient; a new node or driver now dominates.
    • UNCHANGED — no new thesis-relevant evidence arrived in the window. This is a VALID and expected state on most days for most names. Do not dress silence as reaffirmation.
  HARD RULE: If L1 = "No material news in window." then L3 bullet 1 MUST be UNCHANGED. Period. RECONFIRM requires new evidence — absence of bad news is not new evidence.
  If the sector row classifies as NEW (structural shift), at least one ticker in the sector should inherit a NEW or REFUTE tag — sector-level structural change cannot coexist with ticker-level UNCHANGED across the board.

  Bullet 2: WHAT TO WATCH — ONE specific item the PM can put on a calendar. Must include: (a) a named event/filing/data point and (b) a date or date window. Examples of acceptable: "Q1 10-Q filing (expected May 8)" / "PJM capacity auction results May 28" / "Form 4 post-vesting window opens May 15". Banned: multiple items bundled, "further developments," "next earnings" (without date), unnamed filings.

  Bullet 3: COST OF IGNORING — "If wrong, risk is [specific downside, named counterparty, or missed opportunity with a magnitude]." Banned: generic hedges like "balance sheets stress across sector" — name which balance sheet, at what trigger level.

Number of L3 bullets varies by novelty score (see below).

── NOVELTY SCORE (controls output density) ──
Assign each ticker a novelty score (1-5) inline after the ticker name in the first <td>: 1 = fully priced in, 5 = completely new.
Anchors — calibrate against these:
  1 = Pure noise. Routine price move, no news, no filings. → UNCHANGED.
  2 = Minor update. Incremental datapoint consistent with consensus. → likely RECONFIRM or UNCHANGED.
  3 = Material update. A thesis node received genuinely new information that a reasonable PM had not yet priced. → RECONFIRM, REFUTE, or NEW.
  4 = Significant surprise. Consensus assumption is now wrong or a new driver has emerged. Examples: 8-K with material contract, CFO departure, covenant breach, regulatory action, large insider transaction. → Usually REFUTE or NEW.
  5 = Thesis-breaking. The framework used to underwrite this name no longer applies. Examples: going-concern disclosure, take-private announcement, fraud allegation, regulator halts operations, structural industry shift (e.g., competing buyer emerges for the sector's key scarce input). → NEW with [ESCALATE].
CALIBRATION CHECK: If the sector row describes a "structural shift" or "regime change," the sector's novelty is ≥4. If you assign every ticker novelty 1-3 across a sector with a structural shift, you have miscalibrated — rescore.
Density rules:
  - Novelty 1 (QUIET TICKER SHORT-CIRCUIT): If a ticker has no EDGAR filings, no Finnhub news in window, no Source 3/3B coverage, and no high-signal Grok claim — i.e., nothing material arrived — emit a MINIMAL row. Format:
      L1: single bullet = "No material news in window."
      L2: single bullet = "Thesis nodes untouched; last catalyst [YYYY-MM-DD, brief event]." If no prior catalyst is known from the inputs, use "Thesis nodes untouched; no prior catalyst in source window."
      L3: single bullet = "UNCHANGED — no new evidence; next scheduled item [named filing/date or 'none scheduled']."
    That is the ENTIRE row. Do not produce additional bullets. Do not write credibility checks, second-order speculation, or watch items beyond the one already embedded. Quiet tickers should cost ~30 tokens, not 300.
  - Novelty 2: L2 = 2 bullets (THESIS NODE + one of VARIANT PERCEPTION or SECOND-ORDER, whichever has content). L3 = 2 bullets (THESIS STATUS + WHAT TO WATCH). Skip CREDIBILITY CHECK and COST OF IGNORING unless there is something specific to say.
  - Novelty 3: standard 4 bullets in L2, 3 bullets in L3.
  - Novelty 4-5: standard bullets, AND prefix L3 bullet 1 with [ESCALATE]. [ESCALATE] is MANDATORY at novelty ≥4 — it is how the PM knows to read this name first.
Token discipline: A row's length should correlate with its novelty score. A novelty-1 row that takes as many tokens as a novelty-4 row is a prompt-following failure. The PM's scan speed depends on visual weight matching informational weight.

── INDUSTRY SUMMARY ROW (required — always last) ──
AFTER all ticker rows, produce ONE industry-level <tr> with the industry name in the first <td>.
This row exists to tell the PM what is NEW at the sector level — not to teach the sector. Assume the PM knows the sector's base thesis (what the industry does, what its main variables are, what recent cycles looked like). Do not restate it.
  - L1: up to 3 bullets on the biggest sector-wide moves in the last 24h. USE Source 1C (price data) to report actual 1-day changes for key commodities/rates (e.g., "BTC +4.2% to $68,400" / "WTI -2.1% to $71.50"). Include the most important sector-wide event from Source 4 (regulatory action, commodity move, geopolitical event). Be specific. Same staleness rule as ticker rows.
  - L2: 2-3 bullets on the DELTA in macro drivers. Which variable's influence changed today? Did the sector move into a new phase of its cycle, or accelerate/decelerate within the existing phase? Ban: generic cycle descriptions ("hash price at post-halving low," "rates remain elevated") that were true last week too.
  - L3: 2-3 bullets. Net sector stance (tailwind/headwind/neutral) AS A DELTA from prior day's stance, plus the one named data point that would flip it. If sector status = NEW (structural shift), flag [ESCALATE] here.
If the sector-level classification is NEW or REFUTE, at least one ticker below must inherit that signal — a structural sector shift cannot coexist with UNCHANGED across every name.
For the industry row, add style="background:#e8ecf0; font-weight:700;" to each <td> so it is visually distinct from ticker rows.

── SELF-CHECK BEFORE OUTPUTTING (run silently) ──
1. Is every L1 fact dated {today} or {yesterday}? Strike any that aren't.
2. Is every THESIS STATUS of RECONFIRM tied to a specific piece of new evidence named in L1 or L2? If not, change to UNCHANGED.
3. Does every SECOND-ORDER bullet name a specific entity (not a category)? If not, cut the bullet.
4. Does every WHAT TO WATCH bullet have one item and one date? If not, rewrite.
5. If the sector row is NEW/REFUTE, does at least one ticker show NEW or REFUTE? If not, rescore.
6. If any ticker is novelty ≥4, is [ESCALATE] prefixed to its L3 bullet 1? If not, add it.

── RULES ──
Banned words: could, may, might, potentially, likely, appears, seems, we believe, it is worth, notably, importantly.
Banned phrases (filler): "options flow noted," "volume elevated," "watch for further developments," "next earnings" (without date), "peers," "hyperscalers," "the industry."
Banned: repeating the same fact across L1/L2/L3. Each level does different work.
Banned: passive summarization — every bullet must express a judgment or a concrete datapoint.
Banned: using RECONFIRM when no new evidence arrived — that is UNCHANGED.
Do NOT issue buy/sell/avoid recommendations. The PM synthesizes and decides.

Tags allowed: <tr>, <td>, <ul>, <li> only."""


# ─── Helpers ────────────────────────────────────────────────

def _extract_text(message) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def _retry(fn):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Rate limited (attempt {attempt+1}). Waiting {delay}s...", flush=True)
                time.sleep(delay)
            else:
                raise


# ─── Analysis ───────────────────────────────────────────────

def analyze_batch(batch: dict) -> str:
    """Send all five source inputs to Opus for synthesis."""
    industry = batch["industry"]
    tickers = [t["ticker"] for t in batch["tickers"]]
    ticker_list = ", ".join(tickers)
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()

    # Inject dates directly into system prompt — no cross-referencing needed
    system_prompt = ANALYSIS_SYSTEM_TEMPLATE.format(
        today=today_str,
        yesterday=yesterday_str,
    )

    # Build the facts context from EDGAR + Finnhub
    edgar_context = ""
    finnhub_context = ""
    for t in batch["tickers"]:
        edgar_facts = [f for f in t["facts"] if f.get("verified")]
        finnhub_facts = [f for f in t["facts"] if "Finnhub" in f.get("source", "")]
        other_facts = [f for f in t["facts"] if not f.get("verified") and "Finnhub" not in f.get("source", "")]

        if edgar_facts:
            lines = [f"  - {f['headline']} ({f['date']}, {f['source']})" for f in edgar_facts]
            edgar_context += f"\n{t['ticker']}:\n" + "\n".join(lines)
        else:
            edgar_context += f"\n{t['ticker']}: No EDGAR filings."

        if finnhub_facts:
            lines = [f"  - {f['headline']} ({f['date']}, {f['source']})" for f in finnhub_facts]
            finnhub_context += f"\n{t['ticker']}:\n" + "\n".join(lines)

    user_prompt = (
        f"Industry: {industry}\n"
        f"Tickers (in output order): {ticker_list}\n\n"
        f"=== SOURCE 1: SEC EDGAR Filings (verified — ground truth) ==={edgar_context}\n\n"
        f"=== SOURCE 1B: Finnhub News Headlines (aggregated from Bloomberg, Reuters, etc.) ===\n"
        f"{finnhub_context if finnhub_context else 'No Finnhub articles in window.'}\n\n"
        f"=== SOURCE 1C: Price Data (Finnhub Quotes — 1-day change) ===\n"
        f"{batch.get('price_summary', 'No quote data.')}\n\n"
        f"=== SOURCE 2: X.com Narrative & Sentiment (Grok — crowd psychology) ===\n"
        f"{batch.get('grok_raw', 'No data')}\n\n"
        f"=== SOURCE 3: Financial News Search (Sonnet — reported events) ===\n"
        f"{batch.get('sonnet_raw', 'No data')}\n\n"
        f"=== SOURCE 3B: Premium News (Bloomberg, Reuters, WSJ, FT — targeted per-ticker) ===\n"
        f"{batch.get('premium_raw', 'No premium search performed.')}\n\n"
        f"=== SOURCE 4: Industry Context (regulatory, competitor, macro, mgmt — last 7 days) ===\n"
        f"{batch.get('context_raw', 'No data')}\n\n"
        f"=== SOURCE 5: Verification Results (fact-checks of X.com claims) ===\n"
        f"{batch.get('verification_raw', 'No verification performed.')}\n\n"
        f"Synthesize all sources. Produce one <tr> per ticker in the order listed, "
        f"then the INDUSTRY SUMMARY ROW last (use Source 1C price data "
        f"for L1 price moves), then one <tr> per ticker in the order listed. "
        f"High-signal X.com claims can go in L1 tagged [X.com] or [verified by search] "
        f"if Source 5 confirmed them. Refuted claims should be flagged in L2. "
        f"Output rows only."
    )

    def _call():
        return claude_client.messages.create(
            model=ANALYSIS_MODEL,
            max_tokens=4500,
            temperature=0.3,
            messages=[{"role": "user", "content": user_prompt}],
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )

    message = _retry(_call)
    opus_tracker.add(message)

    usage = getattr(message, "usage", None)
    if usage is not None:
        cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        print(
            f"    [OPUS usage] input={input_tokens} "
            f"cache_write={cache_created} cache_read={cache_read} "
            f"output={output_tokens}",
            flush=True,
        )

    if message.stop_reason == "max_tokens":
        print(f"  WARNING: analysis hit max_tokens — rows may be truncated", flush=True)

    return _extract_text(message)


# ─── Cost summary ──────────────────────────────────────────

_PRICING = {
    "sonnet_in":         1.00,    # was 3.00 — Haiku charges $1/M input
    "sonnet_out":        5.00,    # was 15.00 — Haiku charges $5/M output
    "sonnet_cache_w":    1.25,    # was 3.75 — Haiku charges $1.25/M cache write
    "sonnet_cache_r":    0.10,    # was 0.30 — Haiku charges $0.10/M cache read
    "sonnet_search":     0.01,    # unchanged — web search is $10/1K regardless of model
    "opus_in":           5.00,
    "opus_out":         25.00,
    "opus_cache_w":      6.25,
    "opus_cache_r":      0.50,
    "grok_in":           0.20,
    "grok_out":          0.50,
    "grok_tool":         0.005,
}


def build_cost_summary(gather_usage: dict, opus_usage: dict) -> str:
    """Build an HTML cost/token summary block for the email footer."""
    P = _PRICING

    s_in  = gather_usage.get("sonnet_input_tokens", 0)
    s_out = gather_usage.get("sonnet_output_tokens", 0)
    s_cw  = gather_usage.get("sonnet_cache_write_tokens", 0)
    s_cr  = gather_usage.get("sonnet_cache_read_tokens", 0)
    s_srch = gather_usage.get("sonnet_search_calls", 0)
    g_in  = gather_usage.get("grok_input_tokens", 0)
    g_out = gather_usage.get("grok_output_tokens", 0)
    g_tc  = gather_usage.get("grok_tool_calls", 0)
    o_in  = opus_usage.get("opus_input_tokens", 0)
    o_out = opus_usage.get("opus_output_tokens", 0)
    o_cw  = opus_usage.get("opus_cache_write_tokens", 0)
    o_cr  = opus_usage.get("opus_cache_read_tokens", 0)

    sonnet_cost = (
        (s_in / 1e6 * P["sonnet_in"])
        + (s_out / 1e6 * P["sonnet_out"])
        + (s_cw / 1e6 * P["sonnet_cache_w"])
        + (s_cr / 1e6 * P["sonnet_cache_r"])
        + (s_srch * P["sonnet_search"])
    )
    grok_cost = (
        (g_in / 1e6 * P["grok_in"])
        + (g_out / 1e6 * P["grok_out"])
        + (g_tc * P["grok_tool"])
    )
    opus_cost = (
        (o_in / 1e6 * P["opus_in"])
        + (o_out / 1e6 * P["opus_out"])
        + (o_cw / 1e6 * P["opus_cache_w"])
        + (o_cr / 1e6 * P["opus_cache_r"])
    )
    total = sonnet_cost + grok_cost + opus_cost

    def _row(label, in_tok, out_tok, cache_w, cache_r, extras, cost):
        extras_str = ""
        if extras:
            extras_str = " | ".join(extras)
        return (
            f"<tr>"
            f"<td style='padding:4px 8px;'>{label}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{in_tok:,}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{out_tok:,}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{cache_w:,}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{cache_r:,}</td>"
            f"<td style='padding:4px 8px;text-align:right;font-size:11px'>{extras_str}</td>"
            f"<td style='padding:4px 8px;text-align:right;font-weight:600'>${cost:.4f}</td>"
            f"</tr>"
        )

    return f"""
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid #ddd;">
      <h3 style="font-size:13px;color:#888;margin:0 0 8px;">API usage &amp; cost</h3>
      <table style="font-size:12px;border-collapse:collapse;width:100%;">
        <tr style="color:#888;border-bottom:1px solid #eee;">
          <th style="padding:4px 8px;text-align:left">Model</th>
          <th style="padding:4px 8px;text-align:right">Input</th>
          <th style="padding:4px 8px;text-align:right">Output</th>
          <th style="padding:4px 8px;text-align:right">Cache write</th>
          <th style="padding:4px 8px;text-align:right">Cache read</th>
          <th style="padding:4px 8px;text-align:right">Tools</th>
          <th style="padding:4px 8px;text-align:right">Cost</th>
        </tr>
        {_row("Haiku 4.5 (search)", s_in, s_out, s_cw, s_cr, [f"{s_srch} searches"], sonnet_cost)}
        {_row("Grok 4.1 Fast", g_in, g_out, 0, 0, [f"{g_tc} tool calls"], grok_cost)}
        {_row("Opus 4.5 (analysis)", o_in, o_out, o_cw, o_cr, [], opus_cost)}
        <tr style="border-top:1px solid #ddd;font-weight:600;">
          <td style="padding:6px 8px;" colspan="6">Total</td>
          <td style="padding:6px 8px;text-align:right">${total:.4f}</td>
        </tr>
      </table>
      <p style="font-size:11px;color:#aaa;margin:8px 0 0;">
        Monthly est. (1&times;/day): ${total * 30:.2f} &mdash;
        Opus: {(o_in + o_out + o_cw + o_cr):,} tok
        | Sonnet: {(s_in + s_out + s_cw + s_cr):,} tok
        | Grok: {(g_in + g_out):,} tok
      </p>
    </div>"""


# ─── HTML ───────────────────────────────────────────────────

def build_html(batch_results: list[dict], cost_html: str = "") -> str:
    today_str = date.today().strftime("%B %d, %Y")
    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")

    all_rows = ""
    audit_sections = ""

    for result in batch_results:
        all_rows += result["analysis_html"]

        edgar_summary = result.get("edgar_summary", "")
        if edgar_summary:
            audit_sections += f"<b>{result['industry']}:</b> {edgar_summary}<br>"

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; color: #111; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #1a1a2e; color: white; padding: 8px 12px; text-align: left; }}
  td {{ border: 1px solid #ddd; padding: 8px 12px; vertical-align: top; max-width: 300px; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  ul {{ margin: 0; padding-left: 16px; }}
  li {{ margin-bottom: 4px; }}
  h2 {{ color: #1a1a2e; margin-top: 0; }}
  .audit {{ background: #f0f8ff; border: 1px solid #d0e0f0; border-radius: 4px;
            padding: 8px 12px; margin-bottom: 12px; font-size: 11px; color: #555; }}
</style>
</head>
<body>
<h2>Daily Research Brief &mdash; {today_str}</h2>
<p style="font-size:11px; color:#888;">Generated {now_utc} | Pipeline: EDGAR &rarr; Finnhub &rarr; Grok &rarr; Sonnet &rarr; Premium (BBG/Reuters/WSJ/FT) &rarr; Context &rarr; Verify &rarr; Opus</p>

<div class="audit">
  <b>EDGAR filings detected:</b><br>
  {audit_sections if audit_sections else "None in window"}
</div>

<table>
  <tr>
    <th>Name</th>
    <th>Level 1 &mdash; News (Last 24h)</th>
    <th>Level 2 &mdash; Materiality Analysis</th>
    <th>Level 3 &mdash; So What</th>
  </tr>
  {all_rows}
</table>
{cost_html}
</body>
</html>"""


# ─── Email (Gmail SMTP) ───────────────────────────────────

def send_email(html: str) -> None:
    """Send the briefing email via Gmail SMTP with an App Password."""
    subject = f"Daily Research Brief — {date.today().strftime('%B %d, %Y')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())

    print(f"Email sent to {GMAIL_ADDRESS} via Gmail SMTP.", flush=True)


# ─── Main ───────────────────────────────────────────────────

def find_latest_facts() -> Path:
    files = sorted(DATA_DIR.glob("facts_*.json"), reverse=True)
    if not files:
        print(f"No facts files in {DATA_DIR}/. Run gather_news.py first.", flush=True)
        sys.exit(1)
    return files[0]


def _process_one_batch(batch: dict) -> dict:
    """Analyze a single batch and build its result dict."""
    print(f"  Analyzing: {batch['industry']}", flush=True)
    analysis_html = analyze_batch(batch)

    edgar_parts = []
    for t in batch["tickers"]:
        verified = [f for f in t["facts"] if f.get("verified")]
        if verified:
            forms = ", ".join(f["headline"].split(" filed")[0] for f in verified)
            edgar_parts.append(f"{t['ticker']}: {forms}")

    return {
        "industry": batch["industry"],
        "analysis_html": analysis_html,
        "edgar_summary": " | ".join(edgar_parts),
    }


def run(facts_path: Path = None) -> str:
    if facts_path is None:
        facts_path = find_latest_facts()

    print(f"Reading: {facts_path}", flush=True)
    with open(facts_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        all_batches = raw
        gather_usage = {}
    else:
        all_batches = raw["batches"]
        gather_usage = raw.get("gather_usage", {})

    print(f"Analyzing {len(all_batches)} batches (2 at a time)...", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch_results = list(pool.map(_process_one_batch, all_batches))

    opus_usage = opus_tracker.snapshot()
    cost_html = build_cost_summary(gather_usage, opus_usage)

    total_tok = (
        gather_usage.get("sonnet_input_tokens", 0)
        + gather_usage.get("sonnet_output_tokens", 0)
        + gather_usage.get("grok_input_tokens", 0)
        + gather_usage.get("grok_output_tokens", 0)
        + opus_usage.get("opus_input_tokens", 0)
        + opus_usage.get("opus_output_tokens", 0)
    )
    print(
        f"  Total tokens: {total_tok:,} | "
        f"Opus: {opus_usage['opus_input_tokens']:,} in / "
        f"{opus_usage['opus_output_tokens']:,} out",
        flush=True,
    )

    html = build_html(batch_results, cost_html)

    print("Sending email...", flush=True)
    send_email(html)

    print("Done.", flush=True)
    return html


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(path)
