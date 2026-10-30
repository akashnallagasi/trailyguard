"""
TavilyGuard AI — Autonomous Enterprise Risk & Web Intelligence Platform

Nebius x NVIDIA Global AI Hackathon submission.

LLM runtime : Nebius Token Factory (OpenAI-compatible) serving NVIDIA Nemotron-3
Web intel   : Tavily Search API
UI          : Streamlit + Plotly
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
MODEL_SUPER = "nvidia/Nemotron-3-Super-120b-a12b"
MODEL_NANO = "nvidia/Nemotron-3-Nano-30b"

MODEL_OPTIONS = {
    "Nemotron-3 Super (Deep Reasoning)": MODEL_SUPER,
    "Nemotron-3 Nano (Fast Speed)": MODEL_NANO,
}

DEFAULT_QUERY = "AI Data Privacy Laws & Model Regulations 2026"
APP_TITLE = "TavilyGuard AI"
APP_SUBTITLE = "Autonomous Enterprise Risk & Web Intelligence Platform"

LICENSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LICENSE")
README_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")

SECTION_KEYS = (
    "executive_summary",
    "key_findings",
    "risks",
    "recommendations",
)


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TavilyGuard AI | Enterprise Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 8% -10%, rgba(56, 189, 248, 0.10), transparent 50%),
            radial-gradient(900px 500px at 100% 0%, rgba(99, 102, 241, 0.12), transparent 45%),
            #070b14;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #080c16 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }

    .hero-wrap {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(79, 70, 229, 0.10));
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 1.4rem 1.6rem 1.2rem 1.6rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
    }

    .hero-kicker {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        color: #7dd3fc;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2.05rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.15;
        margin: 0;
    }

    .hero-sub {
        color: #cbd5e1;
        margin-top: 0.45rem;
        font-size: 1.02rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.85rem;
    }

    .pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(125, 211, 252, 0.35);
        background: rgba(14, 165, 233, 0.12);
        color: #e0f2fe;
    }

    .status-ok {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(52, 211, 153, 0.45);
        color: #a7f3d0;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        font-size: 0.86rem;
    }

    .status-warn {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.4);
        color: #fde68a;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        font-size: 0.86rem;
    }

    .status-err {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(248, 113, 113, 0.4);
        color: #fecaca;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        font-size: 0.86rem;
    }

    .intel-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
    }

    .intel-card h3 {
        margin-top: 0;
        color: #f1f5f9;
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.18);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.55rem 0.75rem;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }

    pre, code {
        font-family: 'IBM Plex Mono', monospace;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults: dict[str, Any] = {
        "nebius_key": os.getenv("NEBIUS_API_KEY", os.getenv("TOKEN_FACTORY_API_KEY", "")),
        "tavily_key": os.getenv("TAVILY_API_KEY", ""),
        "nebius_status": "untested",
        "tavily_status": "untested",
        "status_detail": {},
        "single_report": None,
        "single_sources": None,
        "single_query": DEFAULT_QUERY,
        "compare_report": None,
        "compare_sources": None,
        "explorer_payload": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------------------------------------------------------------------------
# Client factories & live health checks
# ---------------------------------------------------------------------------
def build_llm_client(api_key: str) -> OpenAI:
    """OpenAI SDK pointed at Nebius Token Factory (NVIDIA Nemotron)."""
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key, timeout=120.0)


def build_tavily_client(api_key: str) -> TavilyClient:
    return TavilyClient(api_key=api_key)


def validate_nebius(api_key: str) -> tuple[bool, str]:
    if not api_key.strip():
        return False, "Nebius Token Factory key is missing."
    try:
        client = build_llm_client(api_key)
        # Lightweight round-trip: list models when available, else a 1-token chat.
        try:
            models = client.models.list()
            names = [m.id for m in getattr(models, "data", [])][:8]
            extra = f" Models visible: {', '.join(names)}." if names else ""
            return True, f"Token Factory reachable at {NEBIUS_BASE_URL}.{extra}"
        except Exception:
            client.chat.completions.create(
                model=MODEL_NANO,
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=8,
                temperature=0,
            )
            return True, "Token Factory chat endpoint is healthy."
    except Exception as exc:
        return False, f"Nebius validation failed: {exc}"


def validate_tavily(api_key: str) -> tuple[bool, str]:
    if not api_key.strip():
        return False, "Tavily API key is missing."
    try:
        client = build_tavily_client(api_key)
        result = client.search(query="Tavily API health check", search_depth="basic", max_results=1)
        hits = len(result.get("results", []) if isinstance(result, dict) else [])
        return True, f"Tavily search healthy ({hits} sample hit(s))."
    except Exception as exc:
        return False, f"Tavily validation failed: {exc}"


# ---------------------------------------------------------------------------
# Tavily retrieval
# ---------------------------------------------------------------------------
def parse_domains(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parts = re.split(r"[\s,;]+", raw.strip())
    return [p.strip().lower() for p in parts if p.strip()]


def tavily_search(
    api_key: str,
    query: str,
    *,
    search_depth: str = "advanced",
    max_results: int = 6,
    include_domains: list[str] | None = None,
) -> dict[str, Any]:
    client = build_tavily_client(api_key)
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    return client.search(**payload)


def extract_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results") or []
    sources = []
    for item in results:
        url = item.get("url") or ""
        domain = urlparse(url).netloc.replace("www.", "") if url else "unknown"
        sources.append(
            {
                "title": item.get("title") or "Untitled",
                "url": url,
                "content": item.get("content") or "",
                "score": float(item.get("score") or 0.0),
                "domain": domain,
            }
        )
    return sources


def format_context(sources: list[dict[str, Any]], tavily_answer: str | None = None) -> str:
    blocks = []
    if tavily_answer:
        blocks.append(f"Tavily composite answer:\n{tavily_answer}")
    for i, src in enumerate(sources, start=1):
        score = f"{src['score']:.3f}" if src.get("score") else "n/a"
        blocks.append(
            f"[{i}] {src['title']} | {src['domain']} | relevance={score}\n"
            f"URL: {src['url']}\n{src['content']}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "No web context retrieved."


# ---------------------------------------------------------------------------
# Nemotron reasoning via Token Factory
# ---------------------------------------------------------------------------
def run_nemotron(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    stream_box: Any | None = None,
) -> str:
    """Call NVIDIA Nemotron through Token Factory; optionally stream tokens to Streamlit."""
    client = build_llm_client(api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if stream_box is not None:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=3500,
            stream=True,
        )
        chunks: list[str] = []
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            chunks.append(delta)
            stream_box.markdown("".join(chunks) + "▌")
        content = "".join(chunks).strip()
        stream_box.markdown(content or "_Empty model stream._")
        return content

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=3500,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model response, including fenced blocks."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                return None
    return None


SINGLE_SYSTEM = """You are TavilyGuard AI, an autonomous enterprise risk and web intelligence analyst.
You reason over real-time Tavily search evidence and produce decision-grade briefings.
Never invent URLs. Ground claims in the provided sources. If evidence is thin, say so.
Return ONLY valid JSON with these string keys:
{
  "executive_summary": "3-6 sentence briefing for executives",
  "key_findings": "markdown bullets of market, competitive, and strategic findings",
  "risks": "markdown bullets of legal, compliance, operational, cyber, and reputational risks",
  "recommendations": "markdown numbered actions with owners/time horizons where possible"
}
"""

COMPARE_SYSTEM = """You are TavilyGuard AI, a competitive intelligence officer.
Compare two entities using Tavily evidence only. Be even-handed and specific.
Return ONLY valid JSON:
{
  "comparison_markdown": "GitHub-flavored markdown table comparing the two entities across strategy, product, market, risk, regulation, and momentum",
  "swot_a": {"strengths": "markdown bullets", "weaknesses": "markdown bullets", "opportunities": "markdown bullets", "threats": "markdown bullets"},
  "swot_b": {"strengths": "markdown bullets", "weaknesses": "markdown bullets", "opportunities": "markdown bullets", "threats": "markdown bullets"},
  "verdict": "short strategic verdict and watch-items"
}
"""


def analyze_single_target(
    api_key: str,
    model: str,
    query: str,
    context: str,
    stream_box: Any | None = None,
) -> dict[str, str]:
    user = (
        f"Research query: {query}\n\n"
        f"Real-time web intelligence from Tavily:\n{context}\n\n"
        "Produce the structured risk intelligence JSON now."
    )
    raw = run_nemotron(api_key, model, SINGLE_SYSTEM, user, stream_box=stream_box)
    parsed = extract_json_object(raw)
    if parsed:
        return {k: str(parsed.get(k, "")).strip() for k in SECTION_KEYS}
    return fallback_sections(raw)


def analyze_comparison(
    api_key: str,
    model: str,
    entity_a: str,
    entity_b: str,
    context_a: str,
    context_b: str,
    stream_box: Any | None = None,
) -> dict[str, Any]:
    user = (
        f"Entity A: {entity_a}\nEntity B: {entity_b}\n\n"
        f"=== Tavily evidence for {entity_a} ===\n{context_a}\n\n"
        f"=== Tavily evidence for {entity_b} ===\n{context_b}\n\n"
        "Produce the comparison JSON now."
    )
    raw = run_nemotron(
        api_key,
        model,
        COMPARE_SYSTEM,
        user,
        temperature=0.25,
        stream_box=stream_box,
    )
    parsed = extract_json_object(raw)
    if parsed:
        return parsed
    return {
        "comparison_markdown": raw,
        "swot_a": {},
        "swot_b": {},
        "verdict": "Model returned unstructured text; see comparison body.",
    }


def fallback_sections(raw: str) -> dict[str, str]:
    """Split free-form markdown into the four dashboard containers."""
    patterns = {
        "executive_summary": r"(?:executive summary|summary)",
        "key_findings": r"(?:key (?:market|strategic)? ?(?:&|and)? ?(?:market|strategic)? findings|findings)",
        "risks": r"(?:identified risk|risk & compliance|risks?)",
        "recommendations": r"(?:actionable recommendations|recommendations)",
    }
    positions: list[tuple[int, str]] = []
    for key, pat in patterns.items():
        match = re.search(rf"^#+\s*{pat}.*$", raw, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            positions.append((match.start(), key))
    positions.sort()
    out = {k: "" for k in SECTION_KEYS}
    if not positions:
        out["executive_summary"] = raw
        return out
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
        chunk = raw[start:end]
        chunk = re.sub(r"^#+\s*.*$", "", chunk, count=1, flags=re.MULTILINE).strip()
        out[key] = chunk
    return out


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------
def report_to_markdown(
    query: str,
    model: str,
    sections: dict[str, str],
    sources: list[dict[str, Any]],
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    source_lines = "\n".join(
        f"- [{s['title']}]({s['url']}) — `{s['domain']}` (score {s['score']:.3f})"
        for s in sources
    ) or "- None retrieved"
    return f"""# TavilyGuard AI Intelligence Report

- **Query:** {query}
- **Generated:** {stamp}
- **Reasoning model:** `{model}` via Nebius Token Factory
- **Web intelligence:** Tavily Search API

## Executive Summary
{sections.get('executive_summary') or '_Not generated._'}

## Key Market & Strategic Findings
{sections.get('key_findings') or '_Not generated._'}

## Identified Risk & Compliance Factors
{sections.get('risks') or '_Not generated._'}

## Actionable Recommendations
{sections.get('recommendations') or '_Not generated._'}

## Sources
{source_lines}

---
*Produced by TavilyGuard AI — NVIDIA Nemotron on Nebius Token Factory + Tavily.*
"""


def comparison_to_markdown(
    entity_a: str,
    entity_b: str,
    model: str,
    report: dict[str, Any],
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    swot_a = report.get("swot_a") or {}
    swot_b = report.get("swot_b") or {}

    def swot_block(title: str, swot: dict[str, Any]) -> str:
        return (
            f"### SWOT — {title}\n"
            f"**Strengths**\n{swot.get('strengths', '_n/a_')}\n\n"
            f"**Weaknesses**\n{swot.get('weaknesses', '_n/a_')}\n\n"
            f"**Opportunities**\n{swot.get('opportunities', '_n/a_')}\n\n"
            f"**Threats**\n{swot.get('threats', '_n/a_')}\n"
        )

    return f"""# TavilyGuard AI Competitive Analysis

- **{entity_a} vs {entity_b}**
- **Generated:** {stamp}
- **Reasoning model:** `{model}` via Nebius Token Factory
- **Web intelligence:** dual Tavily queries

## Comparison Matrix
{report.get('comparison_markdown') or '_Not generated._'}

{swot_block(entity_a, swot_a if isinstance(swot_a, dict) else {})}
{swot_block(entity_b, swot_b if isinstance(swot_b, dict) else {})}

## Strategic Verdict
{report.get('verdict') or '_Not generated._'}

---
*Produced by TavilyGuard AI — NVIDIA Nemotron on Nebius Token Factory + Tavily.*
"""


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def require_keys() -> bool:
    missing = []
    if not st.session_state.nebius_key.strip():
        missing.append("Nebius Token Factory API key")
    if not st.session_state.tavily_key.strip():
        missing.append("Tavily API key")
    if missing:
        st.error(
            "Missing credentials: " + ", ".join(missing) +
            ". Enter them in the sidebar (or set `NEBIUS_API_KEY` / `TAVILY_API_KEY`)."
        )
        return False
    return True


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-kicker">Nebius × NVIDIA Global AI Hackathon</div>
          <h1 class="hero-title">🛡️ {APP_TITLE}</h1>
          <p class="hero-sub">{APP_SUBTITLE}</p>
          <div class="badge-row">
            <span class="pill">NVIDIA Nemotron-3</span>
            <span class="pill">Nebius Token Factory</span>
            <span class="pill">Tavily Web Intelligence</span>
            <span class="pill">Apache-2.0</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_html(ok: bool | None, label: str, detail: str) -> str:
    if ok is True:
        klass = "status-ok"
        icon = "●"
        state = "ONLINE"
    elif ok is False:
        klass = "status-err"
        icon = "●"
        state = "FAILED"
    else:
        klass = "status-warn"
        icon = "●"
        state = "UNTESTED"
    return f'<div class="{klass}"><strong>{icon} {label}: {state}</strong><br/>{detail}</div>'


def render_swot(title: str, swot: dict[str, Any]) -> None:
    st.markdown(f"**{title}**")
    c1, c2 = st.columns(2)
    with c1:
        st.success("Strengths")
        st.markdown(swot.get("strengths") or "_No data._")
        st.info("Opportunities")
        st.markdown(swot.get("opportunities") or "_No data._")
    with c2:
        st.warning("Weaknesses")
        st.markdown(swot.get("weaknesses") or "_No data._")
        st.error("Threats")
        st.markdown(swot.get("threats") or "_No data._")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.caption("Credentials never leave this session except to Nebius Token Factory and Tavily.")

    st.session_state.nebius_key = st.text_input(
        "Nebius Token Factory API Key",
        value=st.session_state.nebius_key,
        type="password",
        help="OpenAI-compatible key for https://api.tokenfactory.nebius.com/v1",
        placeholder="tf-...",
    )
    st.session_state.tavily_key = st.text_input(
        "Tavily API Key",
        value=st.session_state.tavily_key,
        type="password",
        help="Real-time web search key from tavily.com",
        placeholder="tvly-...",
    )

    st.markdown("---")
    st.markdown("### 🧠 Reasoning Engine")
    model_label = st.radio(
        "NVIDIA Nemotron model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Super = deeper analysis. Nano = lower latency actions.",
    )
    selected_model = MODEL_OPTIONS[model_label]
    st.code(selected_model, language="text")

    st.markdown("### 🔎 Tavily Search Parameters")
    search_depth = st.toggle(
        "Advanced search depth",
        value=True,
        help="On = Tavily `advanced` (richer context). Off = `basic`.",
    )
    depth_value = "advanced" if search_depth else "basic"
    st.caption(f"Active depth: `{depth_value}`")

    max_results = st.slider("Max results", min_value=3, max_value=10, value=6)
    domain_filters = st.text_input(
        "Search domain filters",
        placeholder="reuters.com, sec.gov, europa.eu",
        help="Optional comma-separated include_domains for Tavily.",
    )
    include_domains = parse_domains(domain_filters)

    st.markdown("---")
    st.markdown("### 📡 System Status")
    if st.button("Validate API connections", use_container_width=True):
        with st.spinner("Pinging Token Factory and Tavily..."):
            n_ok, n_msg = validate_nebius(st.session_state.nebius_key)
            t_ok, t_msg = validate_tavily(st.session_state.tavily_key)
            st.session_state.nebius_status = "ok" if n_ok else "err"
            st.session_state.tavily_status = "ok" if t_ok else "err"
            st.session_state.status_detail = {"nebius": n_msg, "tavily": t_msg}

    n_state = st.session_state.nebius_status
    t_state = st.session_state.tavily_status
    n_ok = True if n_state == "ok" else False if n_state == "err" else None
    t_ok = True if t_state == "ok" else False if t_state == "err" else None
    n_detail = st.session_state.status_detail.get("nebius") or (
        "Key present — click Validate to test Token Factory."
        if st.session_state.nebius_key.strip()
        else "No key configured."
    )
    t_detail = st.session_state.status_detail.get("tavily") or (
        "Key present — click Validate to test Tavily."
        if st.session_state.tavily_key.strip()
        else "No key configured."
    )
    st.markdown(status_html(n_ok, "Nebius Token Factory", n_detail), unsafe_allow_html=True)
    st.markdown(status_html(t_ok, "Tavily Search", t_detail), unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Base URL")
    st.code(NEBIUS_BASE_URL, language="text")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
render_hero()

tab_single, tab_compare, tab_explorer, tab_hackathon = st.tabs(
    [
        "⚡ Single Target Risk Analysis",
        "⚔️ Competitive Analysis",
        "📊 Interactive Source Explorer",
        "🛠️ Hackathon Submission Checker",
    ]
)


# ---- TAB 1 -----------------------------------------------------------------
with tab_single:
    st.subheader("Single Target Risk Analysis")
    st.caption("Tavily retrieves live web context → NVIDIA Nemotron on Token Factory synthesizes a four-section risk brief.")

    query = st.text_input(
        "Research topic / company / security query",
        value=st.session_state.single_query,
        placeholder=DEFAULT_QUERY,
    )
    st.session_state.single_query = query

    run_scan = st.button("Run Deep Intelligence Scan", type="primary", use_container_width=True)

    if run_scan:
        if require_keys() and query.strip():
            try:
                with st.status("Executing intelligence pipeline...", expanded=True) as status:
                    st.write("1/3 Querying Tavily for real-time web context...")
                    payload = tavily_search(
                        st.session_state.tavily_key,
                        query.strip(),
                        search_depth=depth_value,
                        max_results=max_results,
                        include_domains=include_domains or None,
                    )
                    sources = extract_sources(payload)
                    context = format_context(sources, payload.get("answer"))
                    st.write(f"Retrieved {len(sources)} source(s).")

                    st.write(f"2/3 Streaming Nemotron reasoning via Nebius Token Factory (`{selected_model}`)...")
                    stream_box = st.empty()
                    sections = analyze_single_target(
                        st.session_state.nebius_key,
                        selected_model,
                        query.strip(),
                        context,
                        stream_box=stream_box,
                    )

                    st.write("3/3 Rendering structured briefing...")
                    st.session_state.single_report = {
                        "query": query.strip(),
                        "model": selected_model,
                        "sections": sections,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    st.session_state.single_sources = sources
                    st.session_state.explorer_payload = {
                        "mode": "single",
                        "query": query.strip(),
                        "payloads": [payload],
                        "sources": sources,
                    }
                    status.update(label="Scan complete", state="complete")
            except Exception as exc:
                st.exception(exc)
                st.error("Intelligence scan failed. Verify API keys, model access, and network connectivity.")
        elif not query.strip():
            st.warning("Enter a research query before running the scan.")

    report = st.session_state.single_report
    if report:
        sections = report["sections"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sources", len(st.session_state.single_sources or []))
        c2.metric("Depth", depth_value)
        c3.metric("Model", "Super" if "Super" in report["model"] else "Nano")
        c4.metric("Status", "Ready")

        def section_box(title: str, body: str) -> None:
            st.markdown(title)
            with st.container(border=True):
                st.markdown(body or "_Empty._")

        col_a, col_b = st.columns(2)
        with col_a:
            section_box("#### 📌 Executive Summary", sections.get("executive_summary", ""))
            section_box("#### ⚠️ Identified Risk & Compliance Factors", sections.get("risks", ""))
        with col_b:
            section_box("#### 💡 Key Market & Strategic Findings", sections.get("key_findings", ""))
            section_box("#### 🎯 Actionable Recommendations", sections.get("recommendations", ""))

        md = report_to_markdown(
            report["query"],
            report["model"],
            sections,
            st.session_state.single_sources or [],
        )
        st.download_button(
            "Download report (.md)",
            data=md.encode("utf-8"),
            file_name=f"tavilyguard-report-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        with st.expander("Cited sources"):
            for src in st.session_state.single_sources or []:
                st.markdown(
                    f"- [{src['title']}]({src['url']}) — `{src['domain']}` · score **{src['score']:.3f}**"
                )
    else:
        st.info("Run a Deep Intelligence Scan to populate the four briefing containers.")


# ---- TAB 2 -----------------------------------------------------------------
with tab_compare:
    st.subheader("Competitive Analysis (Side-by-Side)")
    st.caption("Dual Tavily queries feed Nemotron to produce a comparison matrix and SWOT pair.")

    left, right = st.columns(2)
    with left:
        entity_a = st.text_input("Entity / topic A", value="OpenAI", placeholder="Company A")
    with right:
        entity_b = st.text_input("Entity / topic B", value="Anthropic", placeholder="Company B")

    run_compare = st.button("Run Competitive Intelligence", type="primary", use_container_width=True)

    if run_compare:
        if require_keys() and entity_a.strip() and entity_b.strip():
            try:
                with st.status("Running dual-source competitive pipeline...", expanded=True) as status:
                    st.write("1/3 Parallel Tavily retrieval...")

                    def _search(name: str) -> tuple[str, dict[str, Any]]:
                        return name, tavily_search(
                            st.session_state.tavily_key,
                            f"{name} strategy product market regulation risk 2026",
                            search_depth=depth_value,
                            max_results=max_results,
                            include_domains=include_domains or None,
                        )

                    payloads: dict[str, dict[str, Any]] = {}
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(_search, entity_a.strip()), pool.submit(_search, entity_b.strip())]
                        for fut in as_completed(futures):
                            name, payload = fut.result()
                            payloads[name] = payload

                    sources_a = extract_sources(payloads[entity_a.strip()])
                    sources_b = extract_sources(payloads[entity_b.strip()])
                    ctx_a = format_context(sources_a, payloads[entity_a.strip()].get("answer"))
                    ctx_b = format_context(sources_b, payloads[entity_b.strip()].get("answer"))
                    st.write(f"{entity_a}: {len(sources_a)} sources · {entity_b}: {len(sources_b)} sources")

                    st.write(f"2/3 Nemotron comparison on `{selected_model}`...")
                    stream_box = st.empty()
                    cmp_report = analyze_comparison(
                        st.session_state.nebius_key,
                        selected_model,
                        entity_a.strip(),
                        entity_b.strip(),
                        ctx_a,
                        ctx_b,
                        stream_box=stream_box,
                    )
                    st.session_state.compare_report = {
                        "a": entity_a.strip(),
                        "b": entity_b.strip(),
                        "model": selected_model,
                        "report": cmp_report,
                    }
                    all_sources = [
                        {**s, "entity": entity_a.strip()} for s in sources_a
                    ] + [
                        {**s, "entity": entity_b.strip()} for s in sources_b
                    ]
                    st.session_state.compare_sources = all_sources
                    st.session_state.explorer_payload = {
                        "mode": "compare",
                        "query": f"{entity_a.strip()} vs {entity_b.strip()}",
                        "payloads": [payloads[entity_a.strip()], payloads[entity_b.strip()]],
                        "sources": all_sources,
                    }
                    st.write("3/3 Rendering matrix and SWOT...")
                    status.update(label="Competitive analysis complete", state="complete")
            except Exception as exc:
                st.exception(exc)
                st.error("Competitive analysis failed. Check keys, quotas, and entity names.")
        else:
            if not (entity_a.strip() and entity_b.strip()):
                st.warning("Provide both Entity A and Entity B.")

    cmp = st.session_state.compare_report
    if cmp:
        st.markdown("#### Comparison Matrix")
        st.markdown(cmp["report"].get("comparison_markdown") or "_Empty._")

        col_a, col_b = st.columns(2)
        with col_a:
            swot = cmp["report"].get("swot_a") or {}
            render_swot(cmp["a"], swot if isinstance(swot, dict) else {})
        with col_b:
            swot = cmp["report"].get("swot_b") or {}
            render_swot(cmp["b"], swot if isinstance(swot, dict) else {})

        st.markdown("#### Strategic Verdict")
        st.info(cmp["report"].get("verdict") or "No verdict returned.")

        md = comparison_to_markdown(cmp["a"], cmp["b"], cmp["model"], cmp["report"])
        st.download_button(
            "Download comparison (.md)",
            data=md.encode("utf-8"),
            file_name=f"tavilyguard-compare-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.info("Run Competitive Intelligence to generate the matrix and SWOT boards.")


# ---- TAB 3 -----------------------------------------------------------------
with tab_explorer:
    st.subheader("Interactive Source Explorer")
    st.caption("Inspect raw Tavily JSON, domain mix, and relevance confidence from the latest scan.")

    payload_pack = st.session_state.explorer_payload
    if not payload_pack:
        st.warning("No Tavily payload in session yet. Run a scan in Tab 1 or Tab 2.")
    else:
        sources = payload_pack.get("sources") or []
        st.markdown(f"**Latest query:** `{payload_pack.get('query')}` · mode `{payload_pack.get('mode')}`")

        m1, m2, m3 = st.columns(3)
        scores = [float(s.get("score") or 0) for s in sources]
        m1.metric("Result rows", len(sources))
        m2.metric("Unique domains", len({s.get("domain") for s in sources}))
        m3.metric("Avg relevance", f"{(sum(scores) / len(scores)):.3f}" if scores else "n/a")

        if sources:
            domain_counts = Counter(s.get("domain") or "unknown" for s in sources)
            chart_df = {
                "domain": list(domain_counts.keys()),
                "results": list(domain_counts.values()),
            }
            fig = px.bar(
                chart_df,
                x="domain",
                y="results",
                title="Tavily domain distribution",
                color="results",
                color_continuous_scale="Tealgrn",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "IBM Plex Sans"},
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_score = px.bar(
                {
                    "source": [s.get("title", "")[:48] for s in sources],
                    "relevance": [float(s.get("score") or 0) for s in sources],
                    "domain": [s.get("domain") for s in sources],
                },
                x="relevance",
                y="source",
                color="domain",
                orientation="h",
                title="Relevance confidence scores",
            )
            fig_score.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(360, 28 * len(sources)),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_score, use_container_width=True)

            st.markdown("#### Source table")
            st.dataframe(
                [
                    {
                        "title": s.get("title"),
                        "domain": s.get("domain"),
                        "score": round(float(s.get("score") or 0), 4),
                        "url": s.get("url"),
                        "entity": s.get("entity", payload_pack.get("mode")),
                    }
                    for s in sources
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Raw Tavily JSON")
        for i, raw in enumerate(payload_pack.get("payloads") or [], start=1):
            with st.expander(f"Payload {i}"):
                st.json(raw)


# ---- TAB 4 -----------------------------------------------------------------
with tab_hackathon:
    st.subheader("Hackathon Submission Checker & System Info")
    st.caption("Built for the Nebius x NVIDIA Global AI Hackathon with Tavily as the official web intelligence partner.")

    def file_contains(path: str, needle: str) -> bool:
        try:
            with open(path, encoding="utf-8") as handle:
                return needle.lower() in handle.read().lower()
        except OSError:
            return False

    license_ok = os.path.isfile(LICENSE_PATH) and file_contains(LICENSE_PATH, "Apache License")
    readme_ok = os.path.isfile(README_PATH)
    token_factory_tag = file_contains(__file__, "api.tokenfactory.nebius.com")
    nemotron_tag = file_contains(__file__, "nvidia/Nemotron-3-Super-120b-a12b")
    tavily_tag = file_contains(__file__, "tavily") and file_contains(README_PATH, "Tavily")
    apache_readme = file_contains(README_PATH, "Apache")

    checks = [
        ("GitHub / OSS License file present (Apache 2.0)", license_ok),
        ("README documents Apache-2.0 licensing", apache_readme and readme_ok),
        ("Nebius Token Factory base URL wired in app", token_factory_tag),
        ("NVIDIA Nemotron-3 Super model id configured", nemotron_tag),
        ("Tavily SDK + README partnership tag", tavily_tag),
        ("Nemotron Nano fast-action model available", file_contains(__file__, MODEL_NANO)),
        ("Plotly analytics explorer implemented", file_contains(__file__, "plotly")),
    ]

    passed = sum(1 for _, ok in checks if ok)
    st.metric("Checklist score", f"{passed}/{len(checks)}")

    for label, ok in checks:
        st.markdown(f"{'✅' if ok else '❌'}  {label}")

    st.markdown("#### System architecture")
    st.code(
        r"""
┌──────────────┐    keys     ┌──────────────────────────────────────────┐
│  Streamlit   │────────────▶│           Control Plane (sidebar)        │
│  Operator    │             │  Nebius key · Tavily key · model · depth │
└──────┬───────┘             └───────────────────┬──────────────────────┘
       │                                         │
       │  query                                  ▼
       │                         ┌───────────────────────────┐
       ├────────────────────────▶│   Tavily Search API       │
       │                         │   basic | advanced depth  │
       │                         └─────────────┬─────────────┘
       │                                       │ evidence JSON
       │                                       ▼
       │                         ┌───────────────────────────┐
       │                         │  Nebius Token Factory     │
       │                         │  OpenAI-compatible /v1    │
       │                         │  NVIDIA Nemotron-3 Super  │
       │                         │  NVIDIA Nemotron-3 Nano   │
       │                         └─────────────┬─────────────┘
       │                                       │ structured JSON brief
       ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Dashboard                                                          │
│  Tab1 Risk brief · Tab2 Compare/SWOT · Tab3 Plotly sources · Tab4  │
└─────────────────────────────────────────────────────────────────────┘
        """,
        language="text",
    )

    st.markdown("#### Runtime configuration")
    st.json(
        {
            "llm_base_url": NEBIUS_BASE_URL,
            "primary_model": MODEL_SUPER,
            "fast_model": MODEL_NANO,
            "active_model": selected_model,
            "tavily_depth": depth_value,
            "max_results": max_results,
            "include_domains": include_domains,
            "hackathon": "Nebius x NVIDIA Global AI Hackathon",
            "web_intelligence": "Tavily",
            "license": "Apache-2.0",
        }
    )

    st.markdown("#### Sponsor stack")
    s1, s2, s3 = st.columns(3)
    s1.success("**NVIDIA Nemotron-3**\nSuper 120B reasoning + Nano 30B actions")
    s2.info("**Nebius Token Factory**\nOpenAI SDK · `api.tokenfactory.nebius.com/v1`")
    s3.warning("**Tavily**\nReal-time web search & source scores")


st.markdown(
    '<div class="footer-note">TavilyGuard AI · Apache License 2.0 · NVIDIA Nemotron on Nebius Token Factory · Tavily web intelligence</div>',
    unsafe_allow_html=True,
)
