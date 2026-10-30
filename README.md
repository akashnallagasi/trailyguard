# TavilyGuard AI

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron--3-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Nebius Token Factory](https://img.shields.io/badge/Nebius-Token%20Factory-5B8DEF)](https://tokenfactory.nebius.com/)
[![Tavily](https://img.shields.io/badge/Web%20Intel-Tavily-0EA5E9)](https://tavily.com)

**Autonomous Enterprise Risk & Web Intelligence Platform** for the **Nebius x NVIDIA Global AI Hackathon**.

TavilyGuard AI combines **real-time Tavily web search** with **NVIDIA Nemotron-3** reasoning served through the **Nebius Token Factory** OpenAI-compatible API. Operators get an executive risk brief, competitive SWOT, and an interactive source explorer in one Streamlit workspace.

## Overview

Security, policy, and strategy teams need live web evidence — not stale training cutoffs. TavilyGuard AI:

1. Retrieves ranked, domain-filterable context from **Tavily**.
2. Sends that evidence to **NVIDIA Nemotron-3 Super** (deep reasoning) or **Nemotron-3 Nano** (fast actions) on **Nebius Token Factory**.
3. Renders a four-section intelligence product: executive summary, market findings, risk & compliance factors, and actionable recommendations.
4. Exports a clean Markdown dossier for sharing.

## Architecture

```
Operator (Streamlit)
        │
        ├─ Tavily Search API  ──►  live sources + relevance scores
        │
        └─ Nebius Token Factory  ──►  NVIDIA Nemotron-3 Super / Nano
                   │
                   ▼
        Structured brief + Plotly source analytics
```

| Layer | Technology |
| --- | --- |
| LLM engine | OpenAI Python SDK → `https://api.tokenfactory.nebius.com/v1` |
| Primary reasoning model | `nvidia/Nemotron-3-Super-120b-a12b` |
| Fast action model | `nvidia/Nemotron-3-Nano-30b` |
| Web intelligence | Tavily Python SDK (`tavily-python`) |
| UI & analytics | Streamlit + Plotly |

## Features

- **Control panel:** masked Nebius Token Factory and Tavily keys, model toggle, search depth (`basic` / `advanced`), max results (3–10), include-domain filters, live connection validation.
- **Single Target Risk Analysis:** default query `AI Data Privacy Laws & Model Regulations 2026`, Deep Intelligence Scan, four briefing containers, Markdown download.
- **Competitive Analysis:** dual Tavily queries, comparison matrix, side-by-side SWOT.
- **Interactive Source Explorer:** raw Tavily JSON, domain distribution, relevance confidence charts.
- **Hackathon checker:** Apache 2.0 license, Token Factory wiring, NVIDIA Nemotron model IDs, Tavily partnership tag, ASCII architecture diagram.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
```

Set secrets in `.env` or paste them in the sidebar:

```
NEBIUS_API_KEY=your_token_factory_key
TAVILY_API_KEY=your_tavily_key
```

Launch:

```bash
streamlit run app.py
```

Open the printed local URL, validate APIs from the sidebar, then run **Deep Intelligence Scan**.

## Repository layout

```
tavilyguard-ai/
├── app.py              # Streamlit application
├── requirements.txt
├── README.md
├── LICENSE             # Apache License 2.0
└── .env.example
```

## Hackathon notes

This project is built specifically around the sponsor stack:

- **NVIDIA Nemotron-3** models for reasoning quality and latency choice.
- **Nebius Token Factory** as the inference gateway (`OpenAI` client, custom `base_url`).
- **Tavily** as the real-time web intelligence layer (search depth, domain filters, source scores).

Licensed under **Apache License 2.0** so the submission meets typical open-source hackathon requirements.

## License

Copyright 2026 TavilyGuard AI Contributors.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
