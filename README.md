# Product Review Intelligence

> AI-powered review intelligence platform that ingests **Play Store** and **App Store** reviews, extracts user pain points and recurring themes using **Groq LLMs**, and delivers a weekly **Google Doc** report + **Gmail** digest — automatically.

Stop drowning in raw reviews. Get a leadership-ready pulse every week.

---

## 1. Product Overview

Mobile product teams ship fast — but the signal from end-users is locked inside thousands of unstructured store reviews. PMs either:

- Read a random sample and call it "research" (lossy, biased).
- Hire someone to read everything weekly (expensive, doesn't scale).
- Skip it entirely and rely on NPS or in-app surveys (lagging indicators).

**Product Review Intelligence** closes that loop by turning raw store reviews into a structured, actionable, leadership-ready pulse — delivered to a Google Doc and to your inbox every week.

### Problem Statement

PMs, founders, and CX teams need a **continuous, structured, low-effort** way to understand what users are complaining about, what they love, and what to ship next — without manually trawling store pages.

### Target Users

- **Product Managers** — for weekly review themes and prioritization input.
- **Founders / early teams** — for a recurring "voice of customer" digest.
- **Growth & CX leads** — for sentiment trends and recurring pain points.
- **Engineering managers** — for surfacing crash/bug patterns hidden in reviews.

### Product Value

- **Zero manual reading.** The pipeline runs end-to-end on a single API call.
- **Structured output.** Themes, executive summary, action items, and direct user quotes — not a wall of text.
- **Cross-platform.** Same insights from Play Store and App Store reviews.
- **Delivered where work happens.** Reports land in Google Docs + Gmail, ready to share.

### Example Use Cases

- "Send my product team a weekly Monday digest of the top user complaints."
- "Track top 5 recurring themes across both Android and iOS reviews."
- "Generate next-sprint candidates from this week's review pain points."
- "Watch for sudden spikes in negative feedback after a release."

---

## 2. Key Features

| Capability           | Description                                                                          |
| -------------------- | ------------------------------------------------------------------------------------ |
| Play Store ingestion | Pulls fresh reviews via `google-play-scraper` with a configurable lookback window.   |
| App Store ingestion  | Pulls reviews from Apple's public RSS feed (no auth, no key, no scraping fragility). |
| Preprocessing        | Normalizes, deduplicates, trims, and validates each review before LLM ingest.        |
| AI summarization     | Groq LLM (`llama-3.1-8b-instant` by default) produces structured per-chunk JSON.     |
| Theme extraction     | Aggregates and ranks recurring themes across chunks, with frequency and quotes.      |
| Executive summary    | Leadership-facing 250-word digest written for non-technical readers.                 |
| Action items         | Concrete, prioritized next-step recommendations for the product team.                |
| Google Docs delivery | Renders the report into a target Google Doc via the MCP integration server.          |
| Gmail delivery       | Sends the executive summary as an email with a deep link to the full Doc.            |
| Config-driven        | Static knobs (chunk size, retries, lookback) in code; secrets in `.env`.             |
| Render-ready         | One-click `render.yaml` Blueprint deploy, no Docker required.                        |
| Structured logging   | `structlog` JSON logs with stable event keys for observability.                      |
| Fallback models      | Auto-retries with secondary Groq models on 429s or upstream failures.                |

---

## 3. Architecture / Flow

**End-to-end chain:**

```
GitHub Actions Scheduler ─▶ Render API ─▶ AI Summarization Pipeline ─▶ Google Docs + Email Delivery
```

```
┌────────────────────────────────────────────────────────────────────────────┐
│  GitHub Actions (cron: Mon 06:00 IST)  ───▶  POST /run-weekly-pulse        │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │       Weekly Pulse Orchestrator      │
                └──────────────────┬──────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐        ┌────────▼────────┐         ┌───────▼────────┐
│ Play Store     │        │  App Store      │         │  Time-window   │
│ Ingestion      │        │  Ingestion      │         │  resolver      │
└───────┬────────┘        └────────┬────────┘         └────────────────┘
        │                          │
        └──────────────┬───────────┘
                       │
              ┌────────▼─────────┐
              │  Preprocessing   │  ← dedupe · normalize · validate · trim
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │     Chunker      │  ← token-aware batching
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Groq LLM        │  ← per-chunk structured summarization
              │  (+ fallbacks)   │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │   Aggregator     │  ← merge chunks · rank themes · pick quotes
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Pulse Builder   │  ← exec summary · action items · formatting
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  MCP Client      │  ← Google Docs + Gmail via remote MCP server
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Google Doc      │
              │  + Email digest  │
              └──────────────────┘
```

### Stage Breakdown

| Stage               | What it does                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Ingestion**       | Pulls Play Store + App Store reviews in parallel via `asyncio.gather`. Resolves the lookback window (default: 12 weeks). |
| **Preprocessing**   | Strips boilerplate, removes duplicates by hash, validates schema, caps body length.                                      |
| **Chunking**        | Groups reviews into token-budgeted chunks (target ~3,500 tokens) so each LLM call stays cheap and well within context.   |
| **Summarization**   | Calls Groq per chunk with a structured prompt. Returns JSON with themes, sentiment, sample quotes.                       |
| **Aggregation**     | Merges per-chunk outputs, deduplicates themes, ranks by frequency, selects best quotes.                                  |
| **Report Building** | Composes the executive summary, top themes (cap: 5), top action items (cap: 3), formatted Doc body.                      |
| **Delivery**        | Calls the MCP server to write the Google Doc and send the email digest.                                                  |

---

## 4. Tech Stack

| Layer             | Technology                                                                            |
| ----------------- | ------------------------------------------------------------------------------------- |
| API Framework     | FastAPI                                                                               |
| Language          | Python 3.11+                                                                          |
| Validation        | Pydantic v2 + `pydantic-settings`                                                     |
| Async runtime     | `asyncio` (parallel ingest, bounded concurrency on LLM calls)                         |
| HTTP client       | `httpx` (async)                                                                       |
| Logging           | `structlog` (JSON structured logs)                                                    |
| LLM               | Groq API (`llama-3.1-8b-instant` primary; `llama3-8b-8192`, `gemma2-9b-it` fallbacks) |
| Play Store source | `google-play-scraper`                                                                 |
| App Store source  | Apple RSS feed                                                                        |
| Doc delivery      | Google Docs API (via MCP server)                                                      |
| Email delivery    | Gmail API (via MCP server)                                                            |
| Deployment        | Render (Python runtime, no Docker)                                                    |
| Tests             | `pytest`, `pytest-asyncio`                                                            |

---

## 5. Folder Structure

```
Product-Review-Intelligence/
├── app/
│   ├── main.py                        # FastAPI app factory + lifespan
│   ├── api/
│   │   ├── deps.py                    # x-api-key auth (constant-time compare)
│   │   └── routes/
│   │       ├── health.py              # /health, /healthz, /readyz
│   │       └── weekly_pulse.py        # POST /run-weekly-pulse
│   ├── config/
│   │   └── settings.py                # static configs + .env-loaded secrets
│   ├── core/
│   │   ├── logging.py                 # structlog configuration
│   │   └── exceptions.py              # typed AppError hierarchy
│   ├── domain/
│   │   ├── models.py                  # Review, Theme, Pulse domain models
│   │   └── enums.py                   # Source, Sentiment enums
│   └── services/
│       ├── ingest/
│       │   ├── play_store.py          # Play Store ingestion
│       │   ├── app_store.py           # App Store RSS ingestion
│       │   ├── time_window.py         # lookback resolver
│       │   └── base.py                # common ingest contract
│       ├── preprocessing/
│       │   ├── service.py             # dedupe + normalize pipeline
│       │   ├── validators.py
│       │   └── regex_utils.py
│       ├── summarization/
│       │   ├── service.py             # per-chunk orchestration
│       │   ├── chunker.py             # token-aware batching
│       │   ├── groq_client.py         # Groq SDK wrapper + fallbacks
│       │   ├── prompts.py             # versioned prompt templates
│       │   ├── schemas.py             # structured output schemas
│       │   ├── aggregator.py          # cross-chunk merge + ranking
│       │   ├── sanitization.py
│       │   └── heuristic.py           # offline/no-LLM fallback
│       ├── pulse/
│       │   ├── service.py             # pulse assembly
│       │   ├── aggregation.py
│       │   ├── ranking.py
│       │   ├── formatting.py
│       │   ├── text_utils.py
│       │   └── schemas.py
│       ├── mcp/
│       │   ├── client.py              # MCP HTTP client (retries, backoff)
│       │   ├── delivery.py            # Google Docs + Gmail orchestration
│       │   └── schemas.py
│       └── weekly_pulse.py            # top-level orchestrator
├── tests/                             # unit + live integration tests
├── examples/                          # runnable demos for each subsystem
├── docs/
│   └── render-deploy.md
├── render.yaml                        # Render Blueprint
├── requirements.txt
├── pyproject.toml
├── Makefile
└── .env.example
```

### Important Folders

- **`app/config/`** — config-driven core: static knobs (model name, chunk size, lookback) live as frozen dataclasses; only secrets load from `.env`.
- **`app/services/ingest/`** — pluggable per-source ingestors with a shared base contract.
- **`app/services/summarization/`** — Groq orchestration, prompts, chunking, fallback models.
- **`app/services/pulse/`** — the leadership-facing report assembly.
- **`app/services/mcp/`** — the delivery boundary; talks to the separate MCP server.

---

## 6. Local Development Setup

### Clone the repository

```bash
git clone https://github.com/krunalbhandekar/Product-Review-Intelligence.git
cd Product-Review-Intelligence
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Required variables:

```env
API_KEY=your-strong-api-key
GROQ_API_KEY=gsk_xxx
GOOGLE_DOC_ID=1abc...the-target-doc-id
EMAIL_TO=team@yourcompany.com
MCP_SERVER_URL=https://your-mcp-server.example.com
PLAYSTORE_APP_ID=com.example.app
APPSTORE_APP_ID=1234567890
```

| Variable           | Purpose                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| `API_KEY`          | Server-side key required in the `x-api-key` header on every protected request. Constant-time compared. |
| `GROQ_API_KEY`     | Groq API key used for LLM summarization calls.                                                         |
| `GOOGLE_DOC_ID`    | Target Google Doc the weekly report is written to.                                                     |
| `EMAIL_TO`         | Default recipient for the email digest. Can be overridden per request.                                 |
| `MCP_SERVER_URL`   | Base URL of the deployed MCP server handling Google Docs + Gmail integration.                          |
| `PLAYSTORE_APP_ID` | Default Play Store app id (e.g. `com.example.app`). Overridable per request.                           |
| `APPSTORE_APP_ID`  | Default App Store numeric app id (e.g. `1234567890`). Overridable per request.                         |

> **Architecture note:** Operational knobs like `LOOKBACK_WEEKS`, `MAX_REVIEWS_PER_RUN`, chunk sizes, retries, and the Groq model list live in [app/config/settings.py](app/config/settings.py) — change them via a PR, not a dashboard tweak. Only secrets and deployment-specific identifiers belong in `.env`.

---

## 7. Start the Application Locally

```bash
uvicorn app.main:app --reload
```

Default endpoints:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **OpenAPI JSON:** http://127.0.0.1:8000/openapi.json
- **Health check:** http://127.0.0.1:8000/health
- **Liveness:** http://127.0.0.1:8000/healthz
- **Readiness:** http://127.0.0.1:8000/readyz

---

## 8. Running the Weekly Pulse

Trigger an end-to-end pipeline run:

```bash
curl -X POST http://127.0.0.1:8000/run-weekly-pulse \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "playstore_app_id": "com.nextbillion.groww",
    "appstore_app_id": "1404871703",
    "email_to": "example@gmail.com",
    "dry_run": false
}'
```

### Request fields

| Field              | Type           | Default                | Notes                                                              |
| ------------------ | -------------- | ---------------------- | ------------------------------------------------------------------ |
| `playstore_app_id` | string \| null | `PLAYSTORE_APP_ID` env | Skip the Android source by passing `null` _and_ unsetting the env. |
| `appstore_app_id`  | string \| null | `APPSTORE_APP_ID` env  | Same as above for iOS.                                             |
| `email_to`         | string \| null | `EMAIL_TO` env         | Per-request override for the email recipient.                      |
| `dry_run`          | boolean        | `false`                | Skips Google Doc write + email send; everything else runs.         |

> The lookback window is intentionally **not** a request field — it is a server-side operations knob (`LOOKBACK_WEEKS`).

### Sample successful response

```json
{
  "run_id": "wp_2026-05-16T03-21-44Z_a1b2",
  "status": "succeeded",
  "dry_run": false,
  "started_at": "2026-05-16T03:21:44Z",
  "finished_at": "2026-05-16T03:22:19Z",
  "window_start": "2026-02-21T00:00:00Z",
  "window_end": "2026-05-16T00:00:00Z",
  "reviews_ingested": 48,
  "chunks_summarized": 2,
  "email_sent": true,
  "details": {
    "playstore_count": 28,
    "appstore_count": 20,
    "google_doc_id": "1abc...",
    "themes": 5
  }
}
```

### HTTP status semantics

| Code  | Meaning                                                   |
| ----- | --------------------------------------------------------- |
| `200` | Pipeline succeeded end-to-end.                            |
| `204` | No reviews in the lookback window — nothing to ship.      |
| `207` | Doc published but email partially failed (or vice versa). |
| `401` | Missing or invalid `x-api-key`.                           |
| `500` | Pipeline failed after retries.                            |

### Output artifacts

1. **Google Doc** — the full structured report written to `GOOGLE_DOC_ID`.
2. **Email digest** — executive summary + link to the full Doc.
3. **Structured JSON response** — useful for chaining or scheduled jobs.

---

## 9. MCP Server Integration

Google Docs writes and Gmail sends are delegated to a dedicated **MCP (Model Context Protocol) server**, deployed separately.

- **MCP Server GitHub Repo:** `https://github.com/krunalbhandekar/Google-Docs-Gmail-MCP-Server`
- **MCP Server Deployment URL:** `https://google-docs-gmail-mcp-server-u06s.onrender.com`

### Why a separate MCP server?

- **OAuth lives in one place.** Google OAuth tokens, scopes, and refresh handling stay isolated from the core review pipeline.
- **Independent deploy cadence.** Auth / Google API changes can ship without redeploying the analytics service.
- **Reusable surface.** The same MCP server can serve other internal tools (release notes, marketing digests, etc.).
- **Smaller blast radius.** A bug in the report pipeline can never corrupt OAuth state.

### Responsibilities

| Concern                                            | Where it lives   |
| -------------------------------------------------- | ---------------- |
| Review ingestion, LLM summarization, theme ranking | **This service** |
| Google OAuth, Doc writes, Gmail sends              | **MCP server**   |

### Communication flow

```
Product-Review-Intelligence ─── HTTPS ───▶ MCP Server ─── Google APIs ─▶ Doc / Gmail
                              x-api-key                  OAuth tokens
```

The MCP client in [app/services/mcp/client.py](app/services/mcp/client.py) handles retries, exponential backoff (`MCP_RETRY_BACKOFF_SECONDS`), and timeout (`MCP_TIMEOUT`).

Required env variable:

```env
MCP_SERVER_URL=https://your-mcp-server.example.com
```

---

## 10. Deployment on Render

> **Docker is NOT required.** Render uses its native Python runtime directly from `render.yaml`.

### One-click Blueprint deploy

1. Go to the **Render Dashboard**.
2. Click **New +**.
3. Select **Blueprint**.
4. Connect this GitHub repository.
5. Render reads [render.yaml](render.yaml) and provisions the service automatically.
6. Add the required secrets in the Render dashboard (see below).

### Secrets to set in the Render dashboard

These are declared as `sync: false` in `render.yaml`, so they must be entered manually and are never committed:

```env
API_KEY
GROQ_API_KEY
GOOGLE_DOC_ID
EMAIL_TO
PLAYSTORE_APP_ID
APPSTORE_APP_ID
MCP_SERVER_URL
```

### Runtime configuration

- **Python version:** `3.11.9` (pinned via `PYTHON_VERSION`).
- **Auto-deploy:** enabled on pushes to `main`.
- **Health check path:** `/healthz`.
- **Plan:** runs cleanly on Render's **Free tier**.
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

### Manual Web Service deploy (alternative)

1. **New + → Web Service**, connect the repo.
2. Runtime: **Python**.
3. Build command: `pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add the same secrets from above.

See [docs/render-deploy.md](docs/render-deploy.md) for the long-form guide.

---

## 11. Automated Weekly Scheduler (GitHub Actions)

The project supports **fully automated weekly execution** using GitHub Actions — no external cron, no separate worker, no babysitting.

- GitHub Actions triggers the deployed Render API endpoint automatically **every Monday at 6:00 AM IST**.
- The scheduler **only triggers the API**; the actual review ingestion, LLM summarization, Google Docs generation, and email delivery all happen inside the backend service on Render.
- This keeps the scheduler dumb and the pipeline smart — easy to swap orchestrators (Render Cron, AWS EventBridge, Cloud Scheduler) without touching pipeline code.

### Workflow file location

```
.github/workflows/daily-weekly-pulse.yml
```

### Cron explanation

```yaml
cron: "30 0 * * 1"
```

- GitHub Actions uses **UTC**, not local time.
- `00:30 UTC` = **06:00 AM IST** (UTC+5:30).
- The trailing `1` = **Monday**, so the job runs once a week.

### Configure GitHub Secrets

Go to:

```
GitHub Repository → Settings → Secrets and variables → Actions
```

Add the following repository secrets:

| Secret            | Description                    |
| ----------------- | ------------------------------ |
| `RENDER_BASE_URL` | Public Render deployment URL   |
| `API_KEY`         | Backend API authentication key |

Example values:

```text
RENDER_BASE_URL=https://product-review-intelligence.onrender.com
API_KEY=your-prod-key
```

The workflow fails fast with a clear error if either secret is missing.

### Run Scheduler Manually

The workflow also supports an on-demand run via `workflow_dispatch`:

1. Go to: **GitHub Repo → Actions → Weekly Product Pulse Trigger**
2. Click: **Run workflow**
3. Select:
   - **Branch** (usually `main`)
   - **`dry_run` mode** (`true` or `false`)
4. Click: **Run workflow**

Behavior:

- **`dry_run=false`** → Runs the full pipeline, generates the Google Docs report, and sends the email digest.
- **`dry_run=true`** → Runs ingestion, summarization, and aggregation **without** delivery — useful for verifying the pipeline end-to-end without spamming the inbox.

### Monitoring & Logging

| Surface             | What you see                                                                            |
| ------------------- | --------------------------------------------------------------------------------------- |
| GitHub Actions logs | Each attempt, HTTP status, and the full JSON response from the Render API.              |
| Render service logs | Structured `structlog` JSON logs for ingestion, summarization, MCP calls, and delivery. |
| API response body   | `run_id`, counts, themes, and per-stage details for every triggered run.                |
| Retry trail         | Every retry is printed with the attempt number, HTTP code, and backoff duration.        |

### Reliability features built into the workflow

- **Retry logic** — up to 3 attempts with exponential backoff (10s → 20s → 40s) on transient failures (`5xx`, `408`, `429`, network errors).
- **Concurrency protection** — `concurrency.group: weekly-pulse-trigger` with `cancel-in-progress: false` ensures two runs never overlap.
- **Timeout handling** — 15-minute job timeout plus a 10-minute per-request `--max-time` on `curl`.
- **Partial-success handling** — treats `200` (succeeded), `204` (no data in window), and `207` (partial delivery) all as successful triggers, since they are pipeline-level outcomes — not transport failures.
- **Fail-fast on auth errors** — non-retriable `4xx` responses (except `408`/`429`) exit immediately with the response body surfaced in the logs.

---

## 12. Example Output

### Executive Summary (sample)

> Over the last 12 weeks, users overwhelmingly praised the redesigned portfolio screen and the speed of stock order execution. However, a recurring and intensifying theme this week is **login friction post-update v4.8.2** — users report being logged out repeatedly and OTP delivery delays of 2–5 minutes. A secondary cluster of complaints centers on **mutual fund SIP failures on auto-debit days**, with users explicitly calling out lack of in-app communication when a SIP debit fails. Sentiment skewed 62% negative this week, up from 41% last week, driven almost entirely by the login issue. Engagement-positive themes (UI polish, charting depth, IPO discovery) remained stable.

### Top Themes (sample)

| #   | Theme                                | Frequency   | Sentiment | Representative Quote                                                         |
| --- | ------------------------------------ | ----------- | --------- | ---------------------------------------------------------------------------- |
| 1   | Repeated forced logouts after update | 23 mentions | Negative  | "After the latest update I have to log in 5 times a day — this is unusable." |
| 2   | OTP delivery delays                  | 14 mentions | Negative  | "OTP takes 3 minutes to arrive and then says expired."                       |
| 3   | SIP auto-debit silent failure        | 9 mentions  | Negative  | "My SIP failed and nobody told me. Found out 4 days later."                  |
| 4   | New portfolio screen                 | 12 mentions | Positive  | "The new portfolio view is the cleanest I've seen on any broker app."        |
| 5   | IPO discovery flow                   | 7 mentions  | Positive  | "Love how IPOs surface right on the home screen now."                        |

### Action Items (sample)

1. **Investigate session token rotation in v4.8.2** — correlate review timestamps with auth-service logs; prioritize a hotfix if root cause confirms.
2. **Add a SIP-failure in-app notification** — silent failures are the highest-emotion complaint cluster; even a basic toast/email would defuse it.
3. **Audit OTP provider latency P95** — current SLA appears breached based on user-reported delivery times.

---

## 13. API Endpoints

| Method | Endpoint            | Auth        | Description                                          |
| ------ | ------------------- | ----------- | ---------------------------------------------------- |
| `POST` | `/run-weekly-pulse` | `x-api-key` | Runs the full ingest → summarize → deliver pipeline. |
| `GET`  | `/health`           | none        | Application health + identity.                       |
| `GET`  | `/healthz`          | none        | Liveness probe (used by Render).                     |
| `GET`  | `/readyz`           | none        | Readiness probe.                                     |
| `GET`  | `/docs`             | none        | Swagger UI.                                          |
| `GET`  | `/openapi.json`     | none        | OpenAPI schema.                                      |

---

## 14. Scalability & Future Improvements

- **Multi-app dashboard** — manage many apps from one UI; per-app schedules and recipients.
- **Slack integration** — post the digest to a `#product-pulse` channel via the same MCP server pattern.
- **Sentiment trend tracking** — week-over-week deltas, anomaly detection on theme spikes.
- **Vector DB + RAG** — embed all historical reviews; let PMs ask "did anyone complain about X in Q1?".
- **Historical analytics** — persist runs in Postgres for longitudinal cohort/theme analysis.
- **Multi-tenant architecture** — per-tenant API keys, isolated Doc/email destinations.
- **Scheduled jobs** — built-in cron scheduler so no external orchestrator is needed.
- **Web dashboard** — drill into the underlying reviews behind each theme.
- **More sources** — Twitter/X, Reddit, support tickets, in-app feedback.

---

## 15. Challenges Solved

- **LLM token limits** — token-aware chunker batches reviews under `CHUNK_TARGET_TOKENS` (3,500 by default) so each call stays well under context.
- **Chunking strategy** — chunks are review-aligned, never mid-text, so themes stay attributable to specific quotes.
- **Deduplication** — hash-based dedupe in preprocessing prevents the same copy-paste review from biasing theme frequency.
- **Retry handling** — bounded retries with exponential backoff on both Groq and MCP calls; structured logs for every retry.
- **LLM fallback** — primary model `llama-3.1-8b-instant`; on 429 streaks, automatically tries `llama3-8b-8192` then `gemma2-9b-it`.
- **Structured outputs** — Pydantic-validated JSON schemas on every LLM response; malformed outputs are sanitized or rejected, never silently dropped.
- **Review preprocessing** — strips repeated emoji noise, boilerplate auto-translations, and trims to `MAX_REVIEW_BODY_CHARS` so the LLM sees signal, not noise.
- **Async concurrency** — Play Store and App Store ingestion run in parallel; LLM calls have bounded concurrency (`REQUEST_CONCURRENCY`) to respect rate limits.
- **Deployment without Docker on Render's free tier** — pure Python runtime; cold-start friendly; `render.yaml` Blueprint pins Python version and sets secrets as `sync: false`.
- **Config-driven architecture** — static knobs live in code (diffable, code-reviewed); only secrets live in env (rotatable, not committed).

---

## 16. Author

Built by **Krunal Bhandekar**.

- **LinkedIn:** `https://www.linkedin.com/in/krunal-bhandekar/`
- **GitHub:** `https://github.com/krunalbhandekar`

---

_Product Review Intelligence — turning a wall of reviews into a one-page pulse, every week._
