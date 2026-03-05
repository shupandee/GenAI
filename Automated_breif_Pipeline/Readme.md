# Scrollhouse — Content Brief to Script Pipeline

> **AI Automation Intern Take-Home · Problem: Content Brief to Script Pipeline**

An end-to-end AI pipeline that transforms raw Google Form client submissions into structured internal briefs + first-draft scripts — replacing 25 minutes of soul-destroying manual work with a 90-second automated pipeline.

---

## Why This Problem?

**The math:**

| | Manual | Pipeline |
|---|---|---|
| Time per brief | 25 min | ~90 sec |
| Monthly volume | 50 briefs | 50 briefs |
| Monthly time cost | **20.8 hrs** | **1.25 hrs** |
| Net saving | — | **19.5 hrs/mo (~2.4 work days)** |

The Content Approval Loop is arguably more painful, but it's an *inter-company* coordination problem (you can't force clients to respond faster). The brief pipeline is an *internal* process — fully within Scrollhouse's control, entirely automatable, and where quality problems compound downstream (a bad brief = bad script = bad revision loop = missed deadline).

**Why not Client Onboarding?** It's only 6–10 clients/month (~7.5 hrs saved). The brief pipeline touches every piece of content the company makes (40–60/month) and the person doing it finds it "soul-destroying" — which means you're burning out your best people on the most automatable work.

---

## What It Does

### Full Pipeline (3 AI steps, fully automated):

```
Google Form Submission
        ↓
[Webhook / API trigger]
        ↓
Step 1: Transform Brief  ← Gemini AI
        Reformats raw form answers into structured
        internal brief with enriched fields:
        hooks, talking points, deliverable specs
        ↓
Step 2: Quality Check    ← Gemini AI
        Scores brief completeness (0–100)
        Flags missing info before it reaches writer
        Blocks poor briefs from wasting writer time
        ↓
Step 3: Script Generation ← Gemini AI (if QC ≥ 75)
        Scene-by-scene first-draft script
        Ready for scriptwriter to polish
        ↓
[Notify Scriptwriter via Slack/Email]
[Create Notion page + Airtable row]
```

---

## Project Structure

```
scrollhouse/
├── app.py          # Flask backend — all API routes + pipeline logic
├── index.html      # Single-file frontend (HTML/CSS/JS)
├── README.md       # This file
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd scrollhouse
pip install flask
```

### 2. (Optional) Set Gemini API Key

Without a key, the app runs in **demo mode** with realistic mock responses — perfect for the walkthrough.

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a free key at: https://aistudio.google.com/app/apikey

### 3. Run

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Using the App

### Dashboard
- See live stats: briefs processed, scripts generated, time saved
- Visual pipeline architecture diagram
- "The Math" breakdown comparing before/after
- **"Load Demo Data"** button — seeds 3 realistic client briefs through the full pipeline instantly

### Submit Brief
- Simulates a Google Form submission
- **"Fill Demo"** button auto-fills a realistic form
- Watch the 4-step pipeline run in real-time with animated progress
- Results appear in 3 tabs:
  - **Internal Brief** — structured brief with AI-suggested hooks
  - **QC Report** — score, completeness check, issues + suggestions
  - **Script** — full scene-by-scene shooting script

### All Briefs
- View all processed briefs in a sortable list
- Click any brief to open the full detail modal
- Regenerate scripts from existing briefs

### Pipeline Logs
- Audit trail of every pipeline run
- Step-by-step status per run
- Outcome tracking (script generated vs needs revision)

---

## API Reference

All endpoints return JSON.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/submit-brief` | Run full pipeline on form data |
| `GET` | `/api/briefs` | List all processed briefs |
| `GET` | `/api/briefs/<id>` | Get full brief details |
| `GET` | `/api/scripts/<id>` | Get generated script |
| `POST` | `/api/regenerate-script/<id>` | Regenerate script for a brief |
| `POST` | `/api/quality-check/<id>` | Re-run QC on existing brief |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/pipeline-log` | Last 20 pipeline runs |
| `POST` | `/api/seed-demo` | Seed 3 demo briefs |

### Example: Submit Brief

```bash
curl -X POST http://localhost:5000/api/submit-brief \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "FitCore Athletics",
    "industry": "Sports & Fitness",
    "campaign_goal": "Launch summer activewear collection",
    "target_audience": "Millennials 24-35, gym-goers",
    "platforms": "Instagram Reels, TikTok",
    "tone": "Energetic & Aspirational",
    "key_message": "Performance meets style",
    "products": "HydraTech activewear — leggings, sports bras, shorts",
    "cta": "Shop now — link in bio",
    "deadline": "12 June 2025"
  }'
```

**Response:**
```json
{
  "success": true,
  "brief_id": "BRF-A1B2C3D4",
  "data": {
    "internal_brief": { ... },
    "qc": { "score": 88, "passed": true, ... },
    "script": { "scenes": [ ... ], ... }
  }
}
```

---

## Architecture

### Backend (`app.py`)
- **Flask** — lightweight Python web framework
- **Gemini API** — Google's generative AI (via direct HTTP, no SDK dependency)
- **In-memory store** — briefs and scripts stored in Python dicts (replace with Notion/Airtable in production)
- Three AI pipeline functions:
  - `process_brief_to_internal()` — Form → Internal Brief
  - `quality_check_brief()` — Brief → QC Score
  - `generate_script()` — Brief → Scene Script

### Frontend (`index.html`)
- Pure HTML/CSS/JS — zero build step, zero dependencies
- Google Fonts (Syne + DM Mono + DM Sans)
- Fetch API for all backend calls
- Animated pipeline progress visualization

### AI Prompting Strategy
Each Gemini call uses a **system prompt** that defines the role (e.g., "senior content strategist") and mandates structured JSON output. This makes responses deterministic and parseable without hallucinated formatting.

The three prompts are tuned separately:
1. **Transform prompt** — focuses on completeness and enrichment, not invention
2. **QC prompt** — evaluates against a fixed checklist, returns numeric score
3. **Script prompt** — platform-native creative writing with strict scene structure

### Demo Mode
When `GEMINI_API_KEY` is not set, `_mock_gemini_response()` returns realistic hardcoded responses. This means the full app is demonstrable without an API key.

---

## Production Additions (What I'd Build Next)

### Immediate (Week 1–2):
1. **Real Google Form webhook** — Form submission triggers pipeline automatically via Apps Script or Zapier/Make webhook. Zero manual action required.
2. **Notion output** — Create a Notion page per brief with the structured content. Scriptwriters work directly in Notion.
3. **Slack notification** — DM the assigned scriptwriter with brief summary + Notion link when pipeline completes.

### Phase 2 (Week 3–4):
4. **Content Approval Loop tracker** — Since I've already built brief tracking, adding approval status (Waiting → Approved → Rejected → Revision) is a natural extension. This solves the next biggest problem on the list.
5. **Airtable sync** — Write brief status, QC score, and deadlines to Airtable so ops has a single source of truth.
6. **Brief revision feedback loop** — If QC fails, automatically email client asking for the missing information.

### Longer Term:
7. **Performance-based brief improvement** — Feed post-campaign metrics back in to improve hook suggestions for similar clients.
8. **Client portal** — Simple form where clients can submit briefs directly and see their pipeline status.

---

## What the Pipeline Replaces

Previously, a team member would:
1. Open Google Form responses sheet
2. Read through raw client answers
3. Reformat into internal brief structure in Notion (manual copy-paste + rewrite)
4. Add talking points, deliverable specs, tone guidance from memory
5. Do a mental quality check (often inconsistent)
6. Notify the scriptwriter manually

This pipeline **eliminates steps 2–6 entirely**. The team member now just:
1. Receives a Slack notification that a brief is ready
2. Does a 2-minute human review of the AI-generated brief
3. Approves → scriptwriter starts writing

**Time per brief: 25 min → 2 min (human review only)**

---

## Limitations & Honest Notes

- **In-memory storage** — data resets when server restarts. Production needs a real database.
- **No real Notion/Airtable integration** — mocked in this demo. The API calls are straightforward to add.
- **No real Slack notifications** — the notify step is mocked. Adding real Slack requires a Slack app token.
- **Gemini rate limits** — in production, add retry logic with exponential backoff.
- **Script quality** — AI first drafts are good starting points, not finished scripts. Human scriptwriters still add the craft.
- **Edge cases** — extremely short or poorly filled forms will get lower QC scores and won't generate scripts, which is correct behaviour.

---

## Stack Summary

| Layer | Tech | Why |
|-------|------|-----|
| Backend | Python + Flask | Simple, fast, easy to extend. No overhead. |
| AI | Gemini 2.0 Flash | Fast, cost-effective for high-volume brief processing. Free tier available. |
| Frontend | Vanilla HTML/CSS/JS | Zero build step. Easy to understand, easy to hand off. |
| Storage (demo) | Python dicts | Keeps demo self-contained. Swap for Notion API or Postgres in prod. |

---

## Time Estimate

Built this in approximately **8–9 hours**:
- 1h — problem analysis + architecture planning
- 2h — backend pipeline logic + Gemini prompts
- 4h — frontend design + all UI components
- 1h — testing + edge cases
- 1h — README + demo data

---

*Built for Scrollhouse AI Automation Intern application.*