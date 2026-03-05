"""
Scrollhouse - Content Brief to Script Pipeline
Backend: Flask + Gemini AI
"""

import json
import uuid
import time
import os
import re
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from typing import Optional

# ─────────────────────────────────────────────
# LOAD .env MANUALLY (no python-dotenv needed)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
env_file = BASE_DIR / ".env"

if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and val not in ("your_gemini_api_key_here", ""):
                    os.environ.setdefault(key, val)

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# In-memory database (replace with real DB / Notion / Airtable in production)
briefs_db: dict = {}
scripts_db: dict = {}
pipeline_log: list = []

# ─────────────────────────────────────────────
# GEMINI HELPER
# ─────────────────────────────────────────────
def call_gemini(prompt: str, system: str = "") -> str:
    """
    Call Gemini API. Falls back to a structured mock if no API key is set.
    In production, set GEMINI_API_KEY env variable.
    """
    if not GEMINI_API_KEY:
        return _mock_gemini_response(prompt)

    try:
        import urllib.request
        import urllib.error

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System context]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I'll follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        body = json.dumps({
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return _mock_gemini_response(prompt, error=str(e))


def _mock_gemini_response(prompt: str, error: str = "") -> str:
    """Deterministic mock that returns realistic outputs for demo purposes.
    Order matters: QC check BEFORE script, because QC prompt contains the brief JSON which has 'script' in it.
    """
    p = prompt.lower()

    # 1. QUALITY CHECK — must be first (QC prompt contains brief JSON with word "script")
    if "quality check this" in p or "score it and identify" in p:
        return json.dumps({
            "score": 88,
            "passed": True,
            "issues": [],
            "suggestions": [
                "Consider adding a secondary hook variant for A/B testing",
                "The 60-second TikTok brief could use more specific emotional arc guidance"
            ],
            "completeness": {
                "target_audience": True,
                "platform": True,
                "tone": True,
                "cta": True,
                "deadline": True,
                "deliverables": True
            }
        }, indent=2)

    # 2. TRANSFORM BRIEF
    elif "client submission" in p or "brand name:" in p or p.strip().startswith("transform this raw"):
        return json.dumps({
            "brief_title": "Fitness Brand Product Launch — Summer Collection",
            "client_name": "FitCore Athletics",
            "platform": "Instagram Reels + TikTok",
            "target_audience": "Health-conscious millennials (24–35), active lifestyle, mid-to-high income",
            "core_message": "Introducing FitCore's summer activewear line — engineered for performance, designed for style",
            "tone_of_voice": "Energetic, aspirational, authentic — real athletes, real results",
            "deliverables": [
                "3x 30-second Reels (product showcase)",
                "1x 60-second hero TikTok (brand story)",
                "5x 15-second story clips"
            ],
            "key_talking_points": [
                "Moisture-wicking HydraTech fabric",
                "Available in 12 colorways",
                "Limited summer drop — 15 June launch",
                "Priced at 3499 INR / $42"
            ],
            "hooks_suggested": [
                "You've never sweated in something this good-looking.",
                "Summer just called. It wants its activewear back.",
                "Train harder. Look better. Drop 15th June."
            ],
            "cta": "Shop the drop — link in bio",
            "reference_links": ["@fitcore_official previous campaign Q1", "Nike summer 2023 aesthetic"],
            "deadline": "12 June 2025",
            "notes": "Client prefers natural lighting, diverse talent. No heavily edited looks.",
            "ai_quality_score": 92,
            "ai_flags": []
        }, indent=2)

    # 3. SCRIPT GENERATION
    elif "script" in p or "scene" in p or "write a" in p:
        return json.dumps({
            "script_title": "FitCore Summer Drop — Hero Reel",
            "platform": "Instagram Reels",
            "duration": "30 seconds",
            "scenes": [
                {
                    "scene": 1,
                    "timestamp": "0:00-0:03",
                    "visual": "Close-up: hands lacing up bright trainers. Morning light. Fast cut.",
                    "voiceover": "",
                    "on_screen_text": "5:30 AM.",
                    "sound": "Ambient gym sounds fade in"
                },
                {
                    "scene": 2,
                    "timestamp": "0:03-0:10",
                    "visual": "Athlete running — slow motion. Sweat. Joy. FitCore gear prominent.",
                    "voiceover": "You've never sweated in something this good-looking.",
                    "on_screen_text": "",
                    "sound": "Beat drops"
                },
                {
                    "scene": 3,
                    "timestamp": "0:10-0:20",
                    "visual": "Quick cuts: 3 different athletes, 3 different colorways. Studio + outdoor.",
                    "voiceover": "HydraTech fabric. 12 colors. Built for every kind of summer.",
                    "on_screen_text": "HydraTech",
                    "sound": "Upbeat music continues"
                },
                {
                    "scene": 4,
                    "timestamp": "0:20-0:27",
                    "visual": "Product flat lay — clean white background. All colorways fanned out.",
                    "voiceover": "Dropping June 15th.",
                    "on_screen_text": "JUNE 15 - LIMITED DROP",
                    "sound": "Music softens"
                },
                {
                    "scene": 5,
                    "timestamp": "0:27-0:30",
                    "visual": "Logo lock-up. CTA slide.",
                    "voiceover": "",
                    "on_screen_text": "Shop now — link in bio",
                    "sound": "Music out"
                }
            ],
            "scriptwriter_notes": "Keep edits snappy — 8 cuts max. Avoid heavy colour grading, client prefers natural tones. Use diverse talent per brief.",
            "word_count": 38,
            "estimated_read_time": "28 seconds"
        }, indent=2)

    else:
        return json.dumps({"result": "Processed successfully", "mock": True, "note": "Set GEMINI_API_KEY for live AI responses"})


def call_gemini(prompt: str, system: str = "") -> str:
    """
    Call Gemini API. Falls back to a structured mock if no API key is set.
    In production, set GEMINI_API_KEY env variable.
    """
    if not GEMINI_API_KEY:
        return _mock_gemini_response(prompt)

    try:
        import urllib.request
        import urllib.error

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System context]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I'll follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        body = json.dumps({
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return _mock_gemini_response(prompt, error=str(e))


# ─────────────────────────────────────────────
# SYSTEM PROMPTS FOR GEMINI
# ─────────────────────────────────────────────

INTERNAL_BRIEF_SYSTEM = """You are a senior content strategist at Scrollhouse, a short-form video agency.
Your job is to take raw client brief inputs and transform them into a clean, structured internal brief
that scriptwriters can use immediately without any back-and-forth.

Always return a valid JSON object with these fields:
brief_title, client_name, platform, target_audience, core_message, tone_of_voice,
deliverables (array), key_talking_points (array), hooks_suggested (array),
cta, reference_links (array), deadline, notes, ai_quality_score (0-100), ai_flags (array of issues).

Be specific. If information is missing, flag it in ai_flags. Never make up facts the client did not provide.
Return ONLY valid JSON, no markdown, no preamble."""

QUALITY_CHECK_SYSTEM = """You are a quality control specialist at Scrollhouse.
Evaluate an internal content brief for completeness, clarity and actionability.

Return a JSON object with:
score (0-100), passed (true if score >= 75), issues (array of blockers),
suggestions (array of improvements), completeness (object with boolean fields for each key section).

Return ONLY valid JSON."""

SCRIPT_SYSTEM = """You are a professional short-form video scriptwriter at Scrollhouse.
You write punchy, platform-native scripts for Instagram Reels and TikTok.

Given an internal brief, produce a complete scene-by-scene script in JSON format with these fields:
script_title, platform, duration, scenes (array of: scene, timestamp, visual, voiceover, on_screen_text, sound),
scriptwriter_notes, word_count, estimated_read_time.

Scripts must be hook-first, platform native, concise and punchy.
Return ONLY valid JSON, no markdown, no preamble."""


def process_brief_to_internal(raw_form_data: dict) -> dict:
    """Step 1: Transform raw Google Form data into structured internal brief."""
    
    prompt = f"""Transform this raw client brief form submission into our structured internal brief format:

CLIENT SUBMISSION:
Brand Name: {raw_form_data.get('brand_name', 'Not provided')}
Industry: {raw_form_data.get('industry', 'Not provided')}
Campaign Goal: {raw_form_data.get('campaign_goal', 'Not provided')}
Target Audience: {raw_form_data.get('target_audience', 'Not provided')}
Platforms: {raw_form_data.get('platforms', 'Not provided')}
Tone/Vibe: {raw_form_data.get('tone', 'Not provided')}
Key Message: {raw_form_data.get('key_message', 'Not provided')}
Products/Services: {raw_form_data.get('products', 'Not provided')}
Call to Action: {raw_form_data.get('cta', 'Not provided')}
Reference Examples: {raw_form_data.get('references', 'None provided')}
Deadline: {raw_form_data.get('deadline', 'Not specified')}
Budget Tier: {raw_form_data.get('budget_tier', 'Not specified')}
Additional Notes: {raw_form_data.get('additional_notes', 'None')}

Reformat this into our internal brief structure. Flag any missing critical information."""

    response = call_gemini(prompt, system=INTERNAL_BRIEF_SYSTEM)
    
    try:
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response", "raw": response}


def quality_check_brief(internal_brief: dict) -> dict:
    """Step 2: Run AI quality check on the internal brief."""
    
    prompt = f"""Quality check this internal brief:

{json.dumps(internal_brief, indent=2)}

Score it and identify any issues that would block scriptwriting."""

    response = call_gemini(prompt, system=QUALITY_CHECK_SYSTEM)
    
    try:
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"score": 0, "passed": False, "issues": ["QC parse error"], "suggestions": [], "completeness": {}}


def generate_script(internal_brief: dict, script_type: str = "hero_reel") -> dict:
    """Step 3: Generate script from internal brief."""
    
    prompt = f"""Write a {script_type.replace('_', ' ')} script based on this internal brief:

{json.dumps(internal_brief, indent=2)}

Create a complete, production-ready script."""

    response = call_gemini(prompt, system=SCRIPT_SYSTEM)
    
    try:
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Failed to parse script", "raw": response}


def run_full_pipeline(raw_form_data: dict, brief_id: str) -> dict:
    """Run the complete brief → internal brief → QC → script pipeline."""
    
    log_entry = {
        "brief_id": brief_id,
        "started_at": datetime.datetime.now().isoformat(),
        "steps": []
    }
    
    # Step 1: Transform
    log_entry["steps"].append({"step": "transform", "status": "running", "ts": time.time()})
    internal_brief = process_brief_to_internal(raw_form_data)
    internal_brief["brief_id"] = brief_id
    internal_brief["created_at"] = datetime.datetime.now().isoformat()
    internal_brief["source_form"] = raw_form_data
    log_entry["steps"][-1]["status"] = "done"
    
    # Step 2: Quality Check
    log_entry["steps"].append({"step": "quality_check", "status": "running", "ts": time.time()})
    qc_result = quality_check_brief(internal_brief)
    internal_brief["qc"] = qc_result
    log_entry["steps"][-1]["status"] = "done"
    log_entry["steps"][-1]["score"] = qc_result.get("score", 0)
    
    # Step 3: Generate Script (only if QC passed)
    script = None
    if qc_result.get("passed", False):
        log_entry["steps"].append({"step": "script_generation", "status": "running", "ts": time.time()})
        script = generate_script(internal_brief)
        script["brief_id"] = brief_id
        script["generated_at"] = datetime.datetime.now().isoformat()
        scripts_db[brief_id] = script
        log_entry["steps"][-1]["status"] = "done"
    else:
        log_entry["steps"].append({"step": "script_generation", "status": "skipped", "reason": "QC failed"})
    
    # Store
    briefs_db[brief_id] = internal_brief
    log_entry["completed_at"] = datetime.datetime.now().isoformat()
    log_entry["outcome"] = "script_generated" if script else "needs_revision"
    pipeline_log.append(log_entry)
    
    return {
        "brief_id": brief_id,
        "internal_brief": internal_brief,
        "qc": qc_result,
        "script": script,
        "log": log_entry
    }


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    html_path = BASE_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/api/submit-brief", methods=["POST"])
def submit_brief():
    """Receive a raw form submission and run the full pipeline."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    brief_id = f"BRF-{str(uuid.uuid4())[:8].upper()}"
    
    try:
        result = run_full_pipeline(data, brief_id)
        return jsonify({
            "success": True,
            "brief_id": brief_id,
            "data": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/briefs", methods=["GET"])
def list_briefs():
    """List all processed briefs."""
    briefs_list = []
    for bid, brief in briefs_db.items():
        briefs_list.append({
            "brief_id": bid,
            "client_name": brief.get("client_name", "Unknown"),
            "brief_title": brief.get("brief_title", "Untitled"),
            "platform": brief.get("platform", "—"),
            "deadline": brief.get("deadline", "—"),
            "qc_score": brief.get("qc", {}).get("score", 0),
            "qc_passed": brief.get("qc", {}).get("passed", False),
            "has_script": bid in scripts_db,
            "created_at": brief.get("created_at", "")
        })
    briefs_list.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify({"briefs": briefs_list, "total": len(briefs_list)})


@app.route("/api/briefs/<brief_id>", methods=["GET"])
def get_brief(brief_id):
    """Get full brief details."""
    brief = briefs_db.get(brief_id)
    if not brief:
        return jsonify({"error": "Brief not found"}), 404
    return jsonify(brief)


@app.route("/api/scripts/<brief_id>", methods=["GET"])
def get_script(brief_id):
    """Get generated script for a brief."""
    script = scripts_db.get(brief_id)
    if not script:
        return jsonify({"error": "Script not found for this brief"}), 404
    return jsonify(script)


@app.route("/api/regenerate-script/<brief_id>", methods=["POST"])
def regenerate_script(brief_id):
    """Regenerate script for an existing brief."""
    brief = briefs_db.get(brief_id)
    if not brief:
        return jsonify({"error": "Brief not found"}), 404
    
    data = request.get_json() or {}
    script_type = data.get("script_type", "hero_reel")
    
    script = generate_script(brief, script_type)
    script["brief_id"] = brief_id
    script["generated_at"] = datetime.datetime.now().isoformat()
    scripts_db[brief_id] = script
    
    return jsonify({"success": True, "script": script})


@app.route("/api/quality-check/<brief_id>", methods=["POST"])
def rerun_qc(brief_id):
    """Re-run quality check on a brief."""
    brief = briefs_db.get(brief_id)
    if not brief:
        return jsonify({"error": "Brief not found"}), 404
    
    qc = quality_check_brief(brief)
    briefs_db[brief_id]["qc"] = qc
    return jsonify({"success": True, "qc": qc})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Dashboard statistics."""
    total_briefs = len(briefs_db)
    total_scripts = len(scripts_db)
    
    qc_scores = [b.get("qc", {}).get("score", 0) for b in briefs_db.values()]
    avg_score = round(sum(qc_scores) / len(qc_scores), 1) if qc_scores else 0
    
    passed = sum(1 for b in briefs_db.values() if b.get("qc", {}).get("passed", False))
    
    # Time saved estimate: 25 min avg per brief manually, pipeline does it in ~2 min
    time_saved_min = total_briefs * 23
    
    return jsonify({
        "total_briefs": total_briefs,
        "total_scripts": total_scripts,
        "avg_qc_score": avg_score,
        "qc_pass_rate": round((passed / total_briefs * 100), 1) if total_briefs else 0,
        "time_saved_minutes": time_saved_min,
        "time_saved_hours": round(time_saved_min / 60, 1),
        "pipeline_log_count": len(pipeline_log),
        "api_key_set": bool(GEMINI_API_KEY)
    })


@app.route("/api/pipeline-log", methods=["GET"])
def get_pipeline_log():
    return jsonify({"log": pipeline_log[-20:]})  # Last 20 entries


@app.route("/api/seed-demo", methods=["POST"])
def seed_demo():
    """Seed demo data for showcasing the system."""
    demo_briefs = [
        {
            "brand_name": "FitCore Athletics",
            "industry": "Sports & Fitness",
            "campaign_goal": "Launch summer activewear collection and drive online sales",
            "target_audience": "Health-conscious millennials aged 24-35",
            "platforms": "Instagram Reels, TikTok",
            "tone": "Energetic, aspirational, authentic",
            "key_message": "Performance meets style this summer",
            "products": "HydraTech activewear — leggings, sports bras, shorts",
            "cta": "Shop now — link in bio",
            "references": "Nike summer campaigns, Gymshark style",
            "deadline": "12 June 2025",
            "budget_tier": "Mid",
            "additional_notes": "Prefer diverse talent, natural lighting"
        },
        {
            "brand_name": "Brewlab Coffee",
            "industry": "Food & Beverage",
            "campaign_goal": "Brand awareness for new cold brew range",
            "target_audience": "Urban professionals, coffee enthusiasts 22-40",
            "platforms": "Instagram Reels",
            "tone": "Cool, minimal, slightly witty",
            "key_message": "Cold brew that doesn't compromise",
            "products": "Original Cold Brew, Vanilla Nitro, Oat Milk Cold Brew",
            "cta": "Find us at your nearest café",
            "references": "Blue Bottle aesthetic, minimal product-led content",
            "deadline": "20 June 2025",
            "budget_tier": "Low",
            "additional_notes": "No stock footage please, all original"
        },
        {
            "brand_name": "Lumena Skincare",
            "industry": "Beauty & Wellness",
            "campaign_goal": "Product launch for new SPF moisturiser",
            "target_audience": "Women 28-45, skincare-conscious, willing to invest in quality",
            "platforms": "TikTok, Instagram Reels",
            "tone": "Educational, trustworthy, approachable",
            "key_message": "Science-backed SPF that your skin will actually love",
            "products": "Lumena Daily Glow SPF 50 Moisturiser",
            "cta": "Get yours at lumena.com",
            "references": "The Ordinary content style, Paula's Choice",
            "deadline": "5 July 2025",
            "budget_tier": "High",
            "additional_notes": "Feature a dermatologist if possible"
        }
    ]
    
    created = []
    for brief_data in demo_briefs:
        brief_id = f"BRF-{str(uuid.uuid4())[:8].upper()}"
        result = run_full_pipeline(brief_data, brief_id)
        created.append(brief_id)
    
    return jsonify({"success": True, "created": created, "count": len(created)})


if __name__ == "__main__":
    print("🚀 Scrollhouse Brief Pipeline starting...")
    print(f"📡 Gemini API key: {'SET ✓' if GEMINI_API_KEY else 'NOT SET — using demo mode'}")
    print("🌐 Open http://localhost:5000")
    app.run(debug=True, port=5000)