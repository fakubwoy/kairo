import os
import json
import uuid
import requests
import redis as redis_lib
import difflib
import re
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import pdfplumber
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'kairo-secret-key-2026')

# On Railway (HTTPS) the session cookie must be Secure so it is sent with
# same-origin fetches. On local HTTP dev, Secure must be False.
_is_prod = bool(os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('DATABASE_URL','').startswith('postgresql'))
app.config['SESSION_COOKIE_SECURE']   = _is_prod
# SameSite=None is required so the Chrome extension (a cross-origin caller)
# can send credentialed requests (session cookies) to the API.
# SameSite=None is only safe over HTTPS — Railway enforces HTTPS in prod,
# and _is_prod already gates SESSION_COOKIE_SECURE so local dev is unaffected.
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Fix Railway's postgres:// URL (SQLAlchemy requires postgresql://)
database_url = os.environ.get('DATABASE_URL', 'sqlite:///kairo.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ── CORS — explicit allowlist only, credentials enabled ──────────────────────
# We enumerate every allowed origin rather than using a wildcard so that
# supports_credentials=True cannot be abused by arbitrary third-party sites.
#
# Origins:
#   1. The Kairo web app itself (Railway production URL)
#   2. Local dev server
#   3. The Chrome extension — its origin is chrome-extension://<extension-id>
#      Set EXTENSION_ORIGIN in Railway Variables after loading the unpacked
#      extension (copy the 32-char ID shown in chrome://extensions).
#      When you publish to the Chrome Web Store the ID changes — update the var.
_ALLOWED_ORIGINS = [
    'https://kairo.up.railway.app',
    'http://localhost:5000',
]
_ext_origin = os.environ.get('EXTENSION_ORIGIN', '').strip()
if _ext_origin:
    _ALLOWED_ORIGINS.append(_ext_origin)

CORS(
    app,
    origins=_ALLOWED_ORIGINS,
    supports_credentials=True,          # allows session cookies cross-origin
    allow_headers=['Content-Type'],
    methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
)

db = SQLAlchemy(app)

OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')  # Optional: for higher rate limits

# ── Redis (optional) ──────────────────────────────────────────────────────────
# Used as a fast message cache. Falls back gracefully to Postgres if unavailable.
REDIS_URL = os.environ.get('REDIS_URL', '')
_redis = None

def get_redis():
    global _redis
    if _redis is not None:
        return _redis
    if not REDIS_URL:
        return None
    try:
        client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _redis = client
        print("Redis connected")
        return _redis
    except Exception as e:
        print(f"Redis unavailable, using Postgres only: {e}")
        return None

CONV_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

def cache_get_messages(conv_id):
    r = get_redis()
    if r:
        try:
            val = r.get(f"conv:{conv_id}:messages")
            if val:
                return json.loads(val)
        except Exception:
            pass
    return None

def cache_set_messages(conv_id, messages):
    r = get_redis()
    if r:
        try:
            r.setex(f"conv:{conv_id}:messages", CONV_CACHE_TTL, json.dumps(messages))
        except Exception:
            pass

# Groq free models — fast inference, generous free tier. Primary cloud provider.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality on groq free tier
    "llama-3.1-8b-instant",      # fastest, good for chat
    "gemma2-9b-it",              # google gemma fallback
    "mixtral-8x7b-32768",        # mistral fallback
]

# OpenRouter free models — secondary cloud fallback if Groq is unavailable
OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2-7b-instruct:free",
]

# Recommended Ollama model config (shown in UI setup guide)
OLLAMA_RECOMMENDED = {
    "name": "llama3.2",
    "display": "Llama 3.2 (3B)",
    "size_gb": 2.0,
    "pull_cmd": "ollama pull llama3.2",
    "description": "Fast, capable — ideal for interview + resume tasks"
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('instance', exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────

class Student(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100))
    password_hash = db.Column(db.String(256), nullable=True)  # nullable for existing accounts
    college = db.Column(db.String(100), default='VIT Vellore')
    profile_data = db.Column(db.Text, default='{}')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    conversations = db.relationship('Conversation', backref='student', lazy=True)
    resumes = db.relationship('Resume', backref='student', lazy=True)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(36), db.ForeignKey('student.id'), nullable=False)
    messages = db.Column(db.Text, default='[]')
    topic = db.Column(db.String(100), default='general')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(36), db.ForeignKey('student.id'), nullable=False)
    job_description = db.Column(db.Text)
    resume_data = db.Column(db.Text)          # current (possibly edited) JSON
    edited_html = db.Column(db.Text)          # inline-edited HTML snapshot, if any
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    versions = db.relationship('ResumeVersion', backref='resume', lazy=True, order_by='ResumeVersion.version_number')

class ResumeVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resume.id'), nullable=False)
    version_number = db.Column(db.Integer, nullable=False, default=1)
    resume_data = db.Column(db.Text)          # JSON snapshot at this version
    edited_html = db.Column(db.Text)          # HTML snapshot at this version
    label = db.Column(db.String(100))         # e.g. "AI Generated", "Manually edited"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MockInterview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(36), db.ForeignKey('student.id'), nullable=False)
    job_title = db.Column(db.String(200))
    job_description = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, completed
    questions = db.Column(db.Text, default='[]')          # JSON list of question objects
    transcript = db.Column(db.Text, default='[]')         # JSON list of {question, answer, timestamp}
    report = db.Column(db.Text)                           # JSON evaluation report
    overall_score = db.Column(db.Integer)                 # 0-100
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

# ── Credit System ─────────────────────────────────────────────────────────────

# How many credits each action costs
CREDIT_COSTS = {
    'chat_message':       1,   # each AI chat message
    'generate_resume':    5,   # generate a tailored resume
    'mock_interview':    10,   # start a mock interview session
    'upload_document':    2,   # upload and parse a document
    'intro_script':       3,   # generate self-intro video script
}

# Starting credits and bonus amounts
CREDITS_ON_SIGNUP           = 50   # free credits for every new account
CREDITS_REFERRAL_BONUS      = 20   # credits awarded to a regular referrer when referral signs up
CREDITS_AMBASSADOR_BONUS    = 35   # higher reward for official campus ambassadors
CREDITS_REFEREE_BONUS       = 10   # extra credits awarded to the new user who used a referral code

class CreditLedger(db.Model):
    """Append-only ledger of every credit transaction for a student."""
    id          = db.Column(db.Integer, primary_key=True)
    student_id  = db.Column(db.String(36), db.ForeignKey('student.id'), nullable=False)
    delta       = db.Column(db.Integer, nullable=False)          # positive = earned, negative = spent
    action      = db.Column(db.String(80), nullable=False)       # e.g. 'signup_bonus', 'generate_resume'
    description = db.Column(db.String(255))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

class Ambassador(db.Model):
    """Campus ambassador / referral tracking."""
    id            = db.Column(db.Integer, primary_key=True)
    student_id    = db.Column(db.String(36), db.ForeignKey('student.id'), unique=True, nullable=False)
    referral_code = db.Column(db.String(20), unique=True, nullable=False)
    is_ambassador = db.Column(db.Boolean, default=False)         # True = official campus ambassador
    total_referrals = db.Column(db.Integer, default=0)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


# ── Credit helpers ────────────────────────────────────────────────────────────

def _get_balance(student_id):
    """Return current credit balance for a student."""
    row = db.session.execute(
        db.text("SELECT COALESCE(SUM(delta), 0) FROM credit_ledger WHERE student_id = :sid"),
        {'sid': student_id}
    ).fetchone()
    return int(row[0]) if row else 0


def _add_credits(student_id, delta, action, description=''):
    entry = CreditLedger(student_id=student_id, delta=delta, action=action, description=description)
    db.session.add(entry)
    db.session.commit()
    return _get_balance(student_id)


def _spend_credits(student_id, action):
    """Deduct credits for action. Returns (ok: bool, balance: int, cost: int)."""
    cost = CREDIT_COSTS.get(action, 0)
    if cost == 0:
        return True, _get_balance(student_id), 0
    balance = _get_balance(student_id)
    if balance < cost:
        return False, balance, cost
    entry = CreditLedger(student_id=student_id, delta=-cost, action=action,
                          description=f'Used {action.replace("_", " ")}')
    db.session.add(entry)
    db.session.commit()
    return True, balance - cost, cost


def _generate_referral_code(name, student_id):
    """Generate a short unique referral code like ARJUN-4F2A."""
    prefix = (name or 'USER').upper().split()[0][:6]
    suffix = student_id[-4:].upper()
    return f"{prefix}-{suffix}"


def _ensure_ambassador_record(student):
    """Create an Ambassador record if one doesn't exist yet."""
    if not Ambassador.query.filter_by(student_id=student.id).first():
        code = _generate_referral_code(student.name, student.id)
        amb = Ambassador(student_id=student.id, referral_code=code, is_ambassador=False)
        db.session.add(amb)
        db.session.commit()
        return amb
    return Ambassador.query.filter_by(student_id=student.id).first()

# ── LLM Helper ────────────────────────────────────────────────────────────────

def _call_groq(messages, system_prompt, max_tokens=1000):
    """Try Groq models in order. Returns response string or None."""
    if not GROQ_API_KEY:
        return None
    full_messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    for model in GROQ_MODELS:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": full_messages, "max_tokens": max_tokens},
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            elif resp.status_code == 429:
                print(f"Groq 429 on {model}, trying next...")
                continue
            else:
                print(f"Groq {resp.status_code} on {model}: {resp.text[:120]}")
                break
        except Exception as e:
            print(f"Groq error on {model}: {e}")
            break
    return None


def _call_openrouter(messages, system_prompt, max_tokens=1000):
    """Try OpenRouter free models in order. Returns response string or None."""
    if not OPENROUTER_API_KEY:
        return None
    full_messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kairo.app",
        "X-Title": "Kairo"
    }
    for model in OPENROUTER_FREE_MODELS:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": full_messages, "max_tokens": max_tokens},
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            elif resp.status_code in (429, 404):
                print(f"OpenRouter {resp.status_code} on {model}, trying next...")
                continue
            else:
                print(f"OpenRouter {resp.status_code} on {model}: {resp.text[:120]}")
                break
        except Exception as e:
            print(f"OpenRouter error on {model}: {e}")
            break
    return None


def _call_ollama(messages, system_prompt, max_tokens=1000):
    """Try local Ollama. Returns response string or None. Fully silent if not running."""
    try:
        ping = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if ping.status_code != 200:
            return None
    except Exception:
        return None  # not running — no log spam on Railway

    try:
        formatted = ""
        if system_prompt:
            formatted += f"<|system|>\n{system_prompt}\n"
        for m in messages:
            formatted += f"<|{m['role']}|>\n{m['content']}\n"
        formatted += "<|assistant|>\n"
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": formatted, "stream": False},
            timeout=90
        )
        if resp.status_code == 200:
            return resp.json().get('response', '')
    except Exception as e:
        print(f"Ollama generate error: {e}")
    return None


def call_llm(messages, system_prompt="", max_tokens=1000):
    """Call LLM with provider priority: Groq → OpenRouter → Ollama."""
    result = _call_groq(messages, system_prompt, max_tokens)
    if result:
        return result

    result = _call_openrouter(messages, system_prompt, max_tokens)
    if result:
        return result

    result = _call_ollama(messages, system_prompt, max_tokens)
    if result:
        return result

    return "I'm having trouble connecting to the AI service. Please add a GROQ_API_KEY (free at console.groq.com) to your environment variables."


def get_llm_status():
    """Return detailed status of all LLM backends for the UI."""
    status = {
        "groq": {"configured": False, "ok": False, "models": GROQ_MODELS},
        "openrouter": {"configured": False, "ok": False, "models": OPENROUTER_FREE_MODELS},
        "ollama": {
            "installed": False, "model_available": False,
            "model": OLLAMA_MODEL,
            "recommended": OLLAMA_RECOMMENDED,
            "url": OLLAMA_BASE_URL
        },
        "active_backend": None
    }

    # Check Groq
    if GROQ_API_KEY:
        status["groq"]["configured"] = True
        try:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=6
            )
            status["groq"]["ok"] = resp.status_code == 200
        except Exception:
            status["groq"]["ok"] = False

    # Check OpenRouter
    if OPENROUTER_API_KEY:
        status["openrouter"]["configured"] = True
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=6
            )
            status["openrouter"]["ok"] = resp.status_code == 200
        except Exception:
            status["openrouter"]["ok"] = False

    # Check Ollama — silent if not reachable
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            status["ollama"]["installed"] = True
            models_data = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models_data]
            status["ollama"]["model_available"] = OLLAMA_MODEL in model_names
            status["ollama"]["available_models"] = model_names
    except Exception:
        pass  # not running — expected on cloud deployments

    # Determine active backend (first one that's working)
    if status["groq"]["ok"]:
        status["active_backend"] = "groq"
    elif status["openrouter"]["ok"]:
        status["active_backend"] = "openrouter"
    elif status["ollama"]["installed"] and status["ollama"]["model_available"]:
        status["active_backend"] = "ollama"

    return status


def get_interview_system_prompt(profile_data):
    resume_context = (
        "\nNote: The student has already uploaded their existing resume — their profile is pre-populated. "
        "Acknowledge this warmly at the start, tell them what you already know, and only ask about gaps "
        "or things you want to dig deeper on. Don't re-ask information you already have."
    ) if profile_data.get("resume_uploaded") else ""
    return f"""You are Kairo, a warm and empathetic AI career co-pilot helping a student build their professional profile.

Think of yourself as a well-wisher — a senior who genuinely cares about the student's journey and wants to understand their full story, not just their resume bullet points.

Your approach:
1. Start by understanding their TIMELINE — college, year, branch, location
2. Dig into ACADEMICS — CGPA, favourite subjects, what clicked and what didn't
3. Explore STRUGGLES honestly — bad semesters, setbacks, hard moments (these show resilience)
4. Celebrate WINS — hackathons, awards, proud moments, things they're genuinely excited about
5. Go deep on PROJECTS — what problem, what tech, what impact, what they personally built
6. Explore INTERNSHIPS — what they worked on, what they learned, real outcomes
7. Map out SKILLS — not just a list, understand what they're truly strong in vs. dabbled in
8. Understand EXTRACURRICULARS — clubs, events organised, communities led
9. Understand GOALS — where they want to be, what excites them about the future

Current profile data already collected:
{json.dumps(profile_data, indent=2)}

If a field already has data, skip or briefly confirm it — don't re-ask things you already know.
{resume_context}

Guidelines:
- Ask ONE focused question at a time — never multi-part questions
- Sound like a human who cares, not a form being filled out
- When they mention something vague, ask ONE good follow-up: "What was the actual impact?" / "What did you build specifically?" / "How long did that take?"
- Extract specifics: numbers, tech stack, outcomes, timelines, team size, role
- Keep each response to 2-3 sentences max — this is a conversation, not a lecture
- When you have covered all major areas comprehensively, say "PROFILE_COMPLETE" at the very start of your response

Start warmly — greet them by name if you know it, and ask where they're studying and what they're doing."""


def get_resume_system_prompt(profile_data, job_description):
    return f"""You are an expert resume writer specializing in fresher resumes for tech students.

Student Profile:
{json.dumps(profile_data, indent=2)}

Job Description:
{job_description}

Generate a complete, ATS-optimized resume in JSON format with these exact keys:
{{
  "name": "student name",
  "email": "email",
  "phone": "phone if available",
  "linkedin": "linkedin if available", 
  "github": "github if available",
  "summary": "2-3 line professional summary tailored to JD",
  "education": [{{"degree": "", "institution": "", "year": "", "cgpa": ""}}],
  "skills": {{"technical": [], "tools": [], "soft": []}},
  "projects": [{{"name": "", "description": "", "tech": [], "impact": ""}}],
  "experience": [{{"role": "", "company": "", "duration": "", "points": []}}],
  "certifications": [],
  "achievements": [],
  "extracurriculars": []
}}

Only include sections where real data exists. Tailor everything to match the job description keywords.
Return ONLY valid JSON, no markdown."""


def extract_profile_from_conversation(messages):
    """Use LLM to extract structured profile from conversation"""
    system = """Extract a structured student career profile from this conversation.
Return ONLY valid JSON with these exact keys (use empty arrays/strings for fields not mentioned — never invent data):
{
  "name": "", "email": "", "phone": "",
  "college": "", "branch": "", "year": "", "cgpa": "", "location": "",
  "subjects": [{"name": "", "grade": "", "interest": "high/medium/low"}],
  "projects": [{"name": "", "description": "", "tech": [], "role": "", "impact": ""}],
  "internships": [{"company": "", "role": "", "duration": "", "work": "", "description": ""}],
  "skills": {"technical": [], "tools": [], "languages": []},
  "clubs": [], "hackathons": [], "certifications": [],
  "achievements": [], "highlights": [], "struggles": [],
  "goals": "", "industrial_visits": [],
  "extracurriculars": [{"activity": "", "type": "", "role": "", "year": "", "description": ""}],
  "faculty_references": [{"name": "", "designation": "", "department": "", "institution": "", "email": "", "phone": "", "context": ""}]
}"""

    # Strip UI-only metadata before sending to LLM
    clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("_silent")]
    conv_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in clean_messages])
    result = call_llm([{"role": "user", "content": f"Conversation:\n{conv_text}\n\nExtract the profile:"}], system)

    try:
        clean = result.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean)
    except:
        return {}


def deep_merge_profile(existing, new_data):
    """Deep merge new profile data into existing — lists are unioned, strings overwrite only if new value is non-empty."""
    for key, new_val in new_data.items():
        if not new_val and new_val != 0:
            continue  # skip empty strings / empty lists from new extraction
        if key not in existing or not existing[key]:
            existing[key] = new_val
        elif isinstance(new_val, list) and isinstance(existing[key], list):
            # Merge lists by name field to avoid duplicates
            existing_names = {str(item.get('name', item) if isinstance(item, dict) else item).lower() for item in existing[key]}
            for item in new_val:
                item_name = str(item.get('name', item) if isinstance(item, dict) else item).lower()
                if item_name not in existing_names:
                    existing[key].append(item)
                    existing_names.add(item_name)
        elif isinstance(new_val, dict) and isinstance(existing[key], dict):
            deep_merge_profile(existing[key], new_val)
        elif isinstance(new_val, str) and new_val.strip():
            existing[key] = new_val  # strings: overwrite only if new value is non-empty
    return existing

# ── Routes ────────────────────────────────────────────────────────────────────

# ── Demo rate-limit endpoints ─────────────────────────────────────────────────
DEMO_MSG_LIMIT = 3   # exchanges before sign-up gate

@app.route('/api/demo/check', methods=['POST'])
def demo_check():
    data    = request.get_json(silent=True) or {}
    demo_id = str(data.get('demo_id', ''))[:64]
    if not demo_id:
        return jsonify({'used': 0})
    used = session.get(f'demo:{demo_id}', 0)
    return jsonify({'used': used, 'limit': DEMO_MSG_LIMIT})

@app.route('/api/demo/sync', methods=['POST'])
def demo_sync():
    """Only ever increase the count — never decrease it."""
    data    = request.get_json(silent=True) or {}
    demo_id = str(data.get('demo_id', ''))[:64]
    claimed = int(data.get('used', 0))
    if not demo_id:
        return jsonify({'ok': False})
    session.permanent = True
    key     = f'demo:{demo_id}'
    current = session.get(key, 0)
    session[key] = max(current, min(claimed, DEMO_MSG_LIMIT + 1))
    return jsonify({'ok': True, 'used': session[key]})

@app.route('/api/demo/chat', methods=['POST'])
def demo_chat():
    """Demo AI chat — uses Groq via the same LLM stack, no auth required."""
    data    = request.get_json(silent=True) or {}
    demo_id = str(data.get('demo_id', ''))[:64]
    message = str(data.get('message', '')).strip()[:2000]
    system  = str(data.get('system', '')).strip()[:2000]
    history = data.get('messages', [])

    if not message:
        return jsonify({'error': 'empty message'}), 400

    # Enforce server-side limit
    key  = f'demo:{demo_id}'
    used = session.get(key, 0) if demo_id else 0
    if used >= DEMO_MSG_LIMIT:
        return jsonify({'error': 'limit_reached', 'message': 'Demo limit reached.'}), 403

    # Build message list — strip any internal _silent flags
    msgs = [{'role': m['role'], 'content': m['content']}
            for m in history if m.get('role') in ('user', 'assistant')]

    reply = call_llm(msgs, system_prompt=system, max_tokens=300)

    # Increment count
    if demo_id:
        session.permanent = True
        session[key] = min(used + 1, DEMO_MSG_LIMIT + 1)

    return jsonify({'response': reply})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')

@app.route('/interview-prep')
def interview_prep_page():
    return render_template('interview_prep.html')

@app.route('/resume')
def resume_page():
    return render_template('resume.html')

@app.route('/presence')
def presence_page():
    return render_template('presence.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').lower().strip()
    name = data.get('name', '').strip()
    password = data.get('password', '').strip()
    referral_code = data.get('referral_code', '').strip().upper()

    if not email:
        return jsonify({'error': 'Email required'}), 400
    if not password:
        return jsonify({'error': 'Password required'}), 400

    # Auto-correct common typo
    email = email.replace('@vitsudent.ac.in', '@vitstudent.ac.in')

    VALID_VIT_DOMAINS = ('@vitstudent.ac.in', '@vit.ac.in')
    if not any(email.endswith(d) for d in VALID_VIT_DOMAINS):
        return jsonify({'error': 'Please use your VIT email (e.g. name2022@vitstudent.ac.in)'}), 400

    student = Student.query.filter_by(email=email).first()
    is_new = not student
    referral_msg = None

    if is_new:
        # New account — register
        if not name:
            return jsonify({'error': 'Name required for new accounts'}), 400
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        student = Student(
            email=email,
            name=name,
            password_hash=generate_password_hash(password)
        )
        db.session.add(student)
        db.session.commit()

        # Award signup credits
        _add_credits(student.id, CREDITS_ON_SIGNUP, 'signup_bonus',
                     f'Welcome to Kairo! {CREDITS_ON_SIGNUP} free credits.')

        # Process referral code if provided
        if referral_code:
            referrer_amb = Ambassador.query.filter_by(referral_code=referral_code).first()
            if referrer_amb and referrer_amb.student_id != student.id:
                _add_credits(student.id, CREDITS_REFEREE_BONUS, 'referral_bonus',
                             f'Joined via referral code {referral_code}')
                # Ambassadors earn more per referral
                reward = CREDITS_AMBASSADOR_BONUS if referrer_amb.is_ambassador else CREDITS_REFERRAL_BONUS
                _add_credits(referrer_amb.student_id, reward, 'referral_reward',
                             f'Referred a new student ({email})')
                referrer_amb.total_referrals += 1
                db.session.commit()
                referral_msg = f'Referral applied! You got {CREDITS_REFEREE_BONUS} extra credits.'
            else:
                referral_msg = 'Referral code not found — no bonus applied.'

        # Give every new student their own referral code
        _ensure_ambassador_record(student)
    else:
        # Existing account
        if student.password_hash is None:
            # Legacy account with no password — set one now
            if len(password) < 6:
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            student.password_hash = generate_password_hash(password)
            if name and not student.name:
                student.name = name
            db.session.commit()
        elif not check_password_hash(student.password_hash, password):
            return jsonify({'error': 'Incorrect password'}), 401
        # Back-fill ambassador record for legacy accounts
        _ensure_ambassador_record(student)

    session['student_id'] = student.id
    session['student_email'] = student.email

    amb = Ambassador.query.filter_by(student_id=student.id).first()
    return jsonify({
        'id': student.id,
        'email': student.email,
        'name': student.name,
        'profile': json.loads(student.profile_data or '{}'),
        'credits': _get_balance(student.id),
        'referral_code': amb.referral_code if amb else None,
        'is_ambassador': amb.is_ambassador if amb else False,
        'is_new': is_new,
        'referral_msg': referral_msg,
    })

@app.route('/api/auth/me')
def me():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    _ensure_ambassador_record(student)
    amb = Ambassador.query.filter_by(student_id=student.id).first()
    return jsonify({
        'id': student.id,
        'email': student.email,
        'name': student.name,
        'profile': json.loads(student.profile_data or '{}'),
        'credits': _get_balance(student.id),
        'referral_code': amb.referral_code if amb else None,
        'is_ambassador': amb.is_ambassador if amb else False,
        'total_referrals': amb.total_referrals if amb else 0,
    })

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'ok': True})

@app.route('/api/auth/update-name', methods=['POST'])
def update_name():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Not found'}), 404
    new_name = request.json.get('name', '').strip()
    if not new_name:
        return jsonify({'error': 'Name cannot be empty'}), 400
    student.name = new_name
    db.session.commit()
    return jsonify({'ok': True, 'name': student.name})

@app.route('/api/auth/change-password', methods=['POST'])
def change_password():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Not found'}), 404
    data = request.json
    current = data.get('current_password', '')
    new_pw = data.get('new_password', '').strip()
    if student.password_hash and not check_password_hash(student.password_hash, current):
        return jsonify({'error': 'Current password is incorrect'}), 401
    if len(new_pw) < 6:
        return jsonify({'error': 'New password must be at least 6 characters'}), 400
    student.password_hash = generate_password_hash(new_pw)
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    conversation_id = data.get('conversation_id')

    sid = session.get('student_id')
    if not sid:
        sid = 'demo'

    # extract_only: one-shot LLM call, never saved to conversation history
    if data.get('extract_only'):
        extract_system = "You are a JSON extractor. Return only valid JSON, no markdown, no explanation."
        result = call_llm([{"role": "user", "content": user_message}], extract_system)
        return jsonify({'response': result, 'conversation_id': None, 'profile_complete': False, 'messages': []})

    # Credit check for real users (silent bootstraps are free, cost borne by first real message)
    if sid != 'demo' and not data.get('silent_bootstrap'):
        ok, balance, cost = _spend_credits(sid, 'chat_message')
        if not ok:
            return jsonify({
                'error': 'insufficient_credits',
                'message': f'You need {cost} credit(s) but only have {balance}. Earn more by referring friends!',
                'balance': balance,
                'cost': cost
            }), 402

    # Get or create conversation
    conv = None
    if conversation_id:
        conv = Conversation.query.get(conversation_id)

    if not conv and sid != 'demo':
        # Resume latest open conversation if one exists
        conv = Conversation.query.filter_by(
            student_id=sid, topic='profile_building'
        ).order_by(Conversation.created_at.desc()).first()

    if not conv and sid != 'demo':
        conv = Conversation(student_id=sid, topic='profile_building')
        db.session.add(conv)
        db.session.commit()

    # Load messages — Redis first, Postgres fallback
    if conv:
        messages = cache_get_messages(conv.id)
        if messages is None:
            messages = json.loads(conv.messages or '[]')
    else:
        messages = data.get('messages', [])

    # Get student profile for context
    profile_data = {}
    if sid and sid != 'demo':
        student = Student.query.get(sid)
        if student:
            profile_data = json.loads(student.profile_data or '{}')

    # Add user message — tag silent bootstrap so the UI never renders it on reload
    silent_bootstrap = data.get('silent_bootstrap', False)
    user_msg = {"role": "user", "content": user_message}
    if silent_bootstrap:
        user_msg["_silent"] = True
    messages.append(user_msg)

    # Call LLM — strip any UI-only metadata (_silent) before sending
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    system_prompt = get_interview_system_prompt(profile_data)
    ai_response = call_llm(llm_messages, system_prompt)

    # Add AI response
    messages.append({"role": "assistant", "content": ai_response})

    # Check if profile complete
    profile_complete = "PROFILE_COMPLETE" in ai_response
    if profile_complete:
        ai_response = ai_response.replace("PROFILE_COMPLETE", "").strip()
        messages[-1]['content'] = ai_response

        if sid and sid != 'demo':
            extracted = extract_profile_from_conversation(messages)
            student = Student.query.get(sid)
            if student and extracted:
                existing = json.loads(student.profile_data or '{}')
                merged = deep_merge_profile(existing, extracted)
                student.profile_data = json.dumps(merged)
                db.session.commit()

    # Persist — keep last 100 messages in both stores
    trimmed = messages[-100:]
    if conv:
        conv.messages = json.dumps(trimmed)
        conv.updated_at = datetime.utcnow()
        db.session.commit()
        cache_set_messages(conv.id, trimmed)

    return jsonify({
        'response': ai_response,
        'conversation_id': conv.id if conv else None,
        'profile_complete': profile_complete,
        'messages': trimmed
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    sid = session.get('student_id')

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower()

    if ext not in ['pdf', 'png', 'jpg', 'jpeg']:
        return jsonify({'error': 'Only PDF and images allowed'}), 400

    # Credit check
    if sid:
        ok, balance, cost = _spend_credits(sid, 'upload_document')
        if not ok:
            return jsonify({
                'error': 'insufficient_credits',
                'message': f'Uploading a document costs {cost} credits but you only have {balance}.',
                'balance': balance, 'cost': cost
            }), 402
    
    unique_name = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    
    # Extract text from PDF
    extracted_text = ""
    if ext == 'pdf':
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() or ""
        except Exception as e:
            extracted_text = f"Could not extract text: {e}"
    
    # Use LLM to extract structured info from document as JSON
    doc_insights_raw = ""
    doc_structured = {}
    if extracted_text and extracted_text.strip() and not extracted_text.startswith("Could not"):
        doc_insights_raw = call_llm(
            [{"role": "user", "content": f"Document text:\n{extracted_text[:4000]}"}],
            """You are analyzing a student academic document (transcript, marksheet, certificate, or resume).
Extract ALL useful information and return ONLY a valid JSON object with these fields (omit fields that are not present):
{
  "name": "student full name if found",
  "college": "institution name",
  "branch": "degree/program/branch",
  "cgpa": "CGPA or percentage",
  "year": "graduation year or semester info",
  "roll_number": "roll/registration number",
  "semester": "current semester if found",
  "subjects": ["list of course/subject names"],
  "grades": {"subject_name": "grade/marks"},
  "certifications": ["list of certifications"],
  "achievements": ["list of achievements/awards"],
  "skills": ["skills mentioned"],
  "document_type": "transcript|marksheet|certificate|resume|other"
}
Return ONLY the JSON, no explanation, no markdown fences."""
        )
        # Parse LLM response as JSON
        try:
            clean = doc_insights_raw.strip()
            if "```" in clean:
                for part in clean.split("```"):
                    if '{' in part:
                        clean = part.lstrip("json").strip()
                        break
            doc_structured = json.loads(clean)
        except Exception:
            # Fallback: store raw text under a key so frontend can still show something
            doc_structured = {"raw_summary": doc_insights_raw}

    # Save extracted document data into student's profile
    if sid and doc_structured:
        student = Student.query.get(sid)
        if student:
            existing = json.loads(student.profile_data or '{}')
            # Merge document data — keep a list of uploaded docs and merge structured fields
            uploaded_docs = existing.get('uploaded_documents', [])
            uploaded_docs.append({
                'filename': unique_name,
                'original_name': filename,
                'extracted': doc_structured
            })
            existing['uploaded_documents'] = uploaded_docs

            # Merge top-level fields into profile only if not already set
            mergeable = ['name', 'college', 'branch', 'cgpa', 'year', 'roll_number', 'semester']
            for field in mergeable:
                if doc_structured.get(field) and not existing.get(field):
                    existing[field] = doc_structured[field]

            # Merge lists (subjects, certifications, achievements, skills)
            for list_field in ['subjects', 'certifications', 'achievements']:
                if doc_structured.get(list_field):
                    existing_list = existing.get(list_field, [])
                    merged = list({v: True for v in (existing_list + doc_structured[list_field])}.keys())
                    existing[list_field] = merged

            # Merge skills into structured skills object
            if doc_structured.get('skills'):
                sk = existing.get('skills', {})
                if isinstance(sk, dict):
                    tech = sk.get('technical', [])
                    merged_skills = list({v: True for v in (tech + doc_structured['skills'])}.keys())
                    sk['technical'] = merged_skills
                    existing['skills'] = sk

            # Store the full structured doc data for display in dashboard
            existing['document_data'] = doc_structured

            student.profile_data = json.dumps(existing)
            db.session.commit()

    return jsonify({
        'filename': unique_name,
        'original_name': filename,
        'extracted_text': extracted_text[:500] if extracted_text else '',
        'insights': doc_insights_raw,
        'structured': doc_structured
    })

@app.route('/api/resume/parse-upload', methods=['POST'])
def resume_parse_upload():
    """
    Parse an existing resume PDF and merge the extracted structured data
    into the student's profile so the Build Profile chat starts pre-populated.
    No credit charge — this is an onboarding helper, not a generation step.
    """
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext != 'pdf':
        return jsonify({'error': 'Only PDF resumes are supported'}), 400

    unique_name = f"resume_{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    # ── Extract raw text ──────────────────────────────────────────────────────
    extracted_text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                extracted_text += (page.extract_text() or "") + "\n"
    except Exception as e:
        return jsonify({'error': f'Could not read PDF: {e}'}), 422

    if not extracted_text.strip():
        return jsonify({'error': 'PDF appears to be empty or image-only — please use a text-based PDF'}), 422

    # ── LLM: extract rich structured profile from resume ─────────────────────
    RESUME_EXTRACT_PROMPT = """You are parsing a student resume to extract a comprehensive structured career profile.
Return ONLY valid JSON (no markdown, no explanation) with these exact keys — use empty arrays/strings for absent fields, never invent data:
{
  "name": "",
  "email": "",
  "phone": "",
  "linkedin": "",
  "github": "",
  "college": "",
  "branch": "",
  "year": "",
  "cgpa": "",
  "location": "",
  "summary": "",
  "projects": [{"name": "", "description": "", "tech": [], "role": "", "impact": "", "url": "", "duration": ""}],
  "internships": [{"company": "", "role": "", "duration": "", "work": "", "description": ""}],
  "skills": {"technical": [], "tools": [], "languages": []},
  "certifications": [],
  "achievements": [],
  "hackathons": [],
  "extracurriculars": [{"activity": "", "type": "", "role": "", "year": "", "description": ""}],
  "clubs": [],
  "goals": "",
  "highlights": []
}"""

    raw = call_llm(
        [{"role": "user", "content": f"Resume text:\n{extracted_text[:5000]}\n\nExtract the structured profile:"}],
        RESUME_EXTRACT_PROMPT,
        max_tokens=1800
    )

    extracted_profile = {}
    try:
        clean = raw.strip()
        if "```" in clean:
            for part in clean.split("```"):
                if '{' in part:
                    clean = part.lstrip("json").strip()
                    break
        extracted_profile = json.loads(clean)
    except Exception:
        extracted_profile = {}

    if not extracted_profile:
        return jsonify({'error': 'Could not parse resume — try a cleaner PDF'}), 422

    # ── Merge into student profile ────────────────────────────────────────────
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Student not found'}), 404

    existing = json.loads(student.profile_data or '{}')

    # Track uploaded resumes separately so we can show them in the UI
    uploaded_resumes = existing.get('uploaded_resumes', [])
    uploaded_resumes.append({
        'filename': unique_name,
        'original_name': filename,
        'uploaded_at': datetime.utcnow().isoformat(),
    })
    existing['uploaded_resumes'] = uploaded_resumes
    existing['resume_uploaded'] = True   # flag the chat bootstrap can check

    merged = deep_merge_profile(existing, extracted_profile)
    student.profile_data = json.dumps(merged)
    db.session.commit()

    # Return a subset for the UI preview (exclude verbose raw text)
    preview = {k: v for k, v in extracted_profile.items()
               if v and k not in ('summary',)}

    return jsonify({
        'ok': True,
        'profile': extracted_profile,
        'preview': preview,
        'message': 'Resume parsed and profile pre-filled! Kairo will skip questions it already knows the answers to.',
    })


@app.route('/api/documents/<int:doc_index>', methods=['DELETE'])
def delete_document(doc_index):
    """Remove a document and its merged data contribution from the student profile."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Not found'}), 404

    existing = json.loads(student.profile_data or '{}')
    uploaded_docs = existing.get('uploaded_documents', [])

    if doc_index < 0 or doc_index >= len(uploaded_docs):
        return jsonify({'error': 'Document not found'}), 404

    # Remove the document entry
    removed = uploaded_docs.pop(doc_index)
    existing['uploaded_documents'] = uploaded_docs

    # Rebuild merged profile fields from remaining docs
    # Reset fields that may have come from documents, then re-merge remaining docs
    doc_scalar_fields = ['name', 'college', 'branch', 'cgpa', 'year', 'roll_number', 'semester']
    # We'll only reset fields if they aren't confirmed from interview/manual edit
    # Simple approach: rebuild from scratch using remaining docs
    remaining_extracted = [d.get('extracted', {}) for d in uploaded_docs]

    # Remove document-sourced list contributions and rebuild
    for list_field in ['subjects', 'certifications', 'achievements']:
        merged = []
        for ext in remaining_extracted:
            if ext.get(list_field):
                merged += ext[list_field]
        existing[list_field] = list(dict.fromkeys(merged)) if merged else existing.get(list_field, [])

    # Reset skills.technical to only remaining doc skills (keep interview skills separately would be ideal,
    # but for simplicity rebuild from remaining docs; interview-sourced skills are also in skills.technical)
    # We keep this minimal — just remove the deleted doc's unique skills isn't easily trackable, so
    # we leave skills untouched (they could have come from interview too).

    # Update document_data to last remaining doc or empty
    if uploaded_docs:
        existing['document_data'] = uploaded_docs[-1].get('extracted', {})
    else:
        existing.pop('document_data', None)

    # Try to delete the actual file
    removed_filename = removed.get('filename', '')
    if removed_filename:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], removed_filename))
        except Exception:
            pass

    student.profile_data = json.dumps(existing)
    db.session.commit()

    return jsonify({'ok': True, 'remaining': len(uploaded_docs)})



@app.route('/api/fetch-jd', methods=['POST'])
def fetch_jd():
    """
    Scrape a job posting URL and return clean JD text.
    Strategy:
      0. Naukri-specific: extract job ID and call their internal API
      1. Site-specific CSS selectors for known job boards
      2. Heuristic fallback — score all block elements by length & keyword density
      3. LLM cleanup pass to extract only the actual JD content
    """
    import re as _re
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return jsonify({'error': 'Server missing beautifulsoup4 — contact admin'}), 500

    data = request.json or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Normalise — ensure scheme
    if not url.startswith('http'):
        url = 'https://' + url

    parsed_host = url.split('/')[2].lower().replace('www.', '')

    # ── 0a. Naukri — use dedicated scraper first, fallback with URL hints ───────
    if 'naukri.com' in parsed_host:
        naukri_jd = _fetch_naukri_jd(url)
        if naukri_jd and len(naukri_jd.strip()) > 100:
            return jsonify({'jd': naukri_jd, 'source': 'Naukri'})
        return jsonify({
            'error': 'Scraping failed. Paste the job description below.',
            'url_hints': _extract_url_hints(url, 'naukri'),
            'manual_paste': True,
            'source': 'Naukri',
        }), 422

    # ── 0b. Internshala — dedicated scraper ──────────────────────────────────────
    if 'internshala.com' in parsed_host:
        result = _fetch_internshala_jd(url)
        if result and len(result.get('jd', '').strip()) > 100:
            return jsonify({**result, 'source': 'Internshala'})
        return jsonify({
            'error': 'Scraping failed. Paste the job description below.',
            'url_hints': _extract_url_hints(url, 'internshala'),
            'manual_paste': True,
            'source': 'Internshala',
        }), 422

    # ── 0c. Hirist — SSG scraper ──────────────────────────────────────────────────
    if 'hirist.tech' in parsed_host or 'hirist.com' in parsed_host:
        hirist_jd = _fetch_hirist_jd(url)
        if hirist_jd and len(hirist_jd.strip()) > 100:
            return jsonify({'jd': hirist_jd, 'source': 'Hirist'})
        return jsonify({
            'error': 'Scraping failed. Paste the job description below.',
            'url_hints': _extract_url_hints(url, 'hirist'),
            'manual_paste': True,
            'source': 'Hirist',
        }), 422

    # ── 0d. JS-rendered sites — return helpful manual-paste error immediately ──
    # These sites load job content via JavaScript after page load.
    # Server-side requests only get an empty HTML shell — scraping cannot work.
    JS_RENDERED_SITES = {
        'linkedin.com':     'LinkedIn',
        'instahyre.com':    'Instahyre',
        'wellfound.com':    'Wellfound',
        'cutshort.io':      'Cutshort',
        'greenhouse.io':    'Greenhouse',
        'lever.co':         'Lever',
        'workday.com':      'Workday',
    }
    for js_host, js_name in JS_RENDERED_SITES.items():
        if js_host in parsed_host:
            return jsonify({
                'error': f'{js_name} uses client-side rendering so auto-import is not possible. Paste the job description below.',
                'url_hints': _extract_url_hints(url, js_host.split('.')[0]),
                'manual_paste': True,
                'source': js_name,
            }), 422

    # Site-specific selectors: (host_substring, css_selector)
    # Note: naukri, internshala, hirist handled by dedicated scrapers above.
    SITE_SELECTORS = [
        ('foundit.in',        'div.jobDescription, div.job-description'),
        ('instahire.in',      'div.job-description, div.jd-content'),
        ('indeed.com',        'div#jobDescriptionText, div.jobsearch-jobDescriptionText'),
        ('glassdoor.com',     'div.jobDescriptionContent, div[data-test="jobDescriptionContent"]'),
        ('shine.com',         'div.job-description, div.jd-content-wrap'),
        ('monster.com',       'div.job-description, div#JobDescription'),
        ('freshersworld.com', 'div.job-description, div.jobdetails-section'),
        ('timesjobs.com',     'div.jd-desc, div[class*="job-desc"]'),
    ]

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.naukri.com/',
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return jsonify({'error': 'The page took too long to respond. Try again or paste the JD manually.'}), 504
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        if code == 403:
            return jsonify({'error': 'This site blocks automated access. Please paste the JD manually.'}), 422
        return jsonify({'error': f'Could not load the page (HTTP {code}). Try pasting the JD manually.'}), 422
    except Exception as e:
        return jsonify({'error': f'Could not reach that URL: {str(e)}'}), 500

    # Check if response is actually readable HTML (not binary/garbled)
    try:
        resp.encoding = resp.apparent_encoding or 'utf-8'
        page_text = resp.text
    except Exception:
        page_text = resp.content.decode('utf-8', errors='replace')

    soup = BeautifulSoup(page_text, 'html.parser')

    # Remove boilerplate tags
    for tag in soup(['script', 'style', 'noscript', 'nav', 'footer', 'header',
                     'aside', 'form', 'iframe', 'button', 'svg', 'img']):
        tag.decompose()

    jd_text = ''
    source_name = parsed_host.split('.')[0].capitalize()

    # 1. Try site-specific selectors first
    for host_sub, selector_str in SITE_SELECTORS:
        if host_sub in parsed_host:
            for sel in selector_str.split(','):
                el = soup.select_one(sel.strip())
                if el:
                    candidate = el.get_text(separator='\n', strip=True)
                    if len(candidate) > 200:
                        jd_text = candidate
                        break
            if jd_text:
                break

    # 2. Generic heuristic — score block elements
    if not jd_text:
        JD_KEYWORDS = [
            'responsibilities', 'requirements', 'qualifications', 'skills',
            'experience', 'bachelor', 'degree', 'must have', 'nice to have',
            'we are looking', 'you will', 'job description', 'about the role',
            "what you'll do", 'what we expect', 'minimum', 'preferred',
        ]
        best_score = 0
        best_el = None
        for el in soup.find_all(['div', 'section', 'article', 'main', 'td']):
            text = el.get_text(separator=' ', strip=True)
            if len(text) < 200 or len(text) > 15000:
                continue
            # Score by length + keyword hits
            kw_hits = sum(1 for kw in JD_KEYWORDS if kw in text.lower())
            score = len(text) * 0.01 + kw_hits * 40
            if score > best_score:
                best_score = score
                best_el = el
        if best_el:
            jd_text = best_el.get_text(separator='\n', strip=True)

    # 3. Whole-page fallback
    if not jd_text:
        jd_text = soup.get_text(separator='\n', strip=True)

    if not jd_text or len(jd_text) < 100:
        return jsonify({
            'error': 'Could not extract any text from that page. '
                     'The site may require login or block bots. Please paste the JD manually.'
        }), 422

    # 4. Deduplicate blank lines and trim
    lines = [l.strip() for l in jd_text.splitlines()]
    lines = [l for l in lines if l]  # drop empties
    # Remove consecutive duplicate lines (navigation artifacts)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    jd_text = '\n'.join(deduped)

    # 5. LLM cleanup pass — extract only actual JD content, strip nav/ads/noise
    if len(jd_text) > 400:
        cleanup_prompt = (
            "You are given raw text scraped from a job posting webpage. "
            "Extract ONLY the actual job description content: role summary, responsibilities, "
            "requirements, qualifications, skills, and any other relevant job details. "
            "Remove all navigation text, ads, cookie notices, company boilerplate headers/footers, "
            "and unrelated page content. "
            "Return the cleaned job description as plain text, preserving the original wording. "
            "Do NOT summarise or rewrite. Extract faithfully. "
            "If there is no recognisable job description, return exactly: NO_JD_FOUND\n\n"
            + "RAW TEXT:\n" + jd_text[:6000]
        )
        cleaned = call_llm([{"role": "user", "content": cleanup_prompt}],
                           "You are a precise text extractor. Output only the extracted job description, nothing else.")
        if cleaned and 'NO_JD_FOUND' not in cleaned and len(cleaned) > 150:
            jd_text = cleaned.strip()

    # Cap final output
    jd_text = jd_text[:5000]

    return jsonify({'jd': jd_text, 'source': source_name, 'url': url})


def _extract_url_hints(url, source):
    """
    Parse whatever structured info is visible in the URL slug and return it
    as a dict so the frontend can pre-fill fields and show the user what was
    found before asking them to paste the full JD.

    Examples:
      Naukri:      /job-listings-fresher-full-stack-developer-parsh-technologies-llp-rajkot-0-to-1-years-110326928982
      Internshala: /job/detail/remote-artificial-intelligence-ai-specialist-job-at-bryckel-ai1773309672
      Hirist:      /j/senior-backend-developer-healthify-123456
    """
    import re as _re
    from urllib.parse import urlparse, unquote

    hints = {}
    try:
        parsed = urlparse(url)
        # Take the last meaningful path segment, strip query params
        slug = parsed.path.rstrip('/').split('/')[-1]
        slug = unquote(slug)

        # ── Naukri ────────────────────────────────────────────────────────────
        # Pattern: job-listings-<role tokens>-<company tokens>-<location>-<exp>-<job_id>
        # Strip leading "job-listings-" prefix if present
        if source == 'naukri':
            slug = _re.sub(r'^job-listings-', '', slug)
            # Strip trailing numeric job ID
            slug = _re.sub(r'-\d{8,}$', '', slug)
            # Experience pattern e.g. "0-to-1-years" or "3-to-5-years"
            exp_match = _re.search(r'(\d+)-to-(\d+)-year', slug)
            if exp_match:
                hints['experience'] = f"{exp_match.group(1)}-{exp_match.group(2)} years"
                slug = slug[:exp_match.start()].rstrip('-')
            _parse_role_company_location(slug, hints)

        # ── Internshala ───────────────────────────────────────────────────────
        # Pattern: <role>-job-at-<company><numeric_id>  OR  <role>-internship-at-<company>
        elif source == 'internshala':
            slug = _re.sub(r'\d{8,}$', '', slug).rstrip('-')
            # Split on "-job-at-" or "-internship-at-"
            for sep in ('-job-at-', '-internship-at-', '-at-'):
                if sep in slug:
                    role_part, company_part = slug.split(sep, 1)
                    hints['role'] = _slug_to_title(role_part)
                    hints['company'] = _slug_to_title(company_part)
                    if 'internship' in sep or 'internship' in slug:
                        hints['job_type'] = 'Internship'
                    break
            else:
                _parse_role_company_location(slug, hints)

        # ── Hirist ────────────────────────────────────────────────────────────
        # Pattern: <role-words>-<company>-<job_id>
        # After stripping the job ID, split at the last role-keyword boundary.
        elif source == 'hirist':
            slug = _re.sub(r'-\d{4,}$', '', slug)
            ROLE_WORDS = {'developer','engineer','analyst','designer','manager',
                          'lead','senior','junior','fresher','intern','specialist',
                          'architect','consultant','associate','executive','backend',
                          'frontend','fullstack','devops','data','mobile','cloud'}
            tokens = slug.split('-')
            last_role_idx = 0
            for i, t in enumerate(tokens):
                if t.lower() in ROLE_WORDS:
                    last_role_idx = i
            hints['role'] = _slug_to_title('-'.join(tokens[:last_role_idx + 1]))
            company_tokens = tokens[last_role_idx + 1:]
            if company_tokens:
                hints['company'] = _slug_to_title('-'.join(company_tokens))

        # ── Generic fallback for LinkedIn, etc. ───────────────────────────────
        else:
            slug = _re.sub(r'[-_]?\d{6,}$', '', slug)
            _parse_role_company_location(slug, hints)

    except Exception as e:
        print(f"_extract_url_hints error: {e}")

    return hints  # may be empty dict — callers handle that gracefully


def _slug_to_title(slug):
    """Convert a hyphen-separated slug to Title Case, preserving common acronyms."""
    ACRONYMS = {'ai', 'ml', 'api', 'ui', 'ux', 'ios', 'sql', 'aws', 'gcp',
                'llp', 'llc', 'pvt', 'ltd', 'sde', 'swe', 'hr', 'it', 'bi'}
    words = slug.replace('-', ' ').replace('_', ' ').split()
    result = []
    for w in words:
        if w.lower() in ACRONYMS:
            result.append(w.upper())
        else:
            result.append(w.capitalize())
    return ' '.join(result)


def _parse_role_company_location(slug, hints):
    """
    Heuristically split a slug into role / company / location tokens.
    Works by finding location and company keywords as anchors.
    """
    import re as _re

    LOCATION_KEYWORDS = {
        'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai',
        'pune', 'kolkata', 'ahmedabad', 'jaipur', 'surat', 'lucknow', 'noida',
        'gurugram', 'gurgaon', 'rajkot', 'indore', 'bhopal', 'chandigarh',
        'remote', 'wfh', 'work-from-home', 'hybrid',
    }

    tokens = slug.lower().split('-')

    # Find first location token
    loc_idx = None
    for i, t in enumerate(tokens):
        if t in LOCATION_KEYWORDS:
            loc_idx = i
            break

    if loc_idx is not None:
        hints['location'] = _slug_to_title(tokens[loc_idx])
        # Everything before location is role + company — split roughly in half
        pre = tokens[:loc_idx]
        mid = len(pre) // 2
        if mid > 0:
            hints['role'] = _slug_to_title('-'.join(pre[:mid]))
            hints['company'] = _slug_to_title('-'.join(pre[mid:]))
        else:
            hints['role'] = _slug_to_title('-'.join(pre))
    else:
        # No location found — treat first 3 tokens as role, rest as company
        mid = min(3, len(tokens) // 2)
        hints['role'] = _slug_to_title('-'.join(tokens[:mid]))
        if tokens[mid:]:
            hints['company'] = _slug_to_title('-'.join(tokens[mid:]))


def _fetch_naukri_jd(url):
    """
    Extract JD from a Naukri job page.

    Naukri renders job data server-side as JSON inside a <script id="initial-data">
    or window.__INITIAL_STATE__ block. We parse that directly — no internal API needed.

    Falls back to CSS-selector extraction if the JSON approach fails.
    """
    import re as _re
    import json as _json

    # Strip query params for a cleaner fetch
    clean_url = url.split('?')[0]

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        session_r = requests.Session()
        # First visit homepage to get cookies (makes us look like a real browser)
        session_r.get('https://www.naukri.com/', headers=headers, timeout=10)
        resp = session_r.get(clean_url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            print(f"Naukri page returned {resp.status_code}")
            return None
    except Exception as e:
        print(f"Naukri page fetch error: {e}")
        return None

    # Force UTF-8 decoding
    resp.encoding = 'utf-8'
    html = resp.text

    try:
        from bs4 import BeautifulSoup as _BS
    except ImportError:
        return None

    soup = _BS(html, 'html.parser')

    # ── Strategy 1: Extract from embedded JSON in <script> tags ──────────────
    # Naukri injects job data as JSON in various script patterns
    json_patterns = [
        r'window\.__INITIAL_DATA__\s*=\s*({.+?});?\s*</script>',
        r'window\.__INITIAL_STATE__\s*=\s*({.+?});?\s*</script>',
        r'<script[^>]+id=["\']initial-data["\'][^>]*>({.+?})</script>',
        r'"jobDescription"\s*:\s*"((?:[^"\\]|\\.)+)"',
    ]

    for script_tag in soup.find_all('script'):
        script_text = script_tag.string or ''
        if not script_text or len(script_text) < 100:
            continue

        # Try to find JSON blob with job description
        for pattern in json_patterns[:-1]:  # skip the last regex-only pattern
            match = _re.search(pattern, script_text, _re.DOTALL)
            if match:
                try:
                    data = _json.loads(match.group(1))
                    jd = _deep_find_jd(data)
                    if jd and len(jd) > 200:
                        print(f"Naukri: extracted JD from embedded JSON ({len(jd)} chars)")
                        return jd
                except Exception:
                    pass

        # Direct jobDescription string search in script
        if 'jobDescription' in script_text:
            match = _re.search(r'"jobDescription"\s*:\s*"((?:[^"\\]|\\.)*)"', script_text)
            if match:
                try:
                    raw = match.group(1).encode('utf-8').decode('unicode_escape')
                    jd = _BS(raw, 'html.parser').get_text(separator='\n', strip=True)
                    if len(jd) > 200:
                        print(f"Naukri: extracted JD from script regex ({len(jd)} chars)")
                        return jd
                except Exception:
                    pass

    # ── Strategy 2: BeautifulSoup class-prefix matching ──────────────────────
    # Naukri uses CSS Modules with hashed suffixes e.g. styles_JDC__dang-inner-html__h0K4t
    # We match on the stable prefix part of the class name.
    # Priority: most specific inner content div first, then outer containers.
    PREFIX_TARGETS = [
        'styles_JDC__dang-inner-html',   # div.styles_JDC__dang-inner-html__h0K4t (actual content)
        'styles_job-desc-container',      # section.styles_job-desc-container__txpYf (outer wrapper)
        'styles_key-skill',               # div.styles_key-skill__GIPn_ (skills block)
        'styles_JDC',
        'styles_jdc',
    ]

    def find_by_class_prefix(s, prefix):
        for tag in s.find_all(True):
            classes = tag.get('class') or []
            if any(c.startswith(prefix) for c in classes):
                return tag
        return None

    for prefix in PREFIX_TARGETS:
        el = find_by_class_prefix(soup, prefix)
        if el:
            text = el.get_text(separator='\n', strip=True)
            if len(text) > 200:
                print(f"Naukri: extracted JD via class prefix '{prefix}' ({len(text)} chars)")
                return text

    # Also try wildcard CSS selectors as secondary fallback
    css_selectors = [
        'section[class*="job-desc-container"]',
        'div[class*="dang-inner-html"]',
        'div[class*="JDC"]',
        'div[class*="job-desc"]',
        'div#job_description',
    ]
    for sel in css_selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator='\n', strip=True)
            if len(text) > 200:
                print(f"Naukri: extracted JD via CSS selector '{sel}' ({len(text)} chars)")
                return text

    # ── Strategy 3: Find largest text block containing JD keywords ───────────
    jd_keywords = ['responsibilities', 'requirements', 'skills', 'experience',
                   'qualifications', 'you will', 'we are looking', 'role']
    best, best_score = None, 0
    for el in soup.find_all(['div', 'section', 'article']):
        text = el.get_text(separator=' ', strip=True)
        if len(text) < 300 or len(text) > 20000:
            continue
        hits = sum(1 for kw in jd_keywords if kw in text.lower())
        score = hits * 50 + len(text) * 0.005
        if score > best_score:
            best_score, best = score, el
    if best:
        text = best.get_text(separator='\n', strip=True)
        if len(text) > 200:
            print(f"Naukri: extracted JD via heuristic ({len(text)} chars)")
            return text

    print("Naukri: all strategies failed")
    return None


def _fetch_internshala_jd(url):
    """
    Extract full job/internship details from an Internshala listing page.

    Internshala is server-side rendered. The DOM structure (confirmed from DevTools):
      - div.internship_details          — outer wrapper
      - div.text-container              — about the job body text
      - div.text-container.who_can_apply — who can apply section
      - span.round_tabs                 — individual skill tags
      - h2/h3.section_heading           — section headings
      - div.text-container.about_company_text_container — company description

    We scrape all relevant sections and assemble them into clean structured text.
    """
    import re as _re
    try:
        from bs4 import BeautifulSoup as _BS
    except ImportError:
        return None

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://internshala.com/',
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            print(f"Internshala: page returned {resp.status_code}")
            return None
    except Exception as e:
        print(f"Internshala: fetch error: {e}")
        return None

    resp.encoding = 'utf-8'
    soup = _BS(resp.text, 'html.parser')

    sections = []

    # ── Job title ──────────────────────────────────────────────────────────────
    title_el = soup.select_one('h1.heading_title, h1.heading_2_4')
    if title_el:
        sections.append(f"Role: {title_el.get_text(strip=True)}")

    # ── Company name ──────────────────────────────────────────────────────────
    company_el = soup.select_one('a.link_display_like_text, div.company_name a')
    if company_el:
        sections.append(f"Company: {company_el.get_text(strip=True)}")

    # ── Meta info (stipend, duration, location, deadline) ─────────────────────
    meta_items = []
    for item in soup.select('div.item_body, div#stipend_container span, div.other_detail_item'):
        t = item.get_text(separator=' ', strip=True)
        if t:
            meta_items.append(t)
    if meta_items:
        sections.append("Details: " + " | ".join(dict.fromkeys(meta_items)))  # dedup

    # ── Skills required ───────────────────────────────────────────────────────
    skill_tags = soup.select('span.round_tabs')
    if skill_tags:
        skills = [s.get_text(strip=True) for s in skill_tags if s.get_text(strip=True)]
        sections.append("Skills Required: " + ", ".join(skills))

    # ── Main section headings + their body text ───────────────────────────────
    # Walk through the internship_details container section by section
    detail_container = soup.select_one('div.internship_details, div#details_container')
    if detail_container:
        current_heading = None
        current_body = []

        for el in detail_container.find_all(['h2', 'h3', 'p', 'ul', 'div'], recursive=True):
            tag = el.name
            classes = el.get('class') or []

            # Section headings
            if tag in ('h2', 'h3') and any('section_heading' in c for c in classes):
                if current_heading and current_body:
                    combined = '\n'.join(current_body).strip()
                    if len(combined) > 20:
                        sections.append(f"\n{current_heading}:\n{combined}")
                current_heading = el.get_text(strip=True)
                current_body = []
                continue

            # text-container divs hold the actual paragraph content
            if tag == 'div' and any('text-container' in c for c in classes):
                text = el.get_text(separator='\n', strip=True)
                if text and len(text) > 10:
                    current_body.append(text)

            # Paragraphs inside those containers (already captured via parent, skip)

        # Flush last section
        if current_heading and current_body:
            combined = '\n'.join(current_body).strip()
            if len(combined) > 20:
                sections.append(f"\n{current_heading}:\n{combined}")

    # ── Fallback: grab entire internship_details block if sections are thin ───
    if len(sections) < 3:
        fallback = soup.select_one(
            'div.internship_details, div.detail_view, div#about_internship, '
            'div[class*="internship_detail"]'
        )
        if fallback:
            text = fallback.get_text(separator='\n', strip=True)
            if len(text) > 200:
                print(f"Internshala: used fallback selector ({len(text)} chars)")
                return {'jd': text}

    if not sections:
        print("Internshala: all strategies failed")
        return None

    jd_text = '\n'.join(sections)
    print(f"Internshala: extracted JD ({len(jd_text)} chars, {len(sections)} sections)")
    return {'jd': jd_text}


def _fetch_hirist_jd(url):
    """
    Extract job description from a Hirist.tech listing.

    Hirist is a Next.js app but uses SSG — job content is baked into the
    initial HTML for SEO. Confirmed from DevTools:

      <div data-testid="job-description-container">   ← primary target
        <div class="details-container ...">
          <span class="MuiTypography-body1 ...">
            <p><b>Description :</b></p>
            <p>actual jd text...</p>
          </span>
        </div>
      </div>

    The data-testid attribute is stable by design (devs add these for testing).
    We also check JSON-LD as a bonus since Next.js sites often embed it for SEO.
    """
    import json as _json
    try:
        from bs4 import BeautifulSoup as _BS
    except ImportError:
        return None

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.hirist.tech/',
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            print(f"Hirist: page returned {resp.status_code}")
            return None
    except Exception as e:
        print(f"Hirist: fetch error: {e}")
        return None

    resp.encoding = 'utf-8'
    soup = _BS(resp.text, 'html.parser')

    # ── Strategy 1: data-testid — stable, intentional, confirmed in DevTools ──
    selectors = [
        'div[data-testid="job-description-container"]',
        'div[data-testid="jobDescriptionContainer"]',
        'div.details-container',                        # inner wrapper seen in DevTools
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            # Remove any nested nav/button noise
            for noise in el.select('button, nav, [aria-hidden="true"]'):
                noise.decompose()
            text = el.get_text(separator='\n', strip=True)
            if len(text) > 100:
                print(f"Hirist: extracted JD via '{sel}' ({len(text)} chars)")
                return text

    # ── Strategy 2: JSON-LD structured data (Next.js sites embed for SEO) ─────
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = _json.loads(script.string or '')
            if isinstance(data, dict) and data.get('@type') in ('JobPosting', 'jobPosting'):
                desc = data.get('description', '')
                if desc and len(desc) > 100:
                    text = _BS(desc, 'html.parser').get_text(separator='\n', strip=True)
                    print(f"Hirist: extracted JD via JSON-LD ({len(text)} chars)")
                    return text
        except Exception:
            pass

    # ── Strategy 3: __NEXT_DATA__ JSON blob ───────────────────────────────────
    # Next.js SSG pages embed full page props in a <script id="__NEXT_DATA__"> tag
    next_data_tag = soup.find('script', id='__NEXT_DATA__')
    if next_data_tag:
        try:
            data = _json.loads(next_data_tag.string or '')
            jd = _deep_find_jd(data)
            if jd and len(jd) > 100:
                print(f"Hirist: extracted JD via __NEXT_DATA__ ({len(jd)} chars)")
                return jd
        except Exception:
            pass

    print("Hirist: all strategies failed")
    return None


def _deep_find_jd(obj, depth=0):
    """Recursively search a dict/list for jobDescription field and return cleaned text."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        for key in ('jobDescription', 'job_description', 'description', 'jobDesc'):
            val = obj.get(key)
            if isinstance(val, str) and len(val) > 200:
                try:
                    from bs4 import BeautifulSoup as _BS
                    return _BS(val, 'html.parser').get_text(separator='\n', strip=True)
                except Exception:
                    return val
        for v in obj.values():
            result = _deep_find_jd(v, depth + 1)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _deep_find_jd(item, depth + 1)
            if result:
                return result
    return None


@app.route('/api/generate-resume', methods=['POST'])
def generate_resume():
    data = request.json
    job_description = data.get('job_description', '')
    sid = session.get('student_id')
    
    if not job_description:
        return jsonify({'error': 'Job description required'}), 400

    # Credit check
    if sid and sid != 'demo':
        ok, balance, cost = _spend_credits(sid, 'generate_resume')
        if not ok:
            return jsonify({
                'error': 'insufficient_credits',
                'message': f'Generating a resume costs {cost} credits but you only have {balance}. Refer friends to earn more!',
                'balance': balance, 'cost': cost
            }), 402
    
    # Try to get from DB
    if sid and sid != 'demo':
        student = Student.query.get(sid)
        if student:
            profile_data = json.loads(student.profile_data or '{}')
    
    if not profile_data:
        return jsonify({'error': 'No profile data found. Please complete the interview first.'}), 400
    
    system_prompt = get_resume_system_prompt(profile_data, job_description)
    result = call_llm(
        [{"role": "user", "content": "Generate the resume JSON now."}],
        system_prompt
    )
    
    # Parse JSON
    try:
        clean = result.strip()
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                if '{' in part:
                    clean = part.lstrip("json").strip()
                    break
        resume_json = json.loads(clean)
    except:
        resume_json = {"error": "Could not parse resume", "raw": result}
    
    # Save to DB
    if sid and sid != 'demo':
        resume = Resume(
            student_id=sid,
            job_description=job_description,
            resume_data=json.dumps(resume_json)
        )
        db.session.add(resume)
        db.session.flush()  # get resume.id before commit
        # Save initial version snapshot
        ver = ResumeVersion(
            resume_id=resume.id,
            version_number=1,
            resume_data=json.dumps(resume_json),
            label='AI Generated'
        )
        db.session.add(ver)
        db.session.commit()
        resume_json['resume_id'] = resume.id
    
    return jsonify(resume_json)

@app.route('/api/profile', methods=['GET', 'PUT'])
def profile():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Not found'}), 404
    
    if request.method == 'PUT':
        data = request.json
        existing = json.loads(student.profile_data or '{}')
        existing.update(data)
        student.profile_data = json.dumps(existing)
        db.session.commit()
    
    return jsonify({
        'profile': json.loads(student.profile_data or '{}'),
        'name': student.name,
        'email': student.email
    })

@app.route('/api/resumes', methods=['GET'])
def get_resumes():
    sid = session.get('student_id')
    if not sid:
        return jsonify([])
    resumes = Resume.query.filter_by(student_id=sid).order_by(Resume.created_at.desc()).all()
    return jsonify([{
        'id': r.id,
        'created_at': r.created_at.isoformat(),
        'job_description': r.job_description[:100] + '...' if r.job_description and len(r.job_description) > 100 else r.job_description
    } for r in resumes])

@app.route('/api/resumes/<int:resume_id>', methods=['GET'])
def get_resume(resume_id):
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    resume = Resume.query.filter_by(id=resume_id, student_id=sid).first()
    if not resume:
        return jsonify({'error': 'Resume not found'}), 404
    data = json.loads(resume.resume_data or '{}')
    data['resume_id'] = resume.id
    data['created_at'] = resume.created_at.isoformat()
    data['edited_html'] = resume.edited_html  # may be None
    return jsonify(data)

# ── Resume inline edit & save ─────────────────────────────────────────────────

@app.route('/api/resumes/<int:resume_id>/save-edit', methods=['POST'])
def save_resume_edit(resume_id):
    """Save inline-edited HTML and JSON for a resume, creating a new version."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    resume = Resume.query.filter_by(id=resume_id, student_id=sid).first()
    if not resume:
        return jsonify({'error': 'Resume not found'}), 404

    data = request.json
    edited_html = data.get('edited_html')
    resume_json_str = data.get('resume_data')  # optional updated JSON
    label = data.get('label', 'Manually edited')

    # Snapshot current state as a new version
    next_ver = len(resume.versions) + 1
    ver = ResumeVersion(
        resume_id=resume.id,
        version_number=next_ver,
        resume_data=resume.resume_data,
        edited_html=resume.edited_html,
        label=resume.versions[-1].label if resume.versions else 'AI Generated'
    )
    db.session.add(ver)

    # Update resume with new edit
    if edited_html:
        resume.edited_html = edited_html
    if resume_json_str:
        resume.resume_data = resume_json_str if isinstance(resume_json_str, str) else json.dumps(resume_json_str)
    resume.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({'ok': True, 'version': next_ver + 1})


# ── Resume versioning & diff ──────────────────────────────────────────────────

@app.route('/api/resumes/<int:resume_id>/versions', methods=['GET'])
def get_resume_versions(resume_id):
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    resume = Resume.query.filter_by(id=resume_id, student_id=sid).first()
    if not resume:
        return jsonify({'error': 'Resume not found'}), 404

    vers = [{
        'id': v.id,
        'version_number': v.version_number,
        'label': v.label,
        'created_at': v.created_at.isoformat()
    } for v in resume.versions]

    # Include current as "latest"
    vers.append({
        'id': None,
        'version_number': len(vers) + 1,
        'label': 'Current',
        'created_at': resume.updated_at.isoformat() if resume.updated_at else resume.created_at.isoformat()
    })
    return jsonify(vers)


@app.route('/api/resumes/<int:resume_id>/diff', methods=['GET'])
def get_resume_diff(resume_id):
    """Return unified diff between two versions (ver_a, ver_b query params). 
    Use ver_id=None to mean 'current'."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    resume = Resume.query.filter_by(id=resume_id, student_id=sid).first()
    if not resume:
        return jsonify({'error': 'Resume not found'}), 404

    ver_a_id = request.args.get('ver_a')
    ver_b_id = request.args.get('ver_b')

    def get_text(ver_id):
        if ver_id is None or ver_id == 'current':
            raw = resume.edited_html or json.dumps(json.loads(resume.resume_data or '{}'), indent=2)
        else:
            v = ResumeVersion.query.get(int(ver_id))
            if not v or v.resume_id != resume.id:
                return ''
            raw = v.edited_html or json.dumps(json.loads(v.resume_data or '{}'), indent=2)
        # Strip HTML tags for readable diff
        return re.sub(r'<[^>]+>', '', raw)

    text_a = get_text(ver_a_id).splitlines(keepends=True)
    text_b = get_text(ver_b_id).splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(text_a, text_b, fromfile=f'Version {ver_a_id}', tofile=f'Version {ver_b_id}', lineterm=''))

    # Build HTML diff for display
    html_diff = difflib.HtmlDiff(wrapcolumn=80).make_table(
        text_a, text_b,
        fromdesc=f'Version {ver_a_id}',
        todesc=f'Version {ver_b_id}',
        context=True, numlines=3
    )

    return jsonify({
        'unified': ''.join(diff_lines),
        'html_diff': html_diff,
        'changes': len([l for l in diff_lines if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))])
    })


# ── Voice / Whisper transcription ─────────────────────────────────────────────

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Groq Whisper (free, fast). Falls back to error if unavailable."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if not GROQ_API_KEY:
        return jsonify({'error': 'GROQ_API_KEY not configured. Whisper transcription requires Groq.'}), 503

    try:
        # Groq supports whisper-large-v3 via their API
        files = {
            'file': (audio_file.filename or 'audio.webm', audio_file.stream, audio_file.content_type or 'audio/webm'),
            'model': (None, 'whisper-large-v3'),
            'language': (None, 'en'),
            'response_format': (None, 'json'),
        }
        resp = requests.post(
            'https://api.groq.com/openai/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
            files=files,
            timeout=60
        )
        if resp.status_code == 200:
            text = resp.json().get('text', '').strip()
            return jsonify({'text': text})
        else:
            return jsonify({'error': f'Transcription failed: {resp.text[:200]}'}), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── GitHub profile scraping (via public API, no auth needed for basic use) ────

@app.route('/api/github-profile', methods=['POST'])
def github_profile():
    """Fetch GitHub profile + top repos using GitHub public API."""
    username = (request.json or {}).get('username', '').strip().lstrip('@')
    if not username:
        return jsonify({'error': 'GitHub username required'}), 400

    headers = {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Kairo-App'}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    try:
        # User profile
        user_resp = requests.get(f'https://api.github.com/users/{username}', headers=headers, timeout=10)
        if user_resp.status_code == 404:
            return jsonify({'error': f'GitHub user "{username}" not found'}), 404
        if user_resp.status_code == 403:
            return jsonify({'error': 'GitHub API rate limit hit. Add GITHUB_TOKEN to env for higher limits.'}), 429
        if user_resp.status_code != 200:
            return jsonify({'error': 'Could not fetch GitHub profile'}), user_resp.status_code

        user_data = user_resp.json()

        # Repos — sort by stars, pick top 8
        repos_resp = requests.get(
            f'https://api.github.com/users/{username}/repos',
            headers=headers,
            params={'per_page': 100, 'type': 'owner', 'sort': 'updated'},
            timeout=10
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []

        # Filter out forks, sort by stars
        owned = [r for r in repos if not r.get('fork', False)]
        owned.sort(key=lambda r: (r.get('stargazers_count', 0), r.get('watchers_count', 0)), reverse=True)
        top_repos = owned[:8]

        # For each top repo, try to get README for richer description
        enriched_repos = []
        for repo in top_repos[:5]:  # only top 5 to stay within rate limits
            repo_info = {
                'name': repo.get('name'),
                'description': repo.get('description') or '',
                'language': repo.get('language'),
                'stars': repo.get('stargazers_count', 0),
                'url': repo.get('html_url'),
                'topics': repo.get('topics', []),
                'updated_at': repo.get('updated_at', ''),
            }
            # Try README
            try:
                readme_resp = requests.get(
                    f'https://api.github.com/repos/{username}/{repo["name"]}/readme',
                    headers={**headers, 'Accept': 'application/vnd.github.v3.raw'},
                    timeout=5
                )
                if readme_resp.status_code == 200:
                    readme_text = readme_resp.text[:1500]
                    repo_info['readme_snippet'] = readme_text
            except Exception:
                pass
            enriched_repos.append(repo_info)

        # Add remaining repos without README
        for repo in top_repos[5:]:
            enriched_repos.append({
                'name': repo.get('name'),
                'description': repo.get('description') or '',
                'language': repo.get('language'),
                'stars': repo.get('stargazers_count', 0),
                'url': repo.get('html_url'),
                'topics': repo.get('topics', []),
            })

        # Use LLM to extract resume-worthy projects from GitHub data
        github_summary = call_llm(
            [{'role': 'user', 'content': f'''GitHub username: {username}
Bio: {user_data.get("bio", "")}
Public repos: {user_data.get("public_repos", 0)}

Top repositories:
{json.dumps(enriched_repos, indent=2)}

Extract the 3-5 best projects for a student resume. For each project provide:
- name, description (2 sentences, impact-focused), tech stack (array), and any notable metrics (stars, etc.)

Return ONLY valid JSON array: [{{"name": "", "description": "", "tech": [], "impact": "", "url": ""}}]'''}],
            'You are a technical recruiter extracting resume-worthy projects from GitHub. Return only valid JSON.'
        )

        try:
            clean = github_summary.strip()
            if '```' in clean:
                parts = clean.split('```')
                for p in parts:
                    if '[' in p:
                        clean = p.lstrip('json').strip()
                        break
            projects = json.loads(clean)
        except Exception:
            projects = []

        return jsonify({
            'username': username,
            'name': user_data.get('display_login') or user_data.get('login'),
            'bio': user_data.get('bio'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'profile_url': user_data.get('html_url'),
            'avatar_url': user_data.get('avatar_url'),
            'top_repos': enriched_repos,
            'extracted_projects': projects
        })

    except requests.exceptions.Timeout:
        return jsonify({'error': 'GitHub API timed out. Please try again.'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── LinkedIn hints (no scraping — guide-based approach) ──────────────────────

@app.route('/api/linkedin-hints', methods=['POST'])
def linkedin_hints():
    """Parse a user-pasted LinkedIn 'About' / experience text and extract profile data."""
    data = request.json or {}
    linkedin_text = data.get('text', '').strip()
    linkedin_url = data.get('url', '').strip()

    if not linkedin_text:
        return jsonify({'error': 'Please paste your LinkedIn About section or experience text'}), 400

    extracted = call_llm(
        [{'role': 'user', 'content': f'LinkedIn profile text:\n{linkedin_text[:4000]}'}],
        '''Extract structured career profile data from this LinkedIn profile text.
Return ONLY valid JSON:
{
  "summary": "",
  "experience": [{"role": "", "company": "", "duration": "", "points": []}],
  "skills": {"technical": [], "soft": []},
  "education": [{"degree": "", "institution": "", "year": ""}],
  "certifications": [],
  "achievements": []
}'''
    )

    try:
        clean = extracted.strip()
        if '```' in clean:
            for p in clean.split('```'):
                if '{' in p:
                    clean = p.lstrip('json').strip()
                    break
        result = json.loads(clean)
    except Exception:
        result = {'raw': extracted}

    return jsonify({'extracted': result, 'linkedin_url': linkedin_url})


# ── Self-intro video script generation ───────────────────────────────────────

@app.route('/api/generate-intro-script', methods=['POST'])
def generate_intro_script():
    """Generate a personalized self-introduction video script."""
    sid = session.get('student_id')
    data = request.json or {}

    # Credit check
    if sid and sid != 'demo':
        ok, balance, cost = _spend_credits(sid, 'intro_script')
        if not ok:
            return jsonify({
                'error': 'insufficient_credits',
                'message': f'Generating an intro script costs {cost} credits but you only have {balance}.',
                'balance': balance, 'cost': cost
            }), 402

    profile_data = data.get('profile_data', {})
    job_role = data.get('job_role', '').strip()
    duration_seconds = int(data.get('duration_seconds', 60))
    tone = data.get('tone', 'professional')  # professional | casual | energetic

    # Get from DB if logged in
    if sid and sid != 'demo':
        student = Student.query.get(sid)
        if student:
            profile_data = json.loads(student.profile_data or '{}')

    if not profile_data:
        return jsonify({'error': 'No profile data found. Complete your interview first.'}), 400

    word_target = int(duration_seconds * 2.2)  # ~130 wpm, 2.2 words/sec

    system = f'''You are a professional speaking coach helping a student create a compelling self-introduction video script.
Tone: {tone}. Target duration: {duration_seconds} seconds (~{word_target} words).
Generate a natural, spoken script — not a formal essay. Include pauses [...] where appropriate.'''

    prompt = f'''Student profile:
{json.dumps(profile_data, indent=2)}

Target role/company: {job_role or 'general tech roles'}

Generate a complete self-introduction video script with these sections:
1. Hook (attention-grabbing opener)
2. Who I am (name, college, branch)
3. What I do / key skills
4. Best project or achievement (1 highlight)
5. Why this role/company
6. Call to action / close

Also provide:
- Delivery tips (3 bullet points)
- Key phrases to emphasize
- Things to avoid

Return as JSON:
{{
  "script": "full spoken script text",
  "sections": [{{"title": "", "text": "", "duration_estimate_sec": 0}}],
  "word_count": 0,
  "estimated_duration_sec": 0,
  "delivery_tips": [],
  "key_phrases": [],
  "avoid": []
}}'''

    result = call_llm([{'role': 'user', 'content': prompt}], system)

    try:
        clean = result.strip()
        if '```' in clean:
            for p in clean.split('```'):
                if '{' in p:
                    clean = p.lstrip('json').strip()
                    break
        script_data = json.loads(clean)
    except Exception:
        script_data = {'script': result, 'sections': [], 'delivery_tips': [], 'key_phrases': [], 'avoid': []}

    return jsonify(script_data)

@app.route('/api/presence-audit', methods=['POST'])
def presence_audit():
    """
    AI-powered LinkedIn & GitHub presence audit.
    Uses the same Groq (Llama 3.3 70B) backend as everything else.
    No credits charged — this is guidance, not a generation product.
    """
    data = request.json or {}
 
    github_username = data.get('github_username', '').strip()
    linkedin_url    = data.get('linkedin_url', '').strip()
    target_role     = data.get('target_role', '').strip()
    company_type    = data.get('company_type', 'any').strip()
    context         = data.get('context', '').strip()
 
    if not github_username and not linkedin_url:
        return jsonify({'error': 'Provide at least a GitHub username or LinkedIn URL.'}), 400
 
    # Pull the student's Kairo profile for extra context if logged in
    sid = session.get('student_id')
    profile_summary = ''
    if sid and sid != 'demo':
        student = Student.query.get(sid)
        if student:
            p = json.loads(student.profile_data or '{}')
            parts = []
            if p.get('name'):    parts.append(f"Name: {p['name']}")
            if p.get('branch'):  parts.append(f"Branch: {p['branch']}")
            if p.get('college'): parts.append(f"College: {p['college']}")
            if p.get('skills'):
                skills = p['skills']
                if isinstance(skills, list): parts.append(f"Skills: {', '.join(skills[:12])}")
                elif isinstance(skills, dict):
                    flat = []
                    for v in skills.values():
                        if isinstance(v, list): flat.extend(v)
                    parts.append(f"Skills: {', '.join(flat[:12])}")
            if p.get('projects'):
                proj_names = [pr.get('name','') for pr in (p['projects'] if isinstance(p['projects'], list) else []) if pr.get('name')]
                if proj_names: parts.append(f"Projects: {', '.join(proj_names[:5])}")
            if p.get('experience'):
                exp = p['experience']
                if isinstance(exp, list) and exp:
                    parts.append(f"Experience: {exp[0].get('company', '')} — {exp[0].get('role', '')}")
            profile_summary = '\n'.join(parts)
 
    system_prompt = (
        "You are a senior tech recruiter and career coach with deep knowledge of how "
        "Indian tech students are evaluated for internships and jobs at startups, "
        "product companies, and FAANG. You give honest, specific, actionable advice. "
        "Return ONLY valid JSON — no markdown, no explanation, no code fences."
    )
 
    prompt = f"""Analyse the online presence of a student applying for tech roles and produce a structured audit report.
 
STUDENT INPUT:
- GitHub username: {github_username or '(not provided)'}
- LinkedIn URL: {linkedin_url or '(not provided)'}
- Target role: {target_role or 'Software engineering / tech (unspecified)'}
- Target company type: {company_type}
- Student's additional context: {context or '(none provided)'}
 
KAIRO PROFILE SNAPSHOT:
{profile_summary or '(not available)'}
 
IMPORTANT: Since you cannot access live URLs, base your audit on:
1. What a recruiter would typically look for given the role and company type
2. The most common mistakes students make in that context
3. Any specific details the student has shared in their context above
4. Their Kairo profile data to personalise advice about showcasing their actual skills/projects
 
Be specific — reference the target role ({target_role or 'tech'}) and company type ({company_type}) in your tips.
Scores should be honest; most students score 3–6 unless they have strong evidence of a great profile.
 
Return this exact JSON schema (no extra keys):
{{
  "scores": {{
    "github": <integer 1-10>,
    "github_verdict": "<one short phrase, e.g. 'Needs work' / 'Solid foundation' / 'Recruiter-ready'>",
    "linkedin": <integer 1-10>,
    "linkedin_verdict": "<one short phrase>",
    "overall": <integer 1-10>,
    "overall_verdict": "<one short phrase>"
  }},
  "github_tips": [
    {{
      "title": "<short title>",
      "priority": "High | Medium | Low",
      "description": "<2-3 sentences of specific advice>",
      "action": "<one concrete next step the student can do today>"
    }}
  ],
  "linkedin_tips": [
    {{
      "title": "<short title>",
      "priority": "High | Medium | Low",
      "description": "<2-3 sentences of specific advice>",
      "action": "<one concrete next step>"
    }}
  ],
  "consistency_checks": [
    {{
      "label": "<what is being checked, e.g. 'Profile photo present'>",
      "status": "good | warn | bad",
      "detail": "<short explanation or recommendation>"
    }}
  ],
  "weekly_plan": [
    {{
      "week": <1-4>,
      "tasks": ["<task 1>", "<task 2>", "<task 3>"]
    }}
  ]
}}
 
Rules:
- github_tips: 4-6 items
- linkedin_tips: 4-6 items
- consistency_checks: 6-8 items covering both platforms (photo, headline, pinned repos, README, about section, skills section, contact info, posting cadence)
- weekly_plan: exactly 4 weeks, ordered from highest-impact to maintenance
- All advice must be specific to the target role and company type
- Do not invent facts about the student's actual profile; base scores on what they've shared"""
 
    raw = call_llm([{'role': 'user', 'content': prompt}], system_prompt, max_tokens=3000)
 
    # Parse JSON — robust extraction: find the outermost {...} block
    try:
        clean = raw.strip()
        # Strip markdown fences if present
        if '```' in clean:
            for part in clean.split('```'):
                if '{' in part:
                    clean = part.lstrip('json').strip()
                    break
        # Extract the outermost JSON object using brace matching
        start = clean.find('{')
        if start != -1:
            depth, end = 0, -1
            for idx, ch in enumerate(clean[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end != -1:
                clean = clean[start:end]
        result = json.loads(clean)
    except Exception:
        # Graceful fallback so the frontend always gets a valid response shape
        result = {
            'scores': {
                'github': 5, 'github_verdict': 'Unable to score',
                'linkedin': 5, 'linkedin_verdict': 'Unable to score',
                'overall': 5, 'overall_verdict': 'Partial analysis'
            },
            'github_tips': [{'title': 'Analysis incomplete', 'priority': 'Medium',
                             'description': 'The AI could not parse a full response. Try again.',
                             'action': 'Click Run Audit again.'}],
            'linkedin_tips': [],
            'consistency_checks': [],
            'weekly_plan': [],
            '_raw': raw[:500]  # include truncated raw for debugging
        }
 
    return jsonify(result)


@app.route('/api/presence-audit/deep-dive', methods=['POST'])
def presence_audit_deep_dive():
    """
    Returns an in-depth AI explanation for a single audit tip.
    Read-only — no credits charged, no user input beyond the prompt.
    """
    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    system_prompt = (
        "You are a senior tech recruiter and career coach. "
        "Give practical, specific, actionable advice. "
        "Use plain text with ### headings and - bullet points. "
        "No markdown code fences. Be concise but thorough."
    )

    raw = call_llm(
        [{'role': 'user', 'content': prompt}],
        system_prompt,
        max_tokens=600
    )

    return jsonify({'content': raw})


@app.route('/api/conversations/active')
def active_conversation():
    """Return the most recent conversation with its messages for session restore."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'conversation': None})

    conv = Conversation.query.filter_by(
        student_id=sid, topic='profile_building'
    ).order_by(Conversation.created_at.desc()).first()

    if not conv:
        return jsonify({'conversation': None})

    # Try Redis first, fall back to Postgres
    messages = cache_get_messages(conv.id)
    if messages is None:
        messages = json.loads(conv.messages or '[]')
        if messages:
            cache_set_messages(conv.id, messages)  # warm the cache

    if not messages:
        return jsonify({'conversation': None})

    return jsonify({
        'conversation': {
            'id': conv.id,
            'messages': messages,
            'updated_at': (conv.updated_at or conv.created_at).isoformat(),
            'message_count': len(messages)
        }
    })

@app.route('/api/conversations/new', methods=['POST'])
def new_conversation():
    """Start a fresh conversation, archiving the current one."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401

    # Archive old conversations by changing their topic
    old_convs = Conversation.query.filter_by(student_id=sid, topic='profile_building').all()
    for c in old_convs:
        c.topic = 'archived'
    db.session.commit()

    conv = Conversation(student_id=sid, topic='profile_building')
    db.session.add(conv)
    db.session.commit()

    return jsonify({'conversation_id': conv.id})

@app.route('/api/llm-status')
def llm_status():
    return jsonify(get_llm_status())

## ── Mock Interview ────────────────────────────────────────────────────────────

def get_mock_interview_questions_prompt(profile_data, job_title, job_description):
    profile_str = json.dumps(profile_data, indent=2) if isinstance(profile_data, dict) else profile_data
    return f"""You are an expert technical interviewer. Generate exactly 8 interview questions for a candidate.

Job Title: {job_title}
Job Description: {job_description[:1000] if job_description else 'General role'}

Candidate Profile:
{profile_str[:2000]}

Generate a mix of:
- 2 behavioural questions (past experience, teamwork, conflict)
- 2 technical questions (specific to the role/skills)
- 2 profile-based questions (based on candidate's projects or skills listed)
- 1 situational question (hypothetical scenario)
- 1 motivational/culture-fit question

Return ONLY a valid JSON array, no markdown, no preamble:
[
  {{"id": 1, "type": "behavioural", "question": "...", "what_to_look_for": "..."}},
  {{"id": 2, "type": "technical", "question": "...", "what_to_look_for": "..."}},
  ...
]"""

def get_evaluation_prompt(questions, transcript, job_title):
    # Build a map of question_id -> answer from transcript
    answer_map = {str(item.get('question_id','')): item.get('answer','') for item in transcript}
    
    # Build full QA list — every question gets a row, unanswered = blank
    qa_rows = []
    for q in questions:
        qid = str(q.get('id', ''))
        answer = answer_map.get(qid, '').strip()
        qa_rows.append({
            'id': qid,
            'type': q.get('type',''),
            'question': q.get('question',''),
            'answer': answer if answer else '[No answer given]'
        })
    
    answered_count = sum(1 for r in qa_rows if r['answer'] != '[No answer given]')
    total = len(qa_rows)
    
    qa_text = ""
    for r in qa_rows:
        qa_text += f"\nQ{r['id']} [{r['type']}]: {r['question']}\nCandidate answer: {r['answer']}\n"
    
    return f"""You are a strict but fair interview coach evaluating a mock interview for a {job_title} position.

The candidate answered {answered_count} out of {total} questions.

Full interview — questions and answers:
{qa_text[:4000]}

STRICT SCORING RULES:
- An unanswered question ([No answer given]) scores 0/10, no exceptions.
- A one-word or very short answer scores at most 2/10.
- A decent but vague answer scores 4-5/10.
- A good structured answer with examples scores 7-8/10.
- An excellent answer with specific details, metrics, or examples scores 9-10/10.
- If the candidate answered fewer than half the questions, overall_score must be below 30.
- If the candidate answered zero questions, overall_score must be 0.
- Do NOT give credit for effort alone. Score only the quality of actual answers given.
- overall_score is the weighted average of question scores scaled to 100, rounded to nearest integer.

Return ONLY a valid JSON object, no markdown fences, no extra text:
{{
  "overall_score": <integer 0-100, strictly calculated from question scores>,
  "overall_summary": "<2-3 honest sentences about the interview performance>",
  "strengths": ["<only list genuine strengths shown in answers, or 'None demonstrated' if applicable>"],
  "areas_for_improvement": ["<specific, actionable items based on what was weak or missing>"],
  "question_scores": [
    {{
      "question_id": "<id>",
      "question": "<question text>",
      "answer": "<answer given or blank>",
      "score": <0-10>,
      "feedback": "<specific feedback on this answer>",
      "ideal_answer_hint": "<what a strong answer would have included>"
    }}
  ],
  "tips": ["<3 specific tips based on actual weaknesses seen>"],
  "readiness_level": "<one of: Not Ready / Needs Work / Almost There / Interview Ready>"
}}"""


@app.route('/api/mock-interview/start', methods=['POST'])
def start_mock_interview():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.get_json()
    job_title = data.get('job_title', 'Software Engineer')
    job_description = data.get('job_description', '')

    # Credit check
    ok, balance, cost = _spend_credits(sid, 'mock_interview')
    if not ok:
        return jsonify({
            'error': 'insufficient_credits',
            'message': f'Starting a mock interview costs {cost} credits but you only have {balance}. Refer friends to earn more!',
            'balance': balance, 'cost': cost
        }), 402

    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    
    try:
        profile_data = json.loads(student.profile_data or '{}')
    except:
        profile_data = {}
    
    # Generate questions via LLM
    prompt = get_mock_interview_questions_prompt(profile_data, job_title, job_description)
    raw = call_llm([], prompt, max_tokens=2000)
    
    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0]
        questions = json.loads(cleaned)
    except Exception as e:
        # Fallback generic questions
        questions = [
            {"id": 1, "type": "behavioural", "question": "Tell me about yourself and why you're interested in this role.", "what_to_look_for": "Concise summary, enthusiasm"},
            {"id": 2, "type": "behavioural", "question": "Describe a challenging project you worked on and how you overcame obstacles.", "what_to_look_for": "Problem-solving, resilience"},
            {"id": 3, "type": "technical", "question": f"What technical skills are most relevant for a {job_title} role?", "what_to_look_for": "Domain knowledge"},
            {"id": 4, "type": "technical", "question": "Walk me through how you would approach debugging a complex issue in a production system.", "what_to_look_for": "Systematic thinking"},
            {"id": 5, "type": "profile", "question": "Tell me about your most impactful project and what you learned from it.", "what_to_look_for": "Depth, learnings"},
            {"id": 6, "type": "profile", "question": "How have your past experiences prepared you for this role?", "what_to_look_for": "Relevance, growth"},
            {"id": 7, "type": "situational", "question": "If you disagreed with your manager's technical decision, how would you handle it?", "what_to_look_for": "Communication, professionalism"},
            {"id": 8, "type": "motivational", "question": "Where do you see yourself in 3 years, and how does this role fit into that?", "what_to_look_for": "Ambition, alignment"},
        ]
    
    interview = MockInterview(
        student_id=sid,
        job_title=job_title,
        job_description=job_description,
        status='in_progress',
        questions=json.dumps(questions),
        transcript='[]'
    )
    db.session.add(interview)
    db.session.commit()
    
    return jsonify({'interview_id': interview.id, 'questions': questions})


@app.route('/api/mock-interview/<int:interview_id>/submit-answer', methods=['POST'])
def submit_interview_answer(interview_id):
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not authenticated'}), 401
    
    interview = MockInterview.query.filter_by(id=interview_id, student_id=sid).first()
    if not interview:
        return jsonify({'error': 'Interview not found'}), 404
    
    data = request.get_json()
    question_id = data.get('question_id')
    question_text = data.get('question')
    answer = data.get('answer', '')
    
    transcript = json.loads(interview.transcript or '[]')
    # Upsert — update existing entry if question_id already present, else append
    existing = next((t for t in transcript if str(t.get('question_id','')) == str(question_id)), None)
    if existing:
        existing['answer'] = answer
        existing['updated_at'] = datetime.utcnow().isoformat()
    else:
        transcript.append({
            'question_id': question_id,
            'question': question_text,
            'answer': answer,
            'timestamp': datetime.utcnow().isoformat()
        })
    interview.transcript = json.dumps(transcript)
    db.session.commit()
    
    return jsonify({'status': 'saved'})


@app.route('/api/mock-interview/<int:interview_id>/complete', methods=['POST'])
def complete_mock_interview(interview_id):
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not authenticated'}), 401
    
    interview = MockInterview.query.filter_by(id=interview_id, student_id=sid).first()
    if not interview:
        return jsonify({'error': 'Interview not found'}), 404
    
    transcript = json.loads(interview.transcript or '[]')
    questions = json.loads(interview.questions or '[]')
    
    # Generate evaluation
    eval_prompt = get_evaluation_prompt(questions, transcript, interview.job_title)
    raw = call_llm([], eval_prompt, max_tokens=3000)
    
    try:
        cleaned = raw.strip() if raw else ''
        # Strip markdown fences if present
        if cleaned.startswith('```'):
            cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        # Find the outermost JSON object in case there's leading text
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
        report = json.loads(cleaned)
        # Validate required fields exist
        if 'overall_score' not in report:
            raise ValueError("Missing overall_score")
    except Exception as e:
        print(f"Evaluation parse error: {e}\nRaw LLM output: {raw[:500] if raw else 'None'}")
        # Build an honest fallback based on actual transcript
        transcript_len = len(transcript)
        questions_len = len(questions)
        answered = sum(1 for t in transcript if t.get('answer','').strip())
        score = max(0, int((answered / max(questions_len, 1)) * 40))  # max 40 for just answering
        report = {
            "overall_score": score,
            "overall_summary": f"The candidate answered {answered} of {questions_len} questions. Automated scoring unavailable — score reflects participation only.",
            "strengths": ["Participated in the mock interview"] if answered > 0 else ["None demonstrated"],
            "areas_for_improvement": ["Answer all questions", "Use the STAR method for behavioural questions", "Practice speaking in complete sentences"],
            "question_scores": [
                {
                    "question_id": str(q.get('id','')),
                    "question": q.get('question',''),
                    "answer": next((t.get('answer','') for t in transcript if str(t.get('question_id','')) == str(q.get('id',''))), ''),
                    "score": 0,
                    "feedback": "No answer was provided for this question.",
                    "ideal_answer_hint": "Prepare a structured answer using the STAR method."
                } for q in questions
            ],
            "tips": ["Answer every question, even if briefly", "Research common interview questions for your target role", "Practice out loud — recording yourself helps"],
            "readiness_level": "Not Ready" if answered == 0 else "Needs Work"
        }
    
    interview.status = 'completed'
    interview.report = json.dumps(report)
    interview.overall_score = report.get('overall_score', 0)
    interview.completed_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'report': report, 'transcript': transcript})


@app.route('/api/mock-interview/list', methods=['GET'])
def list_mock_interviews():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not authenticated'}), 401
    
    interviews = MockInterview.query.filter_by(
        student_id=sid,
        status='completed'
    ).order_by(MockInterview.created_at.desc()).limit(20).all()
    
    result = []
    for iv in interviews:
        result.append({
            'id': iv.id,
            'job_title': iv.job_title,
            'overall_score': iv.overall_score,
            'created_at': iv.created_at.isoformat(),
            'completed_at': iv.completed_at.isoformat() if iv.completed_at else None
        })
    return jsonify(result)


@app.route('/api/mock-interview/<int:interview_id>', methods=['GET'])
def get_mock_interview(interview_id):
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not authenticated'}), 401
    
    interview = MockInterview.query.filter_by(id=interview_id, student_id=sid).first()
    if not interview:
        return jsonify({'error': 'Interview not found'}), 404
    
    return jsonify({
        'id': interview.id,
        'job_title': interview.job_title,
        'job_description': interview.job_description,
        'status': interview.status,
        'questions': json.loads(interview.questions or '[]'),
        'transcript': json.loads(interview.transcript or '[]'),
        'report': json.loads(interview.report) if interview.report else None,
        'overall_score': interview.overall_score,
        'created_at': interview.created_at.isoformat(),
        'completed_at': interview.completed_at.isoformat() if interview.completed_at else None
    })



# ── Credits API ───────────────────────────────────────────────────────────────

@app.route('/api/credits', methods=['GET'])
def get_credits():
    """Return balance and recent ledger for the logged-in student."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    balance = _get_balance(sid)
    ledger = CreditLedger.query.filter_by(student_id=sid)\
                .order_by(CreditLedger.created_at.desc()).limit(20).all()
    return jsonify({
        'balance': balance,
        'costs': CREDIT_COSTS,
        'ledger': [{
            'delta': e.delta,
            'action': e.action,
            'description': e.description,
            'created_at': e.created_at.isoformat(),
        } for e in ledger]
    })


# ── Referral / Ambassador API ─────────────────────────────────────────────────

@app.route('/api/referral', methods=['GET'])
def get_referral():
    """Return the student's own referral code and stats."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Not found'}), 404
    amb = _ensure_ambassador_record(student)
    if not amb:
        amb = Ambassador.query.filter_by(student_id=sid).first()
    balance = _get_balance(sid)
    per_referral = CREDITS_AMBASSADOR_BONUS if amb and amb.is_ambassador else CREDITS_REFERRAL_BONUS
    return jsonify({
        'referral_code': amb.referral_code,
        'is_ambassador': amb.is_ambassador,
        'total_referrals': amb.total_referrals,
        'credits_per_referral': per_referral,
        'credits_referee_bonus': CREDITS_REFEREE_BONUS,
        'balance': balance,
    })


@app.route('/api/ambassador/stats', methods=['GET'])
def ambassador_stats():
    """Return detailed ambassador stats — only accessible by ambassadors."""
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    amb = Ambassador.query.filter_by(student_id=sid).first()
    if not amb or not amb.is_ambassador:
        return jsonify({'error': 'Not an ambassador'}), 403

    # All referral reward ledger entries for this ambassador
    rewards = CreditLedger.query.filter_by(
        student_id=sid, action='referral_reward'
    ).order_by(CreditLedger.created_at.desc()).all()

    total_credits_earned = sum(r.delta for r in rewards)

    # Extract referred emails from description field (stored as "Referred a new student (email)")
    referred = []
    for r in rewards:
        desc = r.description or ''
        import re as _re
        match = _re.search(r'\((.+?)\)', desc)
        email_shown = match.group(1) if match else '—'
        # Mask email for privacy: show only first 3 chars + domain
        if '@' in email_shown:
            local, domain = email_shown.split('@', 1)
            masked = local[:3] + '***@' + domain
        else:
            masked = email_shown
        referred.append({
            'masked_email': masked,
            'credits_earned': r.delta,
            'date': r.created_at.isoformat(),
        })

    # Weekly breakdown: referrals in last 7 days vs prior 7 days
    from datetime import timedelta
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)
    this_week = sum(1 for r in rewards if r.created_at >= week_ago)
    last_week = sum(1 for r in rewards if two_weeks_ago <= r.created_at < week_ago)

    return jsonify({
        'referral_code': amb.referral_code,
        'total_referrals': amb.total_referrals,
        'total_credits_earned': total_credits_earned,
        'credits_per_referral': CREDITS_AMBASSADOR_BONUS,
        'this_week': this_week,
        'last_week': last_week,
        'referred': referred,          # newest first, emails masked
        'is_ambassador': True,
    })


@app.route('/api/referral/validate', methods=['POST'])
def validate_referral():
    """Check whether a referral code is valid (used during signup preview)."""
    code = (request.json or {}).get('code', '').strip().upper()
    if not code:
        return jsonify({'valid': False, 'message': 'No code provided'}), 400
    amb = Ambassador.query.filter_by(referral_code=code).first()
    if not amb:
        return jsonify({'valid': False, 'message': 'Code not found'})
    referrer = Student.query.get(amb.student_id)
    return jsonify({
        'valid': True,
        'referrer_name': referrer.name.split()[0] if referrer and referrer.name else 'a student',
        'bonus_credits': CREDITS_REFEREE_BONUS,
        'message': f'Code valid! You\'ll get {CREDITS_REFEREE_BONUS} bonus credits on signup.',
    })


@app.route('/api/admin/ambassador', methods=['POST'])
def toggle_ambassador():
    """Promote or demote a student to campus ambassador (admin only via secret key)."""
    secret = request.json.get('admin_secret', '')
    if secret != os.environ.get('ADMIN_SECRET', 'kairo-admin-2026'):
        return jsonify({'error': 'Unauthorized'}), 403
    email = request.json.get('email', '').lower().strip()
    student = Student.query.filter_by(email=email).first()
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    amb = Ambassador.query.filter_by(student_id=student.id).first()
    if not amb:
        return jsonify({'error': 'No ambassador record'}), 404
    amb.is_ambassador = not amb.is_ambassador
    db.session.commit()
    return jsonify({'email': email, 'is_ambassador': amb.is_ambassador,
                    'referral_code': amb.referral_code})


@app.route('/api/admin/grant-credits', methods=['POST'])
def admin_grant_credits():
    """Manually grant credits to a student (admin only)."""
    secret = request.json.get('admin_secret', '')
    if secret != os.environ.get('ADMIN_SECRET', 'kairo-admin-2026'):
        return jsonify({'error': 'Unauthorized'}), 403
    email = request.json.get('email', '').lower().strip()
    amount = int(request.json.get('amount', 0))
    reason = request.json.get('reason', 'Admin grant')
    if amount <= 0:
        return jsonify({'error': 'Amount must be positive'}), 400
    student = Student.query.filter_by(email=email).first()
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    new_balance = _add_credits(student.id, amount, 'admin_grant', reason)
    return jsonify({'email': email, 'granted': amount, 'new_balance': new_balance})


@app.route('/api/health')
def health():
    s = get_llm_status()
    return jsonify({'status': 'ok', 'active_backend': s['active_backend']})

# Startup — create tables and apply any missing column migrations
with app.app_context():
    # Each DDL statement runs inside its own savepoint so that a concurrent
    # worker racing on CREATE TABLE / ALTER TABLE only rolls back that one
    # statement and never poisons the outer transaction.  This eliminates the
    # pg_type UniqueViolation crash seen when gunicorn boots multiple workers
    # at the same time.
    DDL_STATEMENTS = [
        """CREATE TABLE IF NOT EXISTS student (
                id VARCHAR(36) PRIMARY KEY,
                email VARCHAR(120) UNIQUE NOT NULL,
                name VARCHAR(100),
                password_hash VARCHAR(256),
                college VARCHAR(100) DEFAULT 'VIT Vellore',
                profile_data TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )""",
        """CREATE TABLE IF NOT EXISTS conversation (
                id SERIAL PRIMARY KEY,
                student_id VARCHAR(36) NOT NULL REFERENCES student(id),
                messages TEXT DEFAULT '[]',
                topic VARCHAR(100) DEFAULT 'general',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )""",
        """CREATE TABLE IF NOT EXISTS resume (
                id SERIAL PRIMARY KEY,
                student_id VARCHAR(36) NOT NULL REFERENCES student(id),
                job_description TEXT,
                resume_data TEXT,
                edited_html TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )""",
        """CREATE TABLE IF NOT EXISTS resume_version (
                id SERIAL PRIMARY KEY,
                resume_id INTEGER NOT NULL REFERENCES resume(id),
                version_number INTEGER NOT NULL DEFAULT 1,
                resume_data TEXT,
                edited_html TEXT,
                label VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW()
            )""",
        """CREATE TABLE IF NOT EXISTS mock_interview (
                id SERIAL PRIMARY KEY,
                student_id VARCHAR(36) NOT NULL REFERENCES student(id),
                job_title VARCHAR(200),
                job_description TEXT,
                status VARCHAR(20) DEFAULT 'pending',
                questions TEXT DEFAULT '[]',
                transcript TEXT DEFAULT '[]',
                report TEXT,
                overall_score INTEGER,
                created_at TIMESTAMP DEFAULT NOW(),
                completed_at TIMESTAMP
            )""",
        """CREATE TABLE IF NOT EXISTS credit_ledger (
                id SERIAL PRIMARY KEY,
                student_id VARCHAR(36) NOT NULL REFERENCES student(id),
                delta INTEGER NOT NULL,
                action VARCHAR(80) NOT NULL,
                description VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            )""",
        """CREATE TABLE IF NOT EXISTS ambassador (
                id SERIAL PRIMARY KEY,
                student_id VARCHAR(36) NOT NULL REFERENCES student(id) UNIQUE,
                referral_code VARCHAR(20) NOT NULL UNIQUE,
                is_ambassador BOOLEAN DEFAULT FALSE,
                total_referrals INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            )""",
        # Column back-fills
        "ALTER TABLE student ADD COLUMN IF NOT EXISTS password_hash VARCHAR(256)",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
        "ALTER TABLE resume ADD COLUMN IF NOT EXISTS edited_html TEXT",
        "ALTER TABLE resume ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
    ]

    with db.engine.connect() as conn:
        # Savepoints require an explicit transaction — begin one now.
        conn.execute(db.text("BEGIN"))
        for stmt in DDL_STATEMENTS:
            # Each statement gets its own savepoint — if it fails (e.g. a
            # concurrent worker already created the table) we roll back only
            # that savepoint and continue, leaving the transaction healthy.
            try:
                conn.execute(db.text("SAVEPOINT _kairo_ddl"))
                conn.execute(db.text(stmt))
                conn.execute(db.text("RELEASE SAVEPOINT _kairo_ddl"))
            except Exception as e:
                conn.execute(db.text("ROLLBACK TO SAVEPOINT _kairo_ddl"))
                # Log unexpected errors (not just "already exists" races)
                err = str(e).lower()
                if "already exists" not in err and "duplicate" not in err:
                    print(f"DDL warning: {e}")
        conn.execute(db.text("COMMIT"))
        print("Database schema ready.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)