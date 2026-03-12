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

# Fix Railway's postgres:// URL (SQLAlchemy requires postgresql://)
database_url = os.environ.get('DATABASE_URL', 'sqlite:///kairo.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
CORS(app)

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

# ── LLM Helper ────────────────────────────────────────────────────────────────

def _call_groq(messages, system_prompt):
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
                json={"model": model, "messages": full_messages, "max_tokens": 1000},
                timeout=30
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


def _call_openrouter(messages, system_prompt):
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
                json={"model": model, "messages": full_messages, "max_tokens": 1000},
                timeout=30
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


def _call_ollama(messages, system_prompt):
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


def call_llm(messages, system_prompt=""):
    """Call LLM with provider priority: Groq → OpenRouter → Ollama."""
    result = _call_groq(messages, system_prompt)
    if result:
        return result

    result = _call_openrouter(messages, system_prompt)
    if result:
        return result

    result = _call_ollama(messages, system_prompt)
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
    return f"""You are Kairo, an empathetic AI career coach helping a student at VIT Vellore build their professional profile.

Your job is to conduct a friendly, conversational interview to extract rich details about the student's:
- Courses and subjects studied (with marks/grades if available)
- Projects built (tech stack, problem solved, impact)
- Internships and work experience
- Extracurriculars, clubs, hackathons, competitions
- Skills (technical and soft)
- Certifications and achievements
- Industrial visits and workshops

Current profile data collected so far:
{json.dumps(profile_data, indent=2)}

Guidelines:
- Ask ONE focused question at a time
- Be encouraging and dig deeper with follow-up questions
- Extract specifics: numbers, technologies, outcomes, timelines
- If they mention something vague, ask for more details
- After gathering enough on a topic, naturally move to the next
- Keep responses concise and conversational (2-3 sentences max)
- When you have a comprehensive profile, say "PROFILE_COMPLETE" at the start of your response

Start by warmly greeting them and asking about their course/branch."""


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
    system = """Extract a structured student profile from this conversation. 
Return ONLY valid JSON with these keys (use empty arrays/strings if not found):
{
  "name": "", "email": "", "phone": "",
  "branch": "", "year": "", "cgpa": "",
  "subjects": [{"name": "", "grade": "", "interest": "high/medium/low"}],
  "projects": [{"name": "", "description": "", "tech": [], "role": ""}],
  "internships": [{"company": "", "role": "", "duration": "", "work": ""}],
  "skills": {"technical": [], "tools": [], "languages": []},
  "clubs": [], "hackathons": [], "certifications": [],
  "achievements": [], "industrial_visits": []
}"""
    
    conv_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    result = call_llm([{"role": "user", "content": f"Conversation:\n{conv_text}\n\nExtract the profile:"}], system)
    
    try:
        # Clean potential markdown
        clean = result.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean)
    except:
        return {}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')

@app.route('/resume')
def resume_page():
    return render_template('resume.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').lower().strip()
    name = data.get('name', '').strip()
    password = data.get('password', '').strip()

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

    if not student:
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

    session['student_id'] = student.id
    session['student_email'] = student.email

    return jsonify({
        'id': student.id,
        'email': student.email,
        'name': student.name,
        'profile': json.loads(student.profile_data or '{}')
    })

@app.route('/api/auth/me')
def me():
    sid = session.get('student_id')
    if not sid:
        return jsonify({'error': 'Not logged in'}), 401
    student = Student.query.get(sid)
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify({
        'id': student.id,
        'email': student.email,
        'name': student.name,
        'profile': json.loads(student.profile_data or '{}')
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
                existing.update(extracted)
                student.profile_data = json.dumps(existing)
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
    
    # Use LLM to extract structured info from document
    doc_insights = ""
    if extracted_text:
        doc_insights = call_llm(
            [{"role": "user", "content": f"Extract key information from this document:\n{extracted_text[:3000]}"}],
            "You are analyzing a student document (transcript, certificate, or resume). Extract key information in a brief, structured summary. Focus on: courses/grades, achievements, certifications, skills."
        )
    
    return jsonify({
        'filename': unique_name,
        'original_name': filename,
        'extracted_text': extracted_text[:1000],
        'insights': doc_insights
    })

@app.route('/api/generate-resume', methods=['POST'])
def generate_resume():
    data = request.json
    job_description = data.get('job_description', '')
    sid = session.get('student_id')
    
    if not job_description:
        return jsonify({'error': 'Job description required'}), 400
    
    profile_data = data.get('profile_data', {})
    
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

@app.route('/api/health')
def health():
    s = get_llm_status()
    return jsonify({'status': 'ok', 'active_backend': s['active_backend']})

# Startup — create tables and apply any missing column migrations
with app.app_context():
    db.create_all()
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text(
                "ALTER TABLE student ADD COLUMN IF NOT EXISTS password_hash VARCHAR(256)"
            ))
            conn.execute(db.text(
                "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"
            ))
            conn.execute(db.text(
                "ALTER TABLE resume ADD COLUMN IF NOT EXISTS edited_html TEXT"
            ))
            conn.execute(db.text(
                "ALTER TABLE resume ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"
            ))
            conn.commit()
    except Exception as e:
        print(f"Migration note: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)