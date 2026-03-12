import os
import json
import uuid
import requests
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
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

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(36), db.ForeignKey('student.id'), nullable=False)
    job_description = db.Column(db.Text)
    resume_data = db.Column(db.Text)
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
    name = data.get('name', '')
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    # VIT email validation — accepts both student and staff formats:
    # farhaan.khan2022@vitstudent.ac.in  (students)
    # faculty.name@vit.ac.in             (staff/faculty)
    VALID_VIT_DOMAINS = ('@vitstudent.ac.in', '@vit.ac.in')
    if not any(email.endswith(d) for d in VALID_VIT_DOMAINS):
        return jsonify({'error': 'Please use your VIT email (e.g. name2022@vitstudent.ac.in)'}), 400
    
    student = Student.query.filter_by(email=email).first()
    if not student:
        student = Student(email=email, name=name)
        db.session.add(student)
        db.session.commit()
    elif name and not student.name:
        student.name = name
        db.session.commit()
    
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

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    conversation_id = data.get('conversation_id')
    
    sid = session.get('student_id')
    if not sid:
        # Allow demo without login
        sid = 'demo'

    # Get or create conversation
    conv = None
    if conversation_id:
        conv = Conversation.query.get(conversation_id)
    
    if not conv and sid != 'demo':
        conv = Conversation(student_id=sid, topic='profile_building')
        db.session.add(conv)
        db.session.commit()

    # Load messages
    if conv:
        messages = json.loads(conv.messages or '[]')
    else:
        messages = data.get('messages', [])

    # Get student profile for context
    profile_data = {}
    if sid and sid != 'demo':
        student = Student.query.get(sid)
        if student:
            profile_data = json.loads(student.profile_data or '{}')

    # Add user message
    messages.append({"role": "user", "content": user_message})

    # Call LLM
    system_prompt = get_interview_system_prompt(profile_data)
    ai_response = call_llm(messages, system_prompt)

    # Add AI response
    messages.append({"role": "assistant", "content": ai_response})

    # Check if profile complete
    profile_complete = "PROFILE_COMPLETE" in ai_response
    if profile_complete:
        ai_response = ai_response.replace("PROFILE_COMPLETE", "").strip()
        messages[-1]['content'] = ai_response
        
        # Extract and save profile
        if sid and sid != 'demo':
            extracted = extract_profile_from_conversation(messages)
            student = Student.query.get(sid)
            if student and extracted:
                existing = json.loads(student.profile_data or '{}')
                existing.update(extracted)
                student.profile_data = json.dumps(existing)
                db.session.commit()

    # Save conversation
    if conv:
        conv.messages = json.dumps(messages[-50:])  # Keep last 50 messages
        db.session.commit()

    return jsonify({
        'response': ai_response,
        'conversation_id': conv.id if conv else None,
        'profile_complete': profile_complete,
        'messages': messages
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

@app.route('/api/llm-status')
def llm_status():
    return jsonify(get_llm_status())

@app.route('/api/health')
def health():
    s = get_llm_status()
    return jsonify({'status': 'ok', 'active_backend': s['active_backend']})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)