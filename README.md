# Kairo — AI Career Co-pilot for Students

> A conversational AI platform that interviews students, builds their career profiles, and generates tailored resumes for every job application.

Built with Python (Flask) + vanilla HTML/CSS/JS. Uses open-source LLMs via Groq (free tier), OpenRouter, or local Ollama.

---

## Project Structure

```
kairo/
├── app.py                  # Main Flask application (routes, LLM, DB)
├── requirements.txt        # Python dependencies
├── Procfile                # Railway/Heroku process config
├── railway.toml            # Railway deployment config
├── .env.example            # Environment variables template
├── .gitignore
├── uploads/                # User uploaded files (auto-created)
├── instance/               # SQLite DB (auto-created)
└── templates/
    ├── index.html          # Landing page
    ├── dashboard.html      # User dashboard + profile editor
    ├── interview.html      # AI chat interface (profile builder)
    ├── interview_prep.html # Mock interview (AI interviewer + voice answers)
    └── resume.html         # Resume generator + saved resumes viewer
```

---

## Features

**Profile Builder** — video-call style AI interview with TTS; live transcript sidebar; 10-step structured flow covering academics, projects, internships, skills, goals, and faculty references; PiP self-view overlay; deep-merge logic so returning users never lose data; early exit saves profile to the server immediately

**Dashboard** — profile completeness tracker across 8 sections; profile editor always re-fetches fresh data from the server on open; manual editing for all sections including projects, extracurriculars, and faculty references

**Resumes** — JD-tailored generation, ATS-friendly format, inline editing, PDF download, version history with side-by-side diff; self-intro video script generator (target role, duration, tone)

**Mock Interview** — 8 tailored questions (behavioural, technical, profile-based) delivered via TTS; voice or text answers; detailed report with score, readiness level, per-question feedback, and actionable tips

**Voice Input** — Groq Whisper transcription in both profile builder and mock interview; broad browser compatibility including Brave; auto-sends after 1.8s

**Imports** — GitHub profile import, LinkedIn text paste, document upload (PDF/image)

---

## Data Flow — Profile Persistence

Profile data is always written to the server DB (`PUT /api/profile`) and never relied on from localStorage alone. The three paths that update the profile all persist server-side:

1. **Normal interview completion** — LLM signals `PROFILE_COMPLETE`; backend extracts structured profile from full conversation and merges into `profile_data`
2. **Early session end** — frontend extracts profile locally, flattens the nested schema, and calls `PUT /api/profile` before showing the summary screen
3. **Correction chat** — each message triggers a mini extraction on the exchange and calls `PUT /api/profile` with any new fields found; localStorage is then refreshed from the server response

The `extract_profile_from_conversation` schema in `app.py` includes `faculty_references` and structured `extracurriculars` so all three paths capture these fields correctly.

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/fakubwoy/kairo.git
cd kairo
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Get a free LLM API key

**Recommended: Groq (free, fastest)**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create key
3. Add to `.env`: `GROQ_API_KEY=gsk_...`
4. Models used: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `whisper-large-v3`

**Alternative: OpenRouter (free)**
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up → API Keys → Create key
3. Add to `.env`: `OPENROUTER_API_KEY=sk-or-...`

**Alternative: Local Ollama**
```bash
# Install Ollama from ollama.ai
ollama pull llama3.2
# Set in .env: OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Run

```bash
python app.py
# Open http://localhost:5000
```

---

## Deploy to Railway

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/fakubwoy/kairo.git
git push -u origin main
```

### Step 2: Create Railway project
1. Go to [railway.app](https://railway.app)
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `kairo` repository
4. Railway auto-detects Python and installs dependencies

### Step 3: Add environment variables
In Railway dashboard → your service → **Variables** tab, add:

```
SECRET_KEY=<generate a random 32-char string>
GROQ_API_KEY=<your groq key>
```

### Step 4: (Optional) Add Postgres database
1. In Railway → **New** → **Database** → **PostgreSQL**
2. Railway auto-sets `DATABASE_URL` env var
3. Your app uses it automatically

### Step 5: Deploy
Railway auto-deploys on every push to main. Your app will be live at:
`https://your-app.railway.app`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | Flask session secret (random string) |
| `GROQ_API_KEY` | Yes* | Free LLM API key from console.groq.com |
| `OPENROUTER_API_KEY` | No | Fallback LLM key from openrouter.ai |
| `DATABASE_URL` | Auto | Set by Railway Postgres add-on |
| `OLLAMA_BASE_URL` | No | Local Ollama URL (local dev only) |
| `OLLAMA_MODEL` | No | Ollama model name (default: llama3.2) |
| `REDIS_URL` | No | Optional Redis for conversation caching |

*Groq is the primary provider and also powers Whisper voice transcription. For local dev, Ollama works without any API key (voice input requires Groq).

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login or register |
| GET | `/api/auth/me` | Get current user + profile |
| POST | `/api/auth/logout` | Clear session |
| POST | `/api/auth/update-name` | Update display name |
| POST | `/api/auth/change-password` | Change password |

### Profile
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET, PUT | `/api/profile` | Get or update profile data |
| POST | `/api/upload` | Upload PDF/image document |
| DELETE | `/api/documents/<doc_index>` | Delete an uploaded document |

### Chat (Profile Builder)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message to Kairo interview AI. Pass `extract_only: true` for one-shot LLM calls that skip conversation history |
| GET | `/api/conversations/active` | Get latest conversation |
| POST | `/api/conversations/new` | Start a fresh conversation |

### Resumes
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate-resume` | Generate resume from JD |
| GET | `/api/resumes` | List all saved resumes |
| GET | `/api/resumes/<id>` | Get a specific saved resume |
| POST | `/api/resumes/<id>/save-edit` | Save an inline edit to a resume |
| GET | `/api/resumes/<id>/versions` | List version history for a resume |
| GET | `/api/resumes/<id>/diff` | Get diff between two resume versions |

### Mock Interview
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/mock-interview/start` | Generate questions and start a session |
| POST | `/api/mock-interview/<id>/submit-answer` | Submit answer for a question |
| POST | `/api/mock-interview/<id>/complete` | Finish interview and generate report |
| GET | `/api/mock-interview/list` | List all past interview sessions |
| GET | `/api/mock-interview/<id>` | Get a specific session with report and transcript |

### Tools & Utilities
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/transcribe` | Transcribe audio via Groq Whisper |
| POST | `/api/github-profile` | Import profile data from GitHub username |
| POST | `/api/linkedin-hints` | Extract hints from pasted LinkedIn text |
| POST | `/api/generate-intro-script` | Generate a self-intro video script (also accessible from resume page) |
| GET | `/api/llm-status` | Check LLM backend status |
| GET | `/api/health` | Health check |

---

## Roadmap

- [ ] Chrome extension for job application
- [ ] VIT email validation enforcement
- [ ] Credit system for usage limits
- [ ] Campus ambassador referral system

---

## Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Database | SQLite / PostgreSQL |
| LLM (primary) | Llama 3.3 70B via Groq (free) |
| LLM (fallback) | Llama 3.2 via OpenRouter (free) |
| LLM (local) | Llama 3.2 via Ollama |
| Voice / STT | Whisper Large v3 via Groq (free) |
| Caching | Redis (optional) |
| PDF extraction | pdfplumber |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Railway |

No paid APIs required. Groq provides free access to Llama 3.3 70B and Whisper Large v3 with a generous rate limit.

---