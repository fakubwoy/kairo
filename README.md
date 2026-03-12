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
    └── resume.html         # Resume generator + saved resumes viewer
```

---

## Features

| Feature | Status |
|---------|--------|
| Landing page with login | Done |
| Student auth (email + password) | Done |
| AI chat interview interface | Done |
| Profile extraction from chat | Done |
| Manual profile editing from dashboard | Done |
| Document upload (PDF/image) | Done |
| Resume generation from JD | Done |
| Saved resumes list with viewer | Done |
| Resume preview & PDF download | Done |
| Professional resume format (ATS-friendly) | Done |
| Dashboard with profile progress | Done |
| SQLite/Postgres persistence | Done |
| Groq free LLM integration (primary) | Done |
| OpenRouter free LLM fallback | Done |
| Local Ollama fallback | Done |
| Railway deployment ready | Done |

---

## Local Setup

### 1. Clone and install

```bash
git clone <your-repo>
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
4. Models used: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`

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
git remote add origin https://github.com/your-username/kairo.git
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

*Groq is the primary provider. For local dev, Ollama works without any API key.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login or register |
| GET | `/api/auth/me` | Get current user + profile |
| POST | `/api/auth/logout` | Clear session |
| POST | `/api/auth/update-name` | Update display name |
| POST | `/api/auth/change-password` | Change password |
| GET, PUT | `/api/profile` | Get or update profile data |
| POST | `/api/chat` | Send message to Kairo interview AI |
| GET | `/api/conversations/active` | Get latest conversation |
| POST | `/api/conversations/new` | Start a fresh conversation |
| POST | `/api/generate-resume` | Generate resume from JD |
| GET | `/api/resumes` | List all saved resumes |
| GET | `/api/resumes/<id>` | Get a specific saved resume |
| POST | `/api/upload` | Upload PDF/image document |
| GET | `/api/llm-status` | Check LLM backend status |
| GET | `/api/health` | Health check |

---

## Roadmap

- [ ] Voice/audio conversation with Whisper (open-source STT)
- [ ] LinkedIn/GitHub profile scraping hints
- [ ] Chrome extension for job application
- [ ] VIT email validation enforcement
- [ ] Credit system for usage limits
- [ ] Resume versioning and diff view
- [ ] Faculty reference suggestions
- [ ] Campus ambassador referral system
- [ ] Self-intro video script generation

---

## Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Database | SQLite / PostgreSQL |
| LLM (primary) | Llama 3.3 70B via Groq (free) |
| LLM (fallback) | Llama 3.2 via OpenRouter (free) |
| LLM (local) | Llama 3.2 via Ollama |
| Caching | Redis (optional) |
| PDF extraction | pdfplumber |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Railway |

No paid APIs required. Groq provides free access to Llama 3.3 70B with a generous rate limit.

---
