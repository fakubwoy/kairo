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
    ├── dashboard.html      # User dashboard + profile editor + ambassador hub
    ├── interview.html      # AI chat interface (profile builder)
    ├── interview_prep.html # Mock interview (AI interviewer + voice answers)
    ├── resume.html         # Resume generator + saved resumes viewer
    └── presence.html       # Presence audit (GitHub + LinkedIn scorer)
```

---

## Features

**Profile Builder** — video-call style AI interview with TTS; live transcript sidebar; 10-step structured flow covering academics, projects, internships, skills, goals, and faculty references; PiP self-view overlay; deep-merge logic so returning users never lose data; early exit saves profile to the server immediately

**Dashboard** — profile completeness tracker across 8 sections; profile editor always re-fetches fresh data from the server on open; manual editing for all sections including projects, extracurriculars, and faculty references; skills displayed and persisted correctly across page reloads

**Resumes** — JD-tailored generation, ATS-friendly format, inline editing, PDF download, version history with side-by-side diff; self-intro video script generator (target role, duration, tone) with graceful insufficient-credits handling

**Mock Interview** — 8 tailored questions (behavioural, technical, profile-based) delivered via TTS; voice or text answers; detailed report with score, readiness level, per-question feedback, and actionable tips; credit check enforced before session start

**Voice Input** — Groq Whisper transcription in both profile builder and mock interview; broad browser compatibility including Brave; auto-sends after 1.8s

**Presence Audit** — AI-powered LinkedIn & GitHub audit; scores both platforms 1–10; generates 4–6 prioritised tips per platform with clickable deep-dive drawer (read-only AI explanation with why-it-matters, what-good-looks-like, common mistakes, step-by-step fix, and before/after example); profile consistency checklist; 4-week structured action plan; pre-fills from Kairo profile if logged in; no credits charged

**Imports** — GitHub profile import, LinkedIn text paste, document upload (PDF/image)

**Credit System** — append-only ledger tracks every credit transaction; costs per action: chat message (1), document upload (2), intro script (3), resume generation (5), mock interview (10); 50 free credits on signup; insufficient credits shown gracefully in UI across all features including session start buttons

**Referral System** — every user gets a unique referral code (e.g. `ARJUN-4F2A`) on signup; referrer earns 20 credits per successful signup; referred student earns 10 bonus credits; referral code can be applied at signup or via invite link `/?ref=CODE`

**Campus Ambassador Program** — official ambassadors earn 35 credits per referral (vs 20 for regular users); Ambassador Hub on the dashboard shows live stats (total signups, credits earned, week-over-week trend, full referral table with privacy-masked emails); share tools include copy code, copy invite link, WhatsApp share, and a downloadable recruitment poster (PNG with QR code, branding, and ambassador name) generated client-side on canvas; ambassador badge shown in sidebar; ambassador perk displayed in credits panel

---

## Credit System Details

| Event | Credits |
|-------|---------|
| Signup bonus | +50 |
| Referred by someone (new user) | +10 |
| Referring a new user (regular) | +20 |
| Referring a new user (ambassador) | +35 |
| Chat message | −1 |
| Document upload | −2 |
| Intro script generation | −3 |
| Resume generation | −5 |
| Mock interview session | −10 |

Credits are tracked in an append-only `credit_ledger` table. The balance is always computed as `SUM(delta)` — no balance column that can drift out of sync.

---

## Ambassador Program Details

Ambassadors are promoted via the admin API (`POST /api/admin/ambassador`). Once promoted:

- Referral reward increases from 20 → 35 credits per signup automatically
- A "★ Campus Ambassador" badge appears in their sidebar
- The Ambassador Hub section appears on their dashboard with:
  - Total signups driven, total credits earned, this-week vs last-week trend
  - Full table of referred students (emails privacy-masked as `aru***@vitstudent.ac.in`)
  - Share card with one-click copy, WhatsApp share, and poster download
- A downloadable 900×1200px recruitment poster is generated client-side with:
  - Kairo branding, QR code linking to their referral URL, their referral code, and their name
  - Suitable for WhatsApp groups, notice boards, and class chats

---

## Data Flow — Profile Persistence

Profile data is always written to the server DB (`PUT /api/profile`) and never relied on from localStorage alone. The three paths that update the profile all persist server-side:

1. **Normal interview completion** — LLM signals `PROFILE_COMPLETE`; backend extracts structured profile from full conversation and merges into `profile_data`
2. **Early session end** — frontend extracts profile locally, flattens the nested schema, and calls `PUT /api/profile` before showing the summary screen
3. **Correction chat** — each message triggers a mini extraction on the exchange and calls `PUT /api/profile` with any new fields found; localStorage is then refreshed from the server response

On every dashboard page load, `loadProfile()` is called to re-fetch the latest profile from `/api/auth/me`, ensuring skills and other fields are always in sync with the server even if localStorage is stale.

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
ADMIN_SECRET=<your admin secret for ambassador/credit management>
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
| `ADMIN_SECRET` | No | Secret key for admin endpoints|
| `GITHUB_TOKEN` | No | GitHub token for higher API rate limits on profile import |

*Groq is the primary provider and also powers Whisper voice transcription. For local dev, Ollama works without any API key (voice input requires Groq).

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login or register. Pass `referral_code` to apply a referral at signup |
| GET | `/api/auth/me` | Get current user + profile + credit balance |
| POST | `/api/auth/logout` | Clear session |
| POST | `/api/auth/update-name` | Update display name |
| POST | `/api/auth/change-password` | Change password |

### Profile
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET, PUT | `/api/profile` | Get or update profile data |
| POST | `/api/upload` | Upload PDF/image document (costs 2 credits) |
| DELETE | `/api/documents/<doc_index>` | Delete an uploaded document |

### Chat (Profile Builder)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message to Kairo interview AI (costs 1 credit). Pass `silent_bootstrap: true` for the opening message (free). Pass `extract_only: true` for one-shot LLM calls that skip conversation history |
| GET | `/api/conversations/active` | Get latest conversation with full message history |
| POST | `/api/conversations/new` | Archive current conversation and start fresh |

### Resumes
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate-resume` | Generate resume from JD (costs 5 credits) |
| GET | `/api/resumes` | List all saved resumes |
| GET | `/api/resumes/<id>` | Get a specific saved resume |
| POST | `/api/resumes/<id>/save-edit` | Save an inline edit, creates a new version snapshot |
| GET | `/api/resumes/<id>/versions` | List version history for a resume |
| GET | `/api/resumes/<id>/diff` | Get HTML and unified diff between two resume versions |

### Mock Interview
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/mock-interview/start` | Generate 8 tailored questions and start a session (costs 10 credits) |
| POST | `/api/mock-interview/<id>/submit-answer` | Submit or update answer for a question |
| POST | `/api/mock-interview/<id>/complete` | Finish interview and generate scored report |
| GET | `/api/mock-interview/list` | List all completed interview sessions |
| GET | `/api/mock-interview/<id>` | Get a specific session with full report and transcript |

### Credits & Referrals
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/credits` | Get balance and last 20 ledger transactions |
| GET | `/api/referral` | Get own referral code, stats, and per-referral reward (ambassador-aware) |
| POST | `/api/referral/validate` | Check if a referral code is valid before signup |

### Ambassador
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ambassador/stats` | Get detailed ambassador stats — signups, credits earned, week-over-week trend, referral table (ambassador only, 403 otherwise) |

### Tools & Utilities
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/transcribe` | Transcribe audio via Groq Whisper |
| POST | `/api/github-profile` | Import profile data from GitHub username |
| POST | `/api/linkedin-hints` | Extract structured profile data from pasted LinkedIn text |
| POST | `/api/fetch-jd` | Scrape and clean a job description from a URL (supports Naukri, Internshala, Hirist, Indeed, and more) |
| POST | `/api/generate-intro-script` | Generate a self-intro video script (costs 3 credits) |
| GET | `/api/llm-status` | Check status of all LLM backends (Groq, OpenRouter, Ollama) |
| GET | `/api/health` | Health check with active backend name |
| POST | `/api/presence-audit` | Score GitHub + LinkedIn presence and return prioritised tips, consistency checks, and 4-week plan |
| POST | `/api/presence-audit/deep-dive` | Return an in-depth AI explanation for a single audit tip (why it matters, examples, step-by-step fix) |

### Admin (protected by `ADMIN_SECRET`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/ambassador` | Toggle ambassador status for a student by email |
| POST | `/api/admin/grant-credits` | Manually grant credits to a student |

---

## Roadmap

- [ ] Chrome extension for one-click job application
- [ ] VIT course template pre-loading (known subjects, branches, electives)
- [ ] VIT email validation enforcement

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
| QR code generation | QRCode.js (client-side, CDN) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Railway |

No paid APIs required. Groq provides free access to Llama 3.3 70B and Whisper Large v3 with a generous rate limit.

---