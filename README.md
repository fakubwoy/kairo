# 🌟 Kairo — AI Career Co-pilot for Students

> A conversational AI platform that interviews students, builds their career profiles, and generates tailored resumes for every job application.

Built with Python (Flask) + vanilla HTML/CSS/JS. Uses open-source LLMs via OpenRouter (free tier) or local Ollama.

---

## 📁 Project Structure

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
    ├── dashboard.html      # User dashboard
    ├── interview.html      # AI chat interface (profile builder)
    └── resume.html         # Resume generator page
```

---

## ✅ Features (Working - ~45% of full vision)

| Feature | Status |
|---------|--------|
| Landing page with login | ✅ Done |
| Student auth (email-based) | ✅ Done |
| AI chat interview interface | ✅ Done |
| Profile extraction from chat | ✅ Done |
| Document upload (PDF/image) | ✅ Done |
| Resume generation from JD | ✅ Done |
| Resume preview & PDF download | ✅ Done |
| Dashboard with profile progress | ✅ Done |
| SQLite/Postgres persistence | ✅ Done |
| OpenRouter free LLM integration | ✅ Done |
| Local Ollama fallback | ✅ Done |
| Railway deployment ready | ✅ Done |

---

## 🚀 Local Setup

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

**Recommended: OpenRouter (free)**
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up → API Keys → Create key
3. Add to `.env`: `OPENROUTER_API_KEY=sk-or-...`
4. Free models used: `meta-llama/llama-3.2-3b-instruct:free`

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

## 🚂 Deploy to Railway

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
OPENROUTER_API_KEY=<your openrouter key>
```

### Step 4: (Optional) Add Postgres database
1. In Railway → **New** → **Database** → **PostgreSQL**
2. Railway auto-sets `DATABASE_URL` env var
3. Your app uses it automatically

### Step 5: Deploy
Railway auto-deploys on every push to main. Your app will be live at:
`https://your-app.railway.app`

---

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | Flask session secret (random string) |
| `OPENROUTER_API_KEY` | Yes* | Free LLM API key from openrouter.ai |
| `DATABASE_URL` | Auto | Set by Railway Postgres add-on |
| `OLLAMA_BASE_URL` | No | Local Ollama URL (local dev only) |
| `OLLAMA_MODEL` | No | Ollama model name (default: llama3.2) |

*Required for Railway deployment. For local dev, can use Ollama instead.

---

## 🛣️ Roadmap (Next 50%)

- [ ] Voice/audio conversation with Whisper (open-source STT)
- [ ] LinkedIn/GitHub profile scraping hints
- [ ] Chrome extension for job application
- [ ] VIT email validation enforcement
- [ ] Credit system for usage limits
- [ ] Multi-resume management & versioning
- [ ] Faculty reference suggestions
- [ ] Campus ambassador referral system
- [ ] Self-intro video script generation

---

## 🤖 Open Source Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Database | SQLite / PostgreSQL |
| LLM | Llama 3.2 via OpenRouter (free) |
| PDF extraction | pdfplumber |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Railway |

**No paid APIs required.** OpenRouter provides free access to Llama 3.2 (3B) and other models.

---

## 📝 License

MIT — build on it freely.
