# 🤖 FAQ Chatbot

> A RAG-powered customer support chatbot that answers questions from your own knowledge base — no hallucinations, no guessing.

Built with **ChromaDB** + **Mistral AI** + **Flask**. Upload your FAQs, deploy, and let the bot handle support.

---

## ✨ Features

- 💬 **Smart answers** — retrieves relevant context from ChromaDB before calling the LLM
- 🚫 **No hallucinations** — responses are grounded strictly in your knowledge base
- 🧠 **Self-improving** — admin can answer unanswered questions directly from the dashboard, teaching the bot in real time
- 📊 **Analytics dashboard** — track users, queries, response times, and relevance scores
- 📁 **Bulk import** — upload a CSV to populate the knowledge base instantly
- 🔁 **Returning users** — identified by email, query history stacks on the same account

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Vector DB | ChromaDB (SQLite-backed) |
| LLM | Mistral AI |
| Analytics | SQLite |
| Deployment | Render (Docker) |

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/roneya/chat-bot.git
cd chat-bot
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

### 3. Seed the knowledge base

```bash
python ingest.py
```

### 4. Run

```bash
./start.sh
```

App runs at `http://localhost:5001` — admin dashboard at `/admin`.

---

## 📋 CSV Format

To bulk-load Q&A pairs, upload a CSV with these exact columns:

```csv
question,answer
What is your return policy?,We offer 30-day returns on all items.
How do I track my order?,Log in and visit the Orders section.
```

---

## 🖥 Admin Dashboard

| Feature | What it does |
|---|---|
| Upload CSV | Bulk-load Q&A pairs into ChromaDB |
| Knowledge Base | View, search, and delete individual entries |
| Unanswered Queries | See what the bot couldn't answer — add answers on the spot |
| Users | Track who's using the bot and how often |
| Analytics | Response times, relevance scores, top questions |
| Clear DB | Wipe and reload the knowledge base |

---

## 📦 Deploying to Render

The repo includes a `Dockerfile` and `render.yaml`. Just:

1. Push to GitHub
2. Connect repo on [render.com](https://render.com)
3. Add `MISTRAL_API_KEY` as an environment variable
4. Deploy

The embedding model and FAQ data are baked into the Docker image at build time — zero cold-start delay.

---

## 🏗 Architecture

```
User asks question
      ↓
ChromaDB similarity search (top 5 matches)
      ↓
Relevant context injected into prompt
      ↓
Mistral AI generates grounded answer
      ↓
Answer returned + logged to SQLite
```

> **Single source of truth:** ChromaDB. CSV is only used for bulk ingestion — the bot never reads files at runtime.

---

Made by [Rohan Vidhate](https://github.com/roneya)
