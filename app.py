import logging
import os
import sqlite3
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import chromadb
import requests

from ingest import ingest_csv, ingest_pdf, ingest_text, get_chroma_collection

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
CORS(app)

limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

# --- Config --------------------------------------------------------------- #

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set. Add it to .env or export it.")

MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-medium-latest")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "knowledge")
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "1.2"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
DB_PATH = os.environ.get("DB_PATH", os.path.join(CHROMA_PATH, "analytics.db"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "pdf", "txt"}

# --- ChromaDB ------------------------------------------------------------- #

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = get_chroma_collection(chroma_client, COLLECTION_NAME)

# --- Analytics DB --------------------------------------------------------- #


def init_analytics_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT,
            relevance_score REAL,
            response_time_ms INTEGER,
            source_docs INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES queries(id)
        )
    """)
    # Migrations — safely add new columns / constraints to existing tables
    try:
        conn.execute("ALTER TABLE queries ADD COLUMN user_id INTEGER REFERENCES users(id)")
    except Exception:
        pass  # Column already exists
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users (email) WHERE email IS NOT NULL")
    except Exception:
        pass  # Index already exists
    try:
        conn.execute("ALTER TABLE queries ADD COLUMN answered_by_admin INTEGER DEFAULT 0")
    except Exception:
        pass  # Column already exists

    conn.commit()
    conn.close()


init_analytics_db()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def log_query(question, answer, relevance_score, response_time_ms, source_docs, user_id=None):
    db = get_db()
    cursor = db.execute(
        "INSERT INTO queries (user_id, question, answer, relevance_score, response_time_ms, source_docs) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, question, answer, relevance_score, response_time_ms, source_docs),
    )
    db.commit()
    return cursor.lastrowid


def log_document(filename, file_type, chunk_count):
    db = get_db()
    db.execute(
        "INSERT INTO documents (filename, file_type, chunk_count) VALUES (?, ?, ?)",
        (filename, file_type, chunk_count),
    )
    db.commit()


# --- Mistral -------------------------------------------------------------- #

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are a customer support assistant. "
    "Answer questions strictly based on the provided context from our knowledge base. "
    "Be concise and friendly. "
    "If the context doesn't contain the answer, say: 'I don't have information on that. "
    "Please contact our support team at support@shop.com or call 1800-123-456 (Mon–Fri, 9am–6pm).' "
    "Never make up information or answer outside the scope of the knowledge base."
)


def call_mistral(messages: list[dict]) -> str:
    try:
        resp = requests.post(
            MISTRAL_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Mistral API request timed out")
        raise
    except requests.exceptions.HTTPError:
        logger.error("Mistral API error %s: %s", resp.status_code, resp.text)
        raise

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --- RAG pipeline --------------------------------------------------------- #


def retrieve_context(question: str) -> tuple[str | None, float, int]:
    """Returns (context_string, best_relevance_score, num_sources_used)."""
    results = collection.query(
        query_texts=[question],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    context_parts = []
    best_score = float("inf")

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > RELEVANCE_THRESHOLD:
            continue
        best_score = min(best_score, dist)
        source = meta.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else None
    return context, best_score if context_parts else -1, len(context_parts)


def build_messages(context: str | None, chat_history: list[dict], question: str) -> list[dict]:
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\nRelevant Context:\n{context}"
    else:
        system_content += (
            "\n\nNo relevant information was found in the knowledge base. "
            "Let the user know and suggest they upload relevant documents."
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(chat_history[-12:])
    messages.append({"role": "user", "content": question})

    return messages


def ask(question: str, chat_history: list[dict]) -> tuple[str, float, int]:
    """Returns (answer, relevance_score, source_count)."""
    context, score, source_count = retrieve_context(question)
    messages = build_messages(context, chat_history, question)
    answer = call_mistral(messages)
    return answer, score, source_count


# --- File helpers --------------------------------------------------------- #


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Routes: Chat --------------------------------------------------------- #


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/ask", methods=["POST"])
@limiter.limit("20 per minute")
def ask_endpoint():
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Please provide a 'question' field in JSON body"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    logger.info("Question received: %s", question)

    if "history" not in session:
        session["history"] = []

    start = time.time()
    try:
        answer, relevance_score, source_count = ask(question, session["history"])
    except Exception:
        logger.exception("Failed to generate answer")
        return jsonify({"error": "Something went wrong, please try again later"}), 500

    elapsed_ms = int((time.time() - start) * 1000)

    query_id = log_query(question, answer, relevance_score, elapsed_ms, source_count, session.get("user_id"))

    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})
    session.modified = True

    return jsonify({
        "question": question,
        "answer": answer,
        "query_id": query_id,
        "meta": {
            "response_time_ms": elapsed_ms,
            "sources_used": source_count,
            "relevance_score": round(relevance_score, 3) if relevance_score >= 0 else None,
        },
    })


@app.route("/session/start", methods=["POST"])
def start_session():
    data = request.get_json(silent=True)
    if not data or not data.get("name", "").strip():
        return jsonify({"error": "Name is required"}), 400

    name = data["name"].strip()
    email = data.get("email", "").strip() or None
    phone = data.get("phone", "").strip() or None

    db = get_db()

    # If email provided, reuse existing user — email is the unique identifier
    existing = None
    if email:
        existing = db.execute(
            "SELECT id, name FROM users WHERE email = ?", (email,)
        ).fetchone()

    if existing:
        user_id = existing["id"]
        # Update name/phone in case they changed
        db.execute(
            "UPDATE users SET name = ?, phone = ? WHERE id = ?",
            (name, phone, user_id),
        )
    else:
        cursor = db.execute(
            "INSERT INTO users (name, email, phone) VALUES (?, ?, ?)",
            (name, email, phone),
        )
        user_id = cursor.lastrowid

    db.commit()

    session["user_id"] = user_id
    session["user_name"] = name
    session["history"] = []
    session.modified = True

    return jsonify({"status": "ok", "user_id": user_id, "name": name, "returning": existing is not None})


@app.route("/reset", methods=["POST"])
def reset_chat():
    user_id = session.get("user_id")
    user_name = session.get("user_name")
    session.pop("history", None)
    # Keep user info across resets — no need to re-enter details
    if user_id:
        session["user_id"] = user_id
        session["user_name"] = user_name
    return jsonify({"status": "Chat history cleared"})


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json(silent=True)
    if not data or "query_id" not in data or "rating" not in data:
        return jsonify({"error": "Provide 'query_id' and 'rating' (1 or -1)"}), 400
    if data["rating"] not in (1, -1):
        return jsonify({"error": "rating must be 1 (thumbs up) or -1 (thumbs down)"}), 400

    db = get_db()
    db.execute(
        "INSERT INTO feedback (query_id, rating) VALUES (?, ?)",
        (data["query_id"], data["rating"]),
    )
    db.commit()
    return jsonify({"status": "feedback recorded"})


# --- Routes: Document upload ---------------------------------------------- #


@app.route("/upload", methods=["POST"])
@limiter.limit("10 per minute")
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    ext = filename.rsplit(".", 1)[1].lower()
    try:
        if ext == "csv":
            chunk_count = ingest_csv(filepath, collection, source=filename)
        elif ext == "pdf":
            chunk_count = ingest_pdf(filepath, collection, source=filename)
        elif ext == "txt":
            chunk_count = ingest_text(filepath, collection, source=filename)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        logger.exception("Failed to ingest %s", filename)
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    log_document(filename, ext, chunk_count)

    return jsonify({
        "status": "success",
        "filename": filename,
        "chunks_added": chunk_count,
        "total_documents": collection.count(),
    })


# --- Routes: Admin API ---------------------------------------------------- #


@app.route("/api/users", methods=["GET"])
def list_users():
    db = get_db()
    rows = db.execute("""
        SELECT u.id, u.name, u.email, u.phone, u.created_at,
               COUNT(q.id) as query_count
        FROM users u
        LEFT JOIN queries q ON q.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/documents", methods=["GET"])
def list_documents():
    db = get_db()
    rows = db.execute(
        "SELECT id, filename, file_type, chunk_count, uploaded_at FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    db = get_db()
    row = db.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not row:
        return jsonify({"error": "Document not found"}), 404

    filename = row["filename"]

    # Remove chunks from ChromaDB that came from this file
    try:
        results = collection.get(where={"source": filename})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        logger.warning("Could not remove chunks for %s from ChromaDB", filename)

    # Remove from uploads folder
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.commit()

    return jsonify({"status": "deleted", "filename": filename})


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    db = get_db()

    # All counts fetched dynamically from their source
    total_kb_entries = collection.count()  # from ChromaDB
    total_user_queries = db.execute("SELECT COUNT(*) as c FROM queries").fetchone()["c"]
    unanswered_count = db.execute("SELECT COUNT(*) as c FROM queries WHERE source_docs = 0 AND answered_by_admin = 0").fetchone()["c"]
    total_users = db.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    avg_response = db.execute("SELECT AVG(response_time_ms) as avg FROM queries").fetchone()["avg"]
    avg_relevance = db.execute(
        "SELECT AVG(relevance_score) as avg FROM queries WHERE relevance_score >= 0"
    ).fetchone()["avg"]

    recent = db.execute(
        "SELECT question, response_time_ms, relevance_score, source_docs, timestamp "
        "FROM queries ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()

    top_questions = db.execute(
        "SELECT question, COUNT(*) as count FROM queries "
        "GROUP BY question ORDER BY count DESC LIMIT 10"
    ).fetchall()

    no_answer = db.execute(
        "SELECT question, timestamp FROM queries WHERE source_docs = 0 AND answered_by_admin = 0 "
        "ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()

    return jsonify({
        "summary": {
            "total_kb_entries": total_kb_entries,
            "total_user_queries": total_user_queries,
            "unanswered_count": unanswered_count,
            "total_users": total_users,
            "avg_response_time_ms": round(avg_response) if avg_response else 0,
            "avg_relevance_score": round(avg_relevance, 3) if avg_relevance else None,
        },
        "recent_queries": [dict(r) for r in recent],
        "top_questions": [dict(r) for r in top_questions],
        "unanswered_queries": [dict(r) for r in no_answer],
    })


@app.route("/api/kb", methods=["GET"])
def list_kb():
    """Return all documents stored in ChromaDB."""
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    results = collection.get(include=["documents", "metadatas"])
    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    total = len(ids)
    page_ids = ids[offset:offset + limit]
    page_docs = docs[offset:offset + limit]
    page_metas = metas[offset:offset + limit]

    entries = []
    for i, (doc_id, doc, meta) in enumerate(zip(page_ids, page_docs, page_metas)):
        entries.append({
            "id": doc_id,
            "content": doc,
            "source": meta.get("source", "unknown"),
            "type": meta.get("type", "unknown"),
        })

    return jsonify({"total": total, "offset": offset, "limit": limit, "entries": entries})


@app.route("/api/kb/<string:doc_id>", methods=["DELETE"])
def delete_kb_entry(doc_id):
    """Delete a single entry from ChromaDB by ID."""
    try:
        collection.delete(ids=[doc_id])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "deleted", "id": doc_id, "total_documents": collection.count()})


@app.route("/api/kb/clear", methods=["POST"])
def clear_kb():
    """Wipe all entries from ChromaDB collection."""
    try:
        results = collection.get()
        if results["ids"]:
            collection.delete(ids=results["ids"])
        logger.info("Knowledge base cleared — %d entries removed", len(results["ids"]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "cleared", "total_documents": collection.count()})


@app.route("/api/teach", methods=["POST"])
def teach():
    """Admin endpoint to add a Q&A pair directly into ChromaDB."""
    data = request.get_json(silent=True)
    if not data or not data.get("question", "").strip() or not data.get("answer", "").strip():
        return jsonify({"error": "Provide both 'question' and 'answer'"}), 400

    question = data["question"].strip()
    answer = data["answer"].strip()

    doc = f"Q: {question}\nA: {answer}"
    doc_id = f"admin_{uuid.uuid4().hex[:8]}"

    collection.add(
        documents=[doc],
        metadatas=[{"source": "admin", "type": "faq"}],
        ids=[doc_id],
    )

    # Mark all unanswered queries with this question as answered
    db = get_db()
    db.execute(
        "UPDATE queries SET answered_by_admin = 1 WHERE question = ? AND source_docs = 0",
        (question,),
    )
    db.commit()

    logger.info("Admin taught new FAQ: %s", question)
    return jsonify({
        "status": "added",
        "id": doc_id,
        "total_documents": collection.count(),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": MISTRAL_MODEL,
        "documents_in_kb": collection.count(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, port=port, threaded=True)
