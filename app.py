import logging
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import chromadb
import requests

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
CORS(app)

limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

# --- Config --------------------------------------------------------------- #

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set. Add it to .env or export it.")

MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-medium-latest")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "faq")
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "1.2"))

# --- ChromaDB ------------------------------------------------------------- #

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception:
    raise RuntimeError(
        f"ChromaDB collection '{COLLECTION_NAME}' not found. Run ingest.py first."
    )

# --- Mistral -------------------------------------------------------------- #

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant. "
    "Answer concisely based on the provided FAQ context. "
    "If the context doesn't contain the answer, say so honestly."
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


def retrieve_context(question: str) -> str | None:
    results = collection.query(
        query_texts=[question],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    context_parts = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > RELEVANCE_THRESHOLD:
            continue
        context_parts.append(f"Q: {doc}\nA: {meta['answer']}")

    return "\n\n".join(context_parts) if context_parts else None


def build_messages(context: str | None, chat_history: list[dict], question: str) -> list[dict]:
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\nFAQ Context:\n{context}"
    else:
        system_content += (
            "\n\nNo relevant FAQ was found. Let the user know you don't have "
            "that information and suggest they contact support@shop.com or "
            "call 1800-123-456."
        )

    messages = [{"role": "system", "content": system_content}]

    # Keep last 6 turns of history to stay within token limits
    messages.extend(chat_history[-12:])
    messages.append({"role": "user", "content": question})

    return messages


def ask(question: str, chat_history: list[dict]) -> str:
    context = retrieve_context(question)
    messages = build_messages(context, chat_history, question)
    return call_mistral(messages)


# --- Routes --------------------------------------------------------------- #


@app.route("/")
def index():
    return render_template("index.html")


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

    # Session-based conversation history
    if "history" not in session:
        session["history"] = []

    try:
        answer = ask(question, session["history"])
    except Exception:
        logger.exception("Failed to generate answer")
        return jsonify({"error": "Something went wrong, please try again later"}), 500

    # Update history
    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})
    session.modified = True

    return jsonify({"question": question, "answer": answer})


@app.route("/reset", methods=["POST"])
def reset_chat():
    session.pop("history", None)
    return jsonify({"status": "Chat history cleared"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "model": MISTRAL_MODEL, "faqs_loaded": collection.count()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, port=port, threaded=True)
