import os
import sys
import uuid

import chromadb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "knowledge")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))


def get_chroma_collection(client, name):
    """Get or create a ChromaDB collection."""
    return client.get_or_create_collection(name)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by sentences."""
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap by taking last portion
            words = current.split()
            overlap_text = " ".join(words[-overlap // 4:]) if len(words) > overlap // 4 else ""
            current = overlap_text + " " + sentence
        else:
            current = (current + ". " + sentence) if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def ingest_csv(filepath: str, collection, source: str = "csv") -> int:
    """Ingest a CSV file. Supports Q&A format (question,answer) or generic text columns."""
    df = pd.read_csv(filepath)
    df = df.dropna()

    docs = []
    metadatas = []

    if {"question", "answer"}.issubset(df.columns):
        for _, row in df.iterrows():
            docs.append(f"Q: {row['question']}\nA: {row['answer']}")
            metadatas.append({"source": source, "type": "faq"})
    else:
        # Combine all text columns
        for _, row in df.iterrows():
            text = " | ".join(str(v) for v in row.values if pd.notna(v))
            docs.append(text)
            metadatas.append({"source": source, "type": "data"})

    if not docs:
        return 0

    ids = [f"{source}_{uuid.uuid4().hex[:8]}" for _ in docs]
    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    return len(docs)


def ingest_pdf(filepath: str, collection, source: str = "pdf") -> int:
    """Extract text from PDF and ingest as chunks."""
    from PyPDF2 import PdfReader

    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    if not full_text.strip():
        raise ValueError("PDF contains no extractable text")

    chunks = chunk_text(full_text)
    ids = [f"{source}_{uuid.uuid4().hex[:8]}" for _ in chunks]
    metadatas = [{"source": source, "type": "pdf", "chunk": i} for i in range(len(chunks))]

    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def ingest_text(filepath: str, collection, source: str = "txt") -> int:
    """Ingest a plain text file as chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    if not full_text.strip():
        raise ValueError("Text file is empty")

    chunks = chunk_text(full_text)
    ids = [f"{source}_{uuid.uuid4().hex[:8]}" for _ in chunks]
    metadatas = [{"source": source, "type": "text", "chunk": i} for i in range(len(chunks))]

    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def seed_faqs():
    """Seed the knowledge base with the default FAQs CSV."""
    faq_file = "uploads/faqs.csv"
    if not os.path.exists(faq_file):
        print(f"Warning: {faq_file} not found — skipping seed. Upload CSV from the admin panel.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll = get_chroma_collection(client, COLLECTION_NAME)

    # Only seed if collection is empty
    if coll.count() > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {coll.count()} documents, skipping seed.")
        return

    count = ingest_csv(faq_file, coll, source="uploads/faqs.csv")
    print(f"Done — {count} FAQs loaded into ChromaDB ({CHROMA_PATH})")


if __name__ == "__main__":
    seed_faqs()
