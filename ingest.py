import os
import sys

import chromadb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "faq")
FAQ_FILE = "faqs.csv"


def ingest():
    if not os.path.exists(FAQ_FILE):
        print(f"Error: {FAQ_FILE} not found")
        sys.exit(1)

    df = pd.read_csv(FAQ_FILE)

    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must have columns: {required_cols}")
        sys.exit(1)

    df = df.dropna(subset=["question", "answer"])

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Reset collection for fresh ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        documents=df["question"].tolist(),
        metadatas=[{"answer": a} for a in df["answer"].tolist()],
        ids=[f"faq_{i}" for i in range(len(df))],
    )

    print(f"Done — {len(df)} FAQs loaded into ChromaDB ({CHROMA_PATH})")


if __name__ == "__main__":
    ingest()
