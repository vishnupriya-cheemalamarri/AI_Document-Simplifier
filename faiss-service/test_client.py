"""
Quick smoke test for the FAISS service.
Uses random vectors so it has no dependency on an embedding model -
swap in real embeddings from your RAG pipeline when wiring this up.

Usage:
    pip install requests numpy
    python test_client.py
"""

import numpy as np
import requests

BASE_URL = "http://localhost:8000"
DIM = 384  # match your real embedding model's output size, e.g. 384 for all-MiniLM-L6-v2


def main() -> None:
    print("1) health check")
    r = requests.get(f"{BASE_URL}/health")
    r.raise_for_status()
    print("  ", r.json())

    print("\n2) adding 3 documents")
    rng = np.random.default_rng(42)
    vectors = rng.random((3, DIM)).astype("float32").tolist()
    payload = {
        "ids": [1, 2, 3],
        "embeddings": vectors,
        "metadatas": [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "FAISS is a library for efficient similarity search."},
            {"text": "Docker Compose runs multi-container applications."},
        ],
    }
    r = requests.post(f"{BASE_URL}/add", json=payload)
    r.raise_for_status()
    print("  ", r.json())

    print("\n3) searching with a query close to vector #2")
    query = (np.array(vectors[1]) + rng.normal(0, 0.01, DIM)).astype("float32").tolist()
    r = requests.post(f"{BASE_URL}/search", json={"embedding": query, "k": 3})
    r.raise_for_status()
    for hit in r.json():
        print("  ", hit)

    print("\n4) stats")
    r = requests.get(f"{BASE_URL}/stats")
    r.raise_for_status()
    print("  ", r.json())


if __name__ == "__main__":
    main()
