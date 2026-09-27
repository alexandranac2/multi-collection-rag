"""
Tests run against real Chroma and real Docling, with OpenAI embeddings replaced
by a deterministic bag-of-words hash, so no API key or network is needed.
"""

import hashlib
import math
import os
import re

# `or`, not setdefault: CI can export the variable as an empty string
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "test-key-not-used"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from rag_package import MultiCollectionRAG  # noqa: E402

DIM = 256


def fake_embed(texts):
    """Unit vectors from hashed word counts: texts sharing words are close in cosine space."""
    vectors = []
    for text in texts:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own Chroma dir, documents dir and state dir."""
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "DOCUMENTS_BASE_PATH", str(tmp_path / "documents"))
    monkeypatch.setattr(settings, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(settings, "API_KEY", None)
    monkeypatch.setattr(MultiCollectionRAG, "_embed_texts", lambda self, texts: fake_embed(texts))
    return tmp_path


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:  # runs the real lifespan
        yield test_client


def upload(client, name, content, collection="handbook", doc_type="policy", **extra):
    return client.post(
        "/api/documents",
        files={"file": (name, content, "application/octet-stream")},
        data={"folder_name": collection, "doc_type": doc_type, **extra},
    )


VACATION_MD = b"""# Vacation policy

Employees receive 25 vacation days per year. Unused vacation days carry over
until March 31 of the following year.

# Expense policy

Travel expenses are reimbursed within 30 days when receipts are attached.
"""

SETUP_TXT = b"""Installation guide.
Run the installer, accept the licence, then restart the device to finish setup.
"""
