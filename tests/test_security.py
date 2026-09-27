import pytest

from app.config import settings
from app.validation import ensure_within, safe_filename, validate_collection_name
from tests.conftest import VACATION_MD, upload


@pytest.mark.parametrize("name", ["../../etc", "..", "a/b", "a\\b", ".hidden", "ab", "x" * 64, "bad name", "ünïcode"])
def test_collection_name_rejects_paths_and_invalid_names(name):
    with pytest.raises(ValueError):
        validate_collection_name(name)


@pytest.mark.parametrize("name", ["handbook", "hr-policies_2025", "abc"])
def test_collection_name_accepts_simple_names(name):
    assert validate_collection_name(name) == name


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../secret.md", "secret.md"),
        ("..\\..\\win.txt", "win.txt"),
        ("/etc/passwd", "passwd"),
        (".env", "env"),
        ("my report (final).pdf", "my_report_final_.pdf"),
        ("", "upload"),
    ],
)
def test_safe_filename_strips_directories(raw, expected):
    assert safe_filename(raw) == expected


def test_ensure_within_blocks_escape(tmp_path):
    with pytest.raises(ValueError):
        ensure_within(tmp_path / "docs", tmp_path / "docs" / ".." / "outside")


def test_upload_rejects_traversal_in_collection_name(client, isolated_storage):
    response = upload(client, "a.md", VACATION_MD, collection="../../escape")
    assert response.status_code == 422
    assert not (isolated_storage / "escape").exists()


def test_upload_filename_cannot_escape_collection_folder(client, isolated_storage):
    response = upload(client, "../../../evil.md", VACATION_MD)
    assert response.status_code == 201
    saved = isolated_storage / "documents" / "handbook"
    assert [p.name for p in saved.rglob("*") if p.is_file()] == ["evil.md"]
    assert [p for p in isolated_storage.rglob("evil.md")] == list(saved.rglob("evil.md"))


def test_move_rejects_traversal_in_target_collection(client):
    doc_id = upload(client, "a.md", VACATION_MD).json()["id"]
    response = client.put(f"/api/documents/{doc_id}", json={"collection_name": "../outside"})
    assert response.status_code == 422


def test_unsupported_extension_is_a_400(client):
    response = upload(client, "script.sh", b"rm -rf /")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_size_limit(client, monkeypatch, isolated_storage):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    response = upload(client, "big.md", VACATION_MD)
    assert response.status_code == 400
    assert not list((isolated_storage / "documents" / "handbook").rglob("*"))


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "s3cret")
    assert client.get("/api/collections").status_code == 401
    assert client.get("/api/collections", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/collections", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays public


def test_internal_errors_do_not_leak_details(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services.query_service import QueryService

    def explode(self, request):
        raise RuntimeError("/home/secret/path leaked")

    monkeypatch.setattr(QueryService, "query", explode)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/query", json={"query_text": "anything"})
    assert response.status_code == 500
    assert "secret" not in response.text
