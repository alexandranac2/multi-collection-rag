from tests.conftest import SETUP_TXT, VACATION_MD, upload


def test_upload_indexes_markdown_and_reports_its_own_chunks(client):
    body = upload(client, "vacation.md", VACATION_MD, title="Vacation").json()
    assert body["chunk_count"] > 0
    assert body["collection"] == "handbook"
    assert body["title"] == "Vacation"

    # A second document must not change the first one's chunk count
    upload(client, "setup.txt", SETUP_TXT)
    assert client.get(f"/api/documents/{body['id']}").json()["chunk_count"] == body["chunk_count"]


def test_same_stem_different_extension_do_not_collide(client, isolated_storage):
    # Chunk IDs used to be <collection>_<stem>_<i>, so notes.md and notes.txt overwrote each other
    from rag_package import MultiCollectionRAG

    folder = isolated_storage / "docs"
    folder.mkdir()
    (folder / "notes.md").write_bytes(VACATION_MD)
    (folder / "notes.txt").write_bytes(SETUP_TXT)

    rag = MultiCollectionRAG(chroma_persist_dir=str(isolated_storage / "c2"), cache_dir=str(isolated_storage / "s2"))
    rag.add_collection("notes", str(folder))
    stats = rag.ingest_collection("notes")
    assert stats == {"indexed": 2, "skipped": 0, "failed": 0}
    coll = rag.collections["notes"]
    assert coll.count_source(folder / "notes.md") > 0
    assert coll.count_source(folder / "notes.txt") > 0

    # Unchanged files are skipped on the next run (SHA-256 cache)
    assert rag.ingest_collection("notes") == {"indexed": 0, "skipped": 2, "failed": 0}


def test_list_filter_and_collection_stats(client):
    upload(client, "vacation.md", VACATION_MD)
    upload(client, "setup.txt", SETUP_TXT, collection="manuals", doc_type="manual")

    assert client.get("/api/documents").json()["total"] == 2
    assert [
        d["filename"] for d in client.get("/api/documents", params={"collection": "manuals"}).json()["documents"]
    ] == ["setup.txt"]

    handbook = client.get("/api/collections/handbook").json()
    assert handbook["document_count"] == 1
    assert handbook["chunk_count"] > 0


def test_move_document_between_collections(client):
    doc = upload(client, "vacation.md", VACATION_MD).json()
    moved = client.put(f"/api/documents/{doc['id']}", json={"collection_name": "archive", "title": "Old"}).json()

    assert moved["collection"] == "archive"
    assert moved["title"] == "Old"
    assert moved["chunk_count"] == doc["chunk_count"]
    assert client.get("/api/collections/handbook").json()["chunk_count"] == 0


def test_delete_removes_file_chunks_and_entry(client, isolated_storage):
    doc = upload(client, "vacation.md", VACATION_MD).json()
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert not list((isolated_storage / "documents" / "handbook").rglob("*"))
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404
    assert client.get("/api/collections/handbook").json()["chunk_count"] == 0
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_collections_survive_restart_with_their_settings(isolated_storage):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as first:
        created = first.post(
            "/api/collections",
            json={
                "collection_name": "contracts",
                "doc_type": "contract",
                "chunk_size": 900,
                "chunking_strategy": "hybrid",
            },
        )
        assert created.status_code == 201
        assert first.post("/api/collections", json={"collection_name": "contracts"}).status_code == 409

    import app.main as main

    with TestClient(app):
        reloaded = main.rag_instance.collections["contracts"]
        assert (reloaded.doc_type, reloaded.chunk_size, reloaded.chunking_strategy) == (
            "contract",
            900,
            "hybrid",
        )
