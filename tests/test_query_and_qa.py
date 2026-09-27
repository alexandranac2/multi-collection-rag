from tests.conftest import SETUP_TXT, VACATION_MD, upload


def test_query_returns_best_matching_chunk_with_citation(client):
    upload(client, "vacation.md", VACATION_MD)
    upload(client, "setup.txt", SETUP_TXT, collection="manuals", doc_type="manual")

    body = client.post("/api/query", json={"query_text": "how many vacation days per year", "include_qa": False}).json()
    [best] = body["results"]
    assert "vacation" in best["text"].lower()
    citation = best["citations"][0]
    assert citation["filename"] == "vacation.md"
    assert citation["collection"] == "handbook"


def test_query_filters_by_collection_and_doc_type(client):
    upload(client, "vacation.md", VACATION_MD)
    upload(client, "setup.txt", SETUP_TXT, collection="manuals", doc_type="manual")

    for payload in ({"collections": ["manuals"]}, {"doc_types": ["manual"]}):
        body = client.post(
            "/api/query",
            json={"query_text": "vacation days", "include_qa": False, "return_all": True, **payload},
        ).json()
        assert body["results"]
        assert {r["citations"][0]["collection"] for r in body["results"]} == {"manuals"}


def test_curated_qa_competes_with_documents(client):
    upload(client, "setup.txt", SETUP_TXT)
    qa = client.post(
        "/api/qa",
        json={
            "question": "How many vacation days do I get?",
            "answer": "25 per year.",
            "tags": ["hr", "leave"],
        },
    ).json()

    body = client.post("/api/query", json={"query_text": "vacation days do I get"}).json()
    [best] = body["results"]
    assert best["source_type"] == "qa"
    assert best["citations"][0]["qa_id"] == qa["id"]
    assert best["citations"][0]["tags"] == ["hr", "leave"]


def test_qa_crud_and_pagination(client):
    ids = [client.post("/api/qa", json={"question": f"Q{i}?", "answer": f"A{i}"}).json()["id"] for i in range(3)]

    page = client.get("/api/qa", params={"limit": 2, "offset": 0}).json()
    assert len(page["qa_pairs"]) == 2 and page["total"] == 3

    assert client.get(f"/api/qa/{ids[0]}").json()["answer"] == "A0"
    assert client.delete(f"/api/qa/{ids[0]}").status_code == 204
    assert client.delete(f"/api/qa/{ids[0]}").status_code == 404  # used to return 204
    assert client.get(f"/api/qa/{ids[0]}").status_code == 404


def test_query_on_empty_system_returns_no_results(client):
    assert client.post("/api/query", json={"query_text": "anything"}).json()["results"] == []


def test_validation(client):
    assert client.post("/api/query", json={"query_text": ""}).status_code == 422
    assert client.post("/api/query", json={"query_text": "x", "doc_types": ["nope"]}).status_code == 422
    assert client.get("/api/qa", params={"limit": 0}).status_code == 422
