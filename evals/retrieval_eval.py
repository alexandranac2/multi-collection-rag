"""
Retrieval eval: labelled questions over a small fictional corpus (handbook,
device manual, contract), comparing the two chunking strategies.

    python -m evals.retrieval_eval                 # real OpenAI embeddings (needs OPENAI_API_KEY, ~$0.001)
    python -m evals.retrieval_eval --fake-embeddings  # offline smoke test of the harness only

Metrics per strategy:
  file hit@1 / hit@3  the right document is ranked 1st / in the top 3
  answer@1 / answer@3 a chunk containing the expected answer text is ranked 1st / in the top 3
  MRR                 mean reciprocal rank of the first chunk containing the answer
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from rag_package import MultiCollectionRAG

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
K = 3


def build(strategy: str, workdir: Path, fake: bool) -> MultiCollectionRAG:
    rag = MultiCollectionRAG(
        chroma_persist_dir=str(workdir / f"chroma-{strategy}"),
        cache_dir=str(workdir / f"state-{strategy}"),
        openai_api_key="unused" if fake else None,
    )
    if fake:
        from tests.conftest import fake_embed

        rag._embed_texts = fake_embed
    for folder, doc_type in (("policies", "policy"), ("manuals", "manual"), ("contracts", "contract")):
        rag.add_collection(f"eval-{folder}", str(CORPUS / folder), doc_type=doc_type, chunking_strategy=strategy)
        stats = rag.ingest_collection(f"eval-{folder}")
        if stats["failed"]:
            raise RuntimeError(f"{strategy}: failed to ingest {folder}: {stats}")
    return rag


def evaluate(rag: MultiCollectionRAG, cases) -> dict:
    totals = dict.fromkeys(["file@1", "file@3", "answer@1", "answer@3", "mrr"], 0.0)
    chunks = sum(c.count() for c in rag.collections.values())
    for case in cases:
        hits = rag.query_combined(case["query"], n_results=K)
        files = [h["metadata"]["filename"] for h in hits]
        answer_rank = next(
            (i for i, h in enumerate(hits, 1) if case["answer"].lower() in h["text"].lower()),
            None,
        )
        totals["file@1"] += files[:1] == [case["file"]]
        totals["file@3"] += case["file"] in files
        totals["answer@1"] += answer_rank == 1
        totals["answer@3"] += answer_rank is not None
        totals["mrr"] += 1 / answer_rank if answer_rank else 0
    return {"chunks": chunks, **{k: v / len(cases) for k, v in totals.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-embeddings", action="store_true", help="offline harness check; scores are meaningless")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cases = json.loads((HERE / "queries.json").read_text())
    workdir = Path(tempfile.mkdtemp())
    try:
        print(f"{len(cases)} questions{' (FAKE embeddings)' if args.fake_embeddings else ''}")
        print("strategy      chunks  file@1  file@3  answer@1  answer@3   MRR")
        for strategy in ("hybrid", "hierarchical"):
            m = evaluate(build(strategy, workdir, args.fake_embeddings), cases)
            print(
                f"{strategy:<12}  {m['chunks']:>6}  {m['file@1']:>6.2f}  {m['file@3']:>6.2f}"
                f"  {m['answer@1']:>8.2f}  {m['answer@3']:>8.2f}  {m['mrr']:>5.2f}"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
