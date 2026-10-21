"""Small in-memory hybrid index (cosine + BM25, fused by reciprocal rank) and topic support counts."""

from dataclasses import dataclass
from datetime import date

import numpy as np
from rank_bm25 import BM25Okapi

from backend import config
from backend.rag.embeddings import embed
from backend.schemas import Evidence, Experience, RankedTopic


@dataclass
class Index:
    chunks: list[Evidence]
    vectors: np.ndarray
    bm25: BM25Okapi | None


def build(chunks: list[Evidence]) -> Index:
    """Embed and keyword-index one run's chunks."""
    if not chunks:
        return Index([], np.zeros((0, 0)), None)
    vectors = embed([c.excerpt for c in chunks], "document")
    return Index(chunks, vectors, BM25Okapi([_words(c.excerpt) for c in chunks]))


def search(
    index: Index, query: str, companies: set[str], as_of: date
) -> list[Evidence]:
    """Top passages about `companies`, published by `as_of`; the same filter applies to both paths."""
    eligible = [
        c.company in companies and (c.published is None or c.published <= as_of)
        for c in index.chunks
    ]
    if not any(eligible):
        return []
    fused = np.zeros(len(index.chunks))
    for scores in (
        index.vectors @ embed([query], "query")[0],
        index.bm25.get_scores(_words(query)),
    ):
        ordered = [i for i in np.argsort(-scores) if eligible[i]]
        candidates = ordered[: config.CANDIDATES_PER_PATH]
        for rank, i in enumerate(candidates, 1):
            fused[i] += 1 / (config.RRF_K + rank)
    best = np.argsort(-fused)[: config.TOP_PASSAGES]
    return [index.chunks[i] for i in best if fused[i] > 0]


def rank_topics(
    experiences: list[Experience], corpus: dict[str, Evidence]
) -> list[RankedTopic]:
    """Topics ordered by how many accounts mention them, each citing the first chunk of those accounts."""
    first_chunk: dict[str, str] = {}
    for chunk_id, chunk in corpus.items():
        first_chunk.setdefault(chunk.source_id, chunk_id)

    accounts: dict[str, list[Experience]] = {}
    for experience in experiences:
        for topic_id in experience.topic_ids:
            accounts.setdefault(topic_id, []).append(experience)
    by_count = sorted(accounts.items(), key=lambda item: -len(item[1]))
    ranked = by_count[: config.TOP_TOPICS]

    total = len(experiences)
    return [
        RankedTopic(
            topic_id=topic_id,
            rank=rank,
            accounts_with_topic=len(found),
            accounts_total=total,
            confidence_reason=_reason(found, total),
            citations=[
                first_chunk[e.source_id] for e in found if e.source_id in first_chunk
            ],
        )
        for rank, (topic_id, found) in enumerate(ranked, 1)
    ]


def _reason(found: list[Experience], total: int) -> str:
    peers = sum(e.peer for e in found)
    reason = f"observed in {len(found)} of {total} independent accounts"
    return f"{reason}, {peers} from peer companies" if peers else reason


def _words(text: str) -> list[str]:
    return text.lower().split()
