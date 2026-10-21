"""Turn fetched pages and supplied text into deduplicated sources and citable chunks."""

import hashlib
import json
from datetime import date

import trafilatura

from backend import config
from backend.schemas import Evidence, Source, Tier


def from_html(
    url: str, html: str, company: str, tier: Tier, peer: bool
) -> Source | None:
    """Main text, title and publication date of a page; None when it has no readable text."""
    extracted = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True
    )
    if not extracted:
        return None
    page = json.loads(extracted)
    return Source(
        source_id=_hash(url),
        url=url,
        title=page["title"] or url,
        text=page["text"],
        published=date.fromisoformat(page["date"]) if page["date"] else None,
        fetched=date.today(),
        tier=tier,
        company=company,
        peer=peer,
    )


def from_text(
    text: str,
    title: str,
    company: str,
    published: date | None = None,
    peer: bool = False,
) -> Source:
    """A pasted or uploaded document supplied with the request."""
    source_id = _hash(text)
    return Source(
        source_id=source_id,
        url=f"supplied:{source_id}",
        title=title,
        text=text,
        published=published,
        fetched=date.today(),
        tier="supplied",
        company=company,
        peer=peer,
    )


def dedupe(sources: list[Source]) -> list[Source]:
    """Keep the first source per normalized text, so exact cross-posts do not count twice."""
    unique: dict[str, Source] = {}
    for source in sources:
        unique.setdefault(_hash(" ".join(source.text.lower().split())), source)
    return list(unique.values())


def chunk(source: Source, event_date: date | None = None) -> list[Evidence]:
    """Overlapping word windows of about CHUNK_TOKENS tokens each."""
    # shortcut: ~0.75 words per token; count real tokens if the embedder starts truncating
    size = int(config.CHUNK_TOKENS * 0.75)
    overlap = int(config.CHUNK_OVERLAP_TOKENS * 0.75)
    words = source.text.split()
    starts = range(0, max(len(words) - overlap, 1), size - overlap)
    return [
        Evidence(
            chunk_id=f"{source.source_id}-{n}",
            source_id=source.source_id,
            url=source.url,
            title=source.title,
            excerpt=" ".join(words[start : start + size]),
            published=source.published,
            event_date=event_date,
            tier=source.tier,
            company=source.company,
            peer=source.peer,
        )
        for n, start in enumerate(starts)
    ]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]
