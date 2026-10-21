"""Company, technology and interview-experience research: search, fetch, and read interview accounts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from pydantic import BaseModel, Field

from backend import config
from backend.models import generate
from backend.rag import ingest
from backend.schemas import (
    Branch,
    Experience,
    Query,
    Request,
    ResearchResult,
    Source,
    TopicId,
)
from backend.tools import search

TOPIC_LINES = "\n".join(
    f"{topic_id}: {label}" for topic_id, label in config.TOPICS.items()
)


class _Account(BaseModel):
    is_interview_account: bool
    role: str
    date_stated: bool  # decided before the numbers, so the model does not invent a date
    event_year: int | None = Field(ge=2000, le=date.today().year)
    event_month: int | None = Field(ge=1, le=12)
    rounds: list[str]
    topic_ids: list[TopicId]


def research(
    branch: Branch,
    queries: list[Query],
    request: Request,
    exclude: set[str],
    max_pages: int,
) -> ResearchResult:
    """Run one branch: search its queries, fetch up to max_pages new pages, read interview accounts."""
    result = ResearchResult(branch=branch)

    targets: dict[str, Query] = {}  # url -> the query that found it
    for query in queries:
        try:
            hits = search.search(query.text, config.RESULTS_PER_QUERY)
        except Exception as e:  # noqa: BLE001 - a failed search must not stop the branch
            result.errors.append(f"search {query.text!r}: {e}")
            continue
        for hit in hits:
            if hit["href"] not in exclude:
                targets.setdefault(hit["href"], query)

    urls = list(targets)[:max_pages]
    with ThreadPoolExecutor() as pool:
        pages = list(pool.map(_fetch, urls))
    for url, page in zip(urls, pages):
        if isinstance(page, Exception):
            result.errors.append(f"fetch {url}: {page}")
            continue
        company = targets[url].company
        source = ingest.from_html(
            url,
            page,
            company=company,
            tier=search.tier(url, company),
            peer=company != request.company,
        )
        if source:
            result.sources.append(source)

    if branch == "interviews":
        result.experiences = [e for s in result.sources if (e := read_account(s))]
    return result


def read_account(source: Source) -> Experience | None:
    """Return the interview account on a page, or None if the page is not a first-hand account."""
    prompt = f"""Does this page describe the writer's own interview for a software engineering job or internship?
If it is, give:
- the role;
- the year and month the interview took place, only if the page states them;
- the rounds in order;
- every topic asked, using only these topic IDs:
{TOPIC_LINES}"""
    account = generate(prompt, _Account, source.text[: config.EXTRACT_CHARS])
    if not account.is_interview_account:
        return None
    month_known = account.date_stated and account.event_year and account.event_month
    return Experience(
        source_id=source.source_id,
        company=source.company,
        peer=source.peer,
        role=account.role,
        event_date=date(account.event_year, account.event_month, 1)
        if month_known
        else None,
        rounds=account.rounds,
        topic_ids=list(dict.fromkeys(account.topic_ids)),  # drop repeats, keep order
    )


def _fetch(url: str) -> str | Exception:
    """Fetch one page; a failure is returned so one bad source does not stop the branch."""
    try:
        return search.fetch(url)
    except Exception as e:  # noqa: BLE001 - returned to the caller as an error
        return e
