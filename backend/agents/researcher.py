"""Researcher agent: searches, fetches and reads pages for one research branch."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from pydantic import BaseModel, Field

from backend import config
from backend.models import generate
from backend.rag import ingest
from backend.schemas import Branch, Experience, Query, Request, ResearchResult, TopicId
from backend.tools import search

TOPIC_LINES = "\n".join(
    f"{topic_id}: {label}" for topic_id, label in config.TOPICS.items()
)

ACCOUNT_PROMPT = f"""Does this page describe the writer's own interview for a software engineering job or internship?
If it is, give:
- the role;
- the year and month the interview took place, only if the page states them;
- the rounds in order;
- every topic asked, using only these topic IDs:
{TOPIC_LINES}"""


class Account(BaseModel):
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
    """Search the queries, fetch up to max_pages new pages, and read interview accounts on the interview branch.

    Supplied documents whose URL is not in `exclude` join the sources, so the workflow reads them once.
    """
    result = ResearchResult(
        branch=branch, sources=[s for s in request.supplied if s.url not in exclude]
    )

    found: dict[str, Query] = {}  # url -> the query that found it
    for query in queries:
        try:
            hits = search.search(query.text, config.RESULTS_PER_QUERY)
        except Exception as e:  # noqa: BLE001 - one failed search must not stop the branch
            result.errors.append(f"search {query.text!r}: {e}")
            continue
        for hit in hits:
            if hit["href"] not in exclude:
                found.setdefault(hit["href"], query)

    with ThreadPoolExecutor() as pool:
        pages = {url: pool.submit(search.fetch, url) for url in list(found)[:max_pages]}
    for url, page in pages.items():
        try:
            html = page.result()
        except Exception as e:  # noqa: BLE001 - one failed page must not stop the branch
            result.errors.append(f"fetch {url}: {e}")
            continue
        company = found[url].company
        source = ingest.from_html(
            url, html, company, search.tier(url, company), company != request.company
        )
        if source:
            result.sources.append(source)

    if branch == "interviews":
        for source in result.sources:
            account = generate(
                ACCOUNT_PROMPT, Account, source.text[: config.EXTRACT_CHARS]
            )
            if not account.is_interview_account:
                continue
            known = account.date_stated and account.event_year and account.event_month
            when = date(account.event_year, account.event_month, 1) if known else None
            result.experiences.append(
                Experience(
                    source_id=source.source_id,
                    company=source.company,
                    peer=source.peer,
                    role=account.role,
                    event_date=when,
                    rounds=account.rounds,
                    topic_ids=list(dict.fromkeys(account.topic_ids)),  # drop repeats
                )
            )
    return result
