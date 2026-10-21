"""Research questions, search queries per pass, and the evidence coverage check."""

from pydantic import BaseModel

from backend import config
from backend.models import generate, passages_text
from backend.schemas import Branch, Evidence, Query, Request

QUESTIONS: dict[Branch, str] = {
    "company": "What does {company} build, and what recent news should a {role} candidate know?",
    "technology": "How are the technologies named in the job description used at {company}?",
    "interviews": "What interview rounds and coding topics does {company} use for {role} candidates?",
}


class _Queries(BaseModel):
    queries: list[Query]


class CoverageCheck(BaseModel):
    answered: bool
    missing: str  # the evidence still needed; empty when answered


def research_questions(request: Request) -> dict[Branch, str]:
    """One evidence need per research branch."""
    return {
        branch: q.format(company=request.company, role=request.role)
        for branch, q in QUESTIONS.items()
    }


def scope(request: Request, pass_no: int) -> tuple[list[str], list[str]]:
    """Companies and roles to search: the target role, then related roles, then peer companies."""
    if pass_no == 1:
        return [request.company], [request.role]
    if pass_no == 2:
        related = "intern" if request.role_family == "new_grad" else "new_grad"
        return [request.company], config.ROLE_FAMILIES[related]
    group = config.COMPANIES.get(request.company, {}).get("group")
    peers = [
        name
        for name, c in config.COMPANIES.items()
        if c["group"] == group and name != request.company
    ]
    return peers, [request.role]


def search_queries(
    request: Request, pass_no: int, gaps: dict[Branch, str]
) -> list[Query]:
    """Pass 1 covers every branch; later passes widen the scope for the interview gap only."""
    companies, roles = scope(request, pass_no)
    if pass_no == 1:
        needs = research_questions(request)
    else:
        needs = {"interviews": gaps["interviews"]} if "interviews" in gaps else {}
    if not companies or not needs:
        return []

    need_lines = "\n".join(f"- {branch}: {need}" for branch, need in needs.items())
    prompt = f"""Write up to {config.QUERIES_PER_PASS} web search queries that find public evidence for these needs.
Each query has a branch from the list below and a company copied exactly from the company list.

Companies: {", ".join(companies)}
Roles: {", ".join(roles)}
Needs by branch:
{need_lines}

Job description:
{request.jd_text}"""
    queries = generate(prompt, _Queries).queries
    valid = [q for q in queries if q.company in companies and q.branch in needs]
    return valid[: config.QUERIES_PER_PASS]


def check_coverage(question: str, passages: list[Evidence]) -> CoverageCheck:
    """Ask whether the retrieved passages answer the question, and what is missing if not."""
    if not passages:
        return CoverageCheck(answered=False, missing=question)
    prompt = f"""Do the passages answer this question with specific facts?
Question: {question}
If they do not, state in one sentence what evidence is still missing; otherwise leave missing empty."""
    return generate(prompt, CoverageCheck, passages_text(passages))
