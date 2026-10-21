"""Planner agent: writes web search queries for the evidence the workflow still needs."""

from pydantic import BaseModel

from backend import config
from backend.models import generate
from backend.schemas import Branch, Query, Request

QUESTIONS: dict[Branch, str] = {
    "company": "What does {company} build, and what recent news should a {role} candidate know?",
    "technology": "How are the technologies named in the job description used at {company}?",
    "interviews": "What interview rounds and coding topics does {company} use for {role} candidates?",
}


class Queries(BaseModel):
    queries: list[Query]


def plan(
    request: Request, companies: list[str], roles: list[str], needs: dict[Branch, str]
) -> list[Query]:
    """Up to QUERIES_PER_PASS queries for these needs, each tied to a company in scope."""
    need_lines = "\n".join(f"- {branch}: {need}" for branch, need in needs.items())
    prompt = f"""Write up to {config.QUERIES_PER_PASS} web search queries that find public evidence for these needs.
Each query has a branch from the list below and a company copied exactly from the company list.

Companies: {", ".join(companies)}
Roles: {", ".join(roles)}
Needs by branch:
{need_lines}

Job description:
{request.jd_text}"""
    queries = generate(prompt, Queries).queries
    valid = [q for q in queries if q.company in companies and q.branch in needs]
    return valid[: config.QUERIES_PER_PASS]
