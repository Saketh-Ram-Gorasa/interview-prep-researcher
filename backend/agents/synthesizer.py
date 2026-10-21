"""The six-section briefing: cited claims with support checks, resume review and a study schedule."""

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel

from backend.models import generate, passages_text
from backend.schemas import (
    Branch,
    Claim,
    Evidence,
    Gap,
    RankedTopic,
    Report,
    Request,
    ResumeQuestion,
    Round,
    RunInfo,
    StudyDay,
)

CITE_RULES = """Each claim lists in `citations` the IDs of the passages that state it.
Use kind "observed" only for what a passage states, "inferred" for your conclusions and "recommendation" for advice.
Passages marked peer_company describe a different company; name that company in the claim."""


class _Snapshot(BaseModel):
    claims: list[Claim]
    questions_to_ask: list[str]


class _Rounds(BaseModel):
    rounds: list[Round]


class _ResumeReview(BaseModel):
    questions: list[ResumeQuestion]
    gaps: list[Gap]


class _Support(BaseModel):
    labels: list[Literal["supported", "partial", "unsupported"]]


def write_report(
    request: Request,
    passages: dict[Branch, list[Evidence]],
    topics: list[RankedTopic],
    missing: list[str],
    run: RunInfo,
    status: Literal["complete", "partial"],
) -> Report:
    """Write every section, check claim support, and assemble the report."""
    evidence = {p.chunk_id: p for group in passages.values() for p in group}
    snapshot = _company_snapshot(request, passages["company"] + passages["technology"])
    rounds = _interview_rounds(request, passages["interviews"])
    _check_support(snapshot.claims, evidence)
    _check_support([c for r in rounds for c in r.details], evidence)
    review = _review_resume(request)

    return Report(
        company=request.company,
        role=request.role,
        as_of=request.as_of,
        status=status,
        company_snapshot=[c for c in snapshot.claims if _keep(c)],
        interview_rounds=[
            Round(name=r.name, details=kept)
            for r in rounds
            if (kept := [c for c in r.details if _keep(c)])
        ],
        ranked_topics=topics,
        resume_questions=review.questions,
        gaps=review.gaps,
        schedule=_schedule(request, topics),
        questions_to_ask=snapshot.questions_to_ask,
        missing_evidence=missing,
        evidence=evidence,
        run=run,
    )


def _company_snapshot(request: Request, passages: list[Evidence]) -> _Snapshot:
    if not passages:
        return _Snapshot(claims=[], questions_to_ask=[])
    prompt = f"""Write a company snapshot of {request.company} for a {request.role} candidate:
what it builds, recent news, and how the job's technologies are used there.
Then suggest questions the candidate could ask the interviewers.
{CITE_RULES}

Job description:
{request.jd_text}"""
    return generate(prompt, _Snapshot, passages_text(passages))


def _interview_rounds(request: Request, passages: list[Evidence]) -> list[Round]:
    if not passages:
        return []
    prompt = f"""List the interview rounds for {request.role} candidates at {request.company} in the order they happen.
For each round, give claims about its format and content.
{CITE_RULES}"""
    return generate(prompt, _Rounds, passages_text(passages)).rounds


def _review_resume(request: Request) -> _ResumeReview:
    """Deep-dive questions on resume items, and JD requirements the resume does or does not show."""
    if not request.resume_text:
        return _ResumeReview(questions=[], gaps=[])
    prompt = f"""Compare the resume with the job description.
1. Write up to 5 deep-dive questions an interviewer would ask about specific resume items.
2. List the main job requirements and whether the resume shows each one.
   Use "not_shown_in_resume" when the resume does not mention it; never claim the candidate lacks a skill.

Job description:
{request.jd_text}

Resume:
{request.resume_text}"""
    return generate(prompt, _ResumeReview)


def _check_support(claims: list[Claim], evidence: dict[str, Evidence]) -> None:
    """Resolve citation IDs, mark peer-only claims as inferred, then label support in one call."""
    for c in claims:
        c.citations = [i for i in c.citations if i in evidence]
        c.support = "unchecked"
        if c.citations and all(evidence[i].peer for i in c.citations):
            c.kind = "inferred"  # peer evidence cannot support an observed claim about the target

    cited = [c for c in claims if c.citations]
    if not cited:
        return
    listing = "\n\n".join(
        f"Claim {n}: {c.text}\nExcerpts:\n"
        + "\n".join(f"- {evidence[i].excerpt}" for i in c.citations)
        for n, c in enumerate(cited, 1)
    )
    prompt = 'For each claim, in order, label whether its excerpts support it: "supported", "partial" or "unsupported".'
    for c, label in zip(cited, generate(prompt, _Support, listing).labels):
        c.support = label


def _keep(claim: Claim) -> bool:
    """Drop unsupported claims and observed claims left without a valid citation."""
    return claim.support != "unsupported" and (
        claim.kind != "observed" or bool(claim.citations)
    )


def _schedule(request: Request, topics: list[RankedTopic]) -> list[StudyDay]:
    """One topic per day in rank order, cycling back to the top when days outnumber topics."""
    if not request.interview_date or not topics:
        return []
    ranked = [t.topic_id for t in sorted(topics, key=lambda t: t.rank)]
    days = (request.interview_date - request.as_of).days
    return [
        StudyDay(
            day=request.as_of + timedelta(days=i),
            minutes=request.daily_study_minutes,
            topic_ids=[ranked[i % len(ranked)]],
        )
        for i in range(days)
    ]
