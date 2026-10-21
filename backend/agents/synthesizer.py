"""Synthesizer agent: writes the six-section briefing with checked citations."""

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

SUPPORT_PROMPT = 'For each claim, in order, label whether its excerpts support it: "supported", "partial" or "unsupported".'


class Snapshot(BaseModel):
    claims: list[Claim]
    questions_to_ask: list[str]


class Rounds(BaseModel):
    rounds: list[Round]


class ResumeReview(BaseModel):
    questions: list[ResumeQuestion]
    gaps: list[Gap]


class Support(BaseModel):
    labels: list[Literal["supported", "partial", "unsupported"]]


def write_report(
    request: Request,
    passages: dict[Branch, list[Evidence]],
    evidence: dict[str, Evidence],
    topics: list[RankedTopic],
    missing: list[str],
    run: RunInfo,
    status: Literal["complete", "partial"],
) -> Report:
    """Write every section, check claim support, and assemble the report.

    `passages` are what each section's prompt sees; `evidence` is every chunk the report may cite.
    """
    # 1. Company snapshot and questions to ask
    snapshot = Snapshot(claims=[], questions_to_ask=[])
    if company_passages := passages["company"] + passages["technology"]:
        prompt = f"""Write a company snapshot of {request.company} for a {request.role} candidate:
what it builds, recent news, and how the job's technologies are used there.
Then suggest questions the candidate could ask the interviewers.
{CITE_RULES}

Job description:
{request.jd_text}"""
        snapshot = generate(prompt, Snapshot, passages_text(company_passages))

    # 2. Interview rounds
    rounds: list[Round] = []
    if passages["interviews"]:
        prompt = f"""List the interview rounds for {request.role} candidates at {request.company} in the order they happen.
For each round, give claims about its format and content.
{CITE_RULES}"""
        rounds = generate(prompt, Rounds, passages_text(passages["interviews"])).rounds

    # Check every claim from 1 and 2: resolve citations, then one support call
    claims = snapshot.claims + [c for r in rounds for c in r.details]
    for c in claims:
        c.citations = [i for i in c.citations if i in evidence]
        c.support = (
            "unsupported" if c.kind == "observed" and not c.citations else "unchecked"
        )
        if c.citations and all(evidence[i].peer for i in c.citations):
            c.kind = "inferred"  # peer evidence cannot support an observed claim about the target
    if cited := [c for c in claims if c.citations]:
        listing = "\n\n".join(
            f"Claim {n}: {c.text}\nExcerpts:\n"
            + "\n".join(f"- {evidence[i].excerpt}" for i in c.citations)
            for n, c in enumerate(cited, 1)
        )
        for c, label in zip(cited, generate(SUPPORT_PROMPT, Support, listing).labels):
            c.support = label

    # 4 and 5. Resume deep-dive questions and JD gaps
    review = ResumeReview(questions=[], gaps=[])
    if request.resume_text:
        prompt = f"""Compare the resume with the job description.
1. Write up to 5 deep-dive questions an interviewer would ask about specific resume items.
2. List the main job requirements and whether the resume shows each one.
   Use "not_shown_in_resume" when the resume does not mention it; never claim the candidate lacks a skill.

Job description:
{request.jd_text}

Resume:
{request.resume_text}"""
        review = generate(prompt, ResumeReview)

    # 6. Study schedule: one topic per day in rank order, cycling when days outnumber topics
    ranked = [t.topic_id for t in sorted(topics, key=lambda t: t.rank)]
    days = (
        (request.interview_date - request.as_of).days
        if request.interview_date and ranked
        else 0
    )
    schedule = [
        StudyDay(
            day=request.as_of + timedelta(days=i),
            minutes=request.daily_study_minutes,
            topic_ids=[ranked[i % len(ranked)]],
        )
        for i in range(days)
    ]

    return Report(
        company=request.company,
        role=request.role,
        as_of=request.as_of,
        status=status,
        company_snapshot=[c for c in snapshot.claims if c.support != "unsupported"],
        interview_rounds=[
            Round(name=r.name, details=kept)
            for r in rounds
            if (kept := [c for c in r.details if c.support != "unsupported"])
        ],
        ranked_topics=topics,  # 3
        resume_questions=review.questions,
        gaps=review.gaps,
        schedule=schedule,
        questions_to_ask=snapshot.questions_to_ask,
        missing_evidence=missing,
        evidence=evidence,
        run=run,
    )
