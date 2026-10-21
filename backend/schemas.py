"""Records shared by the agents, workflow, API and evals."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.config import MODE, TOPICS

Branch = Literal["company", "technology", "interviews"]
Tier = Literal["official", "community", "other", "supplied"]
TopicId = Literal[tuple(TOPICS)]  # constrains model output to known topic IDs


class Query(BaseModel):
    branch: Branch
    company: str
    text: str


class Source(BaseModel):
    """A fetched page or supplied document, before chunking."""

    source_id: str
    url: str  # canonical URL, or the supplied file ID
    title: str
    text: str
    published: date | None = None
    fetched: date
    tier: Tier
    company: str
    peer: bool = False


class Request(BaseModel):
    company: str
    role: str
    role_family: Literal["new_grad", "intern"] = "new_grad"
    jd_text: str
    resume_text: str = ""
    interview_date: date | None = None
    daily_study_minutes: int = Field(90, ge=15, le=600)
    mode: Literal["live", "frozen"] = MODE
    supplied: list[
        Source
    ] = []  # pasted or uploaded evidence; the only corpus in frozen mode
    as_of: date = Field(default_factory=date.today)


class Experience(BaseModel):
    """One first-hand interview account read from a source."""

    source_id: str
    company: str
    peer: bool
    role: str
    event_date: date | None  # when the interview took place, not when it was posted
    rounds: list[str]
    topic_ids: list[TopicId]


class ResearchResult(BaseModel):
    branch: Branch
    sources: list[Source] = []
    experiences: list[Experience] = []
    errors: list[str] = []


class Evidence(BaseModel):
    """One citable excerpt; claims and topics cite it by chunk_id."""

    chunk_id: str
    source_id: str
    url: str
    title: str
    excerpt: str
    published: date | None = None  # None means unknown, never treated as recent
    event_date: date | None = None
    tier: Tier
    company: str
    peer: bool = False  # about a peer company; the report must label it


class Claim(BaseModel):
    text: str
    kind: Literal["observed", "inferred", "recommendation"]
    citations: list[str]
    support: Literal["supported", "partial", "unsupported", "unchecked"] = "unchecked"


class Round(BaseModel):
    name: str  # e.g. "Online assessment", "Technical phone screen"
    details: list[Claim]


class RankedTopic(BaseModel):
    topic_id: TopicId
    rank: int
    accounts_with_topic: int  # independent experience groups that mention it
    accounts_total: int  # independent experience groups in the eligible corpus
    confidence_reason: str
    citations: list[str] = []


class ResumeQuestion(BaseModel):
    question: str
    resume_item: str  # the resume line it probes
    why: str  # the JD requirement or topic it connects to


class Gap(BaseModel):
    requirement: str  # taken from the JD
    status: Literal["shown", "partial", "not_shown_in_resume"]
    note: str = ""


class StudyDay(BaseModel):
    day: date
    minutes: int
    topic_ids: list[TopicId]


class RunInfo(BaseModel):
    chat_model: str
    embed_model: str
    passes: int
    duration_s: float
    model_calls: int = 0
    pages_fetched: int = 0
    errors: list[str] = []


class Report(BaseModel):
    """The six-section briefing shared by the terminal, API, evals and web page."""

    company: str
    role: str
    as_of: date
    status: Literal["complete", "partial"]
    company_snapshot: list[Claim]  # 1
    interview_rounds: list[Round]  # 2
    ranked_topics: list[RankedTopic]  # 3
    resume_questions: list[ResumeQuestion]  # 4
    gaps: list[Gap]  # 5
    schedule: list[StudyDay]  # 6, with questions_to_ask
    questions_to_ask: list[str]
    missing_evidence: list[str]  # what was searched for and not found
    evidence: dict[str, Evidence]  # chunk_id -> excerpt
    run: RunInfo

    @model_validator(mode="after")
    def _citations_resolve(self):
        claims = [
            *self.company_snapshot,
            *(c for r in self.interview_rounds for c in r.details),
        ]
        cited = {i for c in claims for i in c.citations} | {
            i for t in self.ranked_topics for i in t.citations
        }
        if dangling := cited - self.evidence.keys():
            raise ValueError(f"dangling citation IDs: {sorted(dangling)}")
        return self
