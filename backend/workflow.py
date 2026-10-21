"""Graph state, LangGraph wiring, budgets and the common run function. Terminal entry point: python -m backend.workflow"""

import argparse
import operator
import time
from datetime import date
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend import config
from backend.agents import planner, researcher, synthesizer
from backend.models import usage
from backend.rag import ingest, retrieval
from backend.schemas import (
    Branch,
    Evidence,
    Experience,
    Query,
    Report,
    Request,
    ResearchResult,
    RunInfo,
)
from backend.tools import resume


class State(TypedDict):
    request: Request
    started: float  # time.monotonic() at the start of the run
    pass_no: int
    queries: list[Query]
    results: Annotated[
        list[ResearchResult], operator.add
    ]  # parallel branches append here
    supplied_experiences: list[Experience]
    experiences: list[Experience]
    passages: dict[Branch, list[Evidence]]
    corpus: dict[str, Evidence]
    gaps: dict[Branch, str]
    report: Report


def run(request: Request) -> Report:
    """The one entry point shared by the terminal, API and evals."""
    usage.clear()
    started = time.monotonic()
    state = GRAPH.invoke(
        {
            "request": request,
            "started": started,
            "pass_no": 1,
            "results": [],
            "supplied_experiences": [
                e for s in request.supplied if (e := researcher.read_account(s))
            ],
            "experiences": [],
            "passages": {"company": [], "technology": [], "interviews": []},
            "corpus": {},
            "gaps": {},
        }
    )
    report = state["report"]
    report.run.duration_s = round(time.monotonic() - started, 1)
    report.run.model_calls = usage["calls"]
    return report


def _plan(state: State) -> dict:
    return {
        "queries": planner.search_queries(
            state["request"], state["pass_no"], state["gaps"]
        )
    }


def _dispatch(state: State) -> list[Send] | str:
    """One research branch per query branch, sharing what is left of the page budget."""
    fetched = _fetched_urls(state)
    branches = sorted({q.branch for q in state["queries"]})
    remaining = config.MAX_FETCHED_PAGES - len(fetched)
    if not branches or remaining <= 0:
        return (
            "collect" if state["pass_no"] == 1 else "synthesize"
        )  # pass 1 still indexes supplied evidence
    return [
        Send(
            "research",
            {
                "branch": branch,
                "queries": [q for q in state["queries"] if q.branch == branch],
                "request": state["request"],
                "exclude": fetched,
                "max_pages": remaining // len(branches),
            },
        )
        for branch in branches
    ]


def _research(payload: dict) -> dict:
    return {"results": [researcher.research(**payload)]}


def _collect(state: State) -> dict:
    """Merge every pass so far, index it, retrieve passages per question and check coverage."""
    request = state["request"]
    sources = ingest.dedupe(
        request.supplied + [s for r in state["results"] for s in r.sources]
    )
    kept = {s.source_id for s in sources}
    found = state["supplied_experiences"] + [
        e for r in state["results"] for e in r.experiences
    ]
    accounts = {e.source_id: e for e in found if e.source_id in kept}

    chunks = [
        c
        for s in sources
        for c in ingest.chunk(s, getattr(accounts.get(s.source_id), "event_date", None))
    ]
    index = retrieval.build(
        chunks
    )  # shortcut: re-embeds every chunk each pass; cache vectors by chunk_id if slow

    questions = planner.research_questions(request)
    interview_companies = {request.company, *planner.scope(request, 3)[0]}
    passages = {
        branch: retrieval.search(
            index,
            question,
            interview_companies if branch == "interviews" else {request.company},
            request.as_of,
        )
        for branch, question in questions.items()
    }

    checked = (
        questions if state["pass_no"] == 1 else {"interviews": questions["interviews"]}
    )
    gaps = {
        branch: gap for branch, gap in state["gaps"].items() if branch not in checked
    }
    for branch, question in checked.items():
        coverage = planner.check_coverage(question, passages[branch])
        if not coverage.answered:
            gaps[branch] = coverage.missing

    return {
        "passages": passages,
        "corpus": {c.chunk_id: c for c in chunks},
        "experiences": list(accounts.values()),
        "gaps": gaps,
        "pass_no": state["pass_no"] + 1,
    }


def _next(state: State) -> str:
    """Widen again only for an interview gap, in live mode, within the pass and time budgets."""
    within_time = time.monotonic() - state["started"] < config.RUN_BUDGET_S
    widen = state["request"].mode == "live" and "interviews" in state["gaps"]
    return (
        "plan"
        if widen and state["pass_no"] <= config.MAX_PASSES and within_time
        else "synthesize"
    )


def _synthesize(state: State) -> dict:
    request = state["request"]
    topics = retrieval.rank_topics(state["experiences"], state["corpus"])
    evidence = {p.chunk_id: p for group in state["passages"].values() for p in group}
    evidence |= {i: state["corpus"][i] for t in topics for i in t.citations}
    out_of_time = time.monotonic() - state["started"] >= config.RUN_BUDGET_S
    run_info = (
        RunInfo(  # duration and model calls are filled in by run() after synthesis
            chat_model=config.CHAT_MODEL,
            embed_model=config.EMBED_MODEL,
            passes=state["pass_no"] - 1,
            duration_s=0,
            pages_fetched=len(_fetched_urls(state)),
            errors=[e for r in state["results"] for e in r.errors],
        )
    )
    report = synthesizer.write_report(
        request,
        state["passages"],
        evidence,
        topics,
        list(state["gaps"].values()),
        run_info,
        "partial" if out_of_time else "complete",
    )
    return {"report": report}


def _fetched_urls(state: State) -> set[str]:
    return {s.url for r in state["results"] for s in r.sources}


def _graph():
    graph = StateGraph(State)
    graph.add_node("plan", _plan)
    graph.add_node("research", _research)
    graph.add_node("collect", _collect)
    graph.add_node("synthesize", _synthesize)
    graph.add_conditional_edges(
        START, lambda s: "plan" if s["request"].mode == "live" else "collect"
    )
    graph.add_conditional_edges("plan", _dispatch)
    graph.add_edge("research", "collect")
    graph.add_conditional_edges("collect", _next)
    graph.add_edge("synthesize", END)
    return graph.compile()


GRAPH = _graph()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research a company and role; print the briefing as JSON."
    )
    parser.add_argument("--company", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--role-family", choices=["new_grad", "intern"], default="new_grad"
    )
    parser.add_argument(
        "--jd", type=Path, required=True, help="job description text file"
    )
    parser.add_argument("--resume", type=Path, help="resume PDF or text file")
    parser.add_argument("--interview-date", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--minutes", type=int, default=90, help="daily study minutes")
    parser.add_argument(
        "--evidence",
        type=Path,
        nargs="*",
        default=[],
        help="text files used as supplied evidence",
    )
    parser.add_argument(
        "--frozen", action="store_true", help="use only --evidence; no web search"
    )
    args = parser.parse_args()

    resume_text = ""
    if args.resume:
        is_pdf = args.resume.suffix.lower() == ".pdf"
        resume_text = (
            resume.read_pdf(args.resume.read_bytes())
            if is_pdf
            else args.resume.read_text("utf-8")
        )

    request = Request(
        company=args.company,
        role=args.role,
        role_family=args.role_family,
        jd_text=args.jd.read_text("utf-8"),
        resume_text=resume_text,
        interview_date=args.interview_date,
        daily_study_minutes=args.minutes,
        mode="frozen" if args.frozen else config.MODE,
        supplied=[
            ingest.from_text(f.read_text("utf-8"), f.name, args.company)
            for f in args.evidence
        ],
    )
    print(run(request).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
