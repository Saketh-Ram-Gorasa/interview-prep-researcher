"""Frozen evaluation: run dataset cases through baseline variants and the workflow, then score them.

Run: python -m evals.evaluate --split dev
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from backend import config
from backend.agents.researcher import TOPIC_LINES
from backend.models import generate, usage
from backend.rag import ingest
from backend.schemas import Request, TopicId
from backend.workflow import run

DATASET = Path(__file__).with_name("dataset.json")
RESULTS = Path("runs")
VARIANTS = ["generic", "no_retrieval", "rag"]
NOTES = [
    "rag is the frozen workflow: it runs one pass, so single-pass RAG and the full loop are the same here",
    "fallback lift is not measured: frozen mode does not widen to related roles or peer companies",
    "citation completeness is not measured: the synthesizer drops observed claims without valid citations",
]


class CaseSource(BaseModel):
    title: str
    text: str
    company: str  # a company other than the case's is peer evidence
    published: date | None = None


class Case(BaseModel):
    id: str
    split: Literal["dev", "test"]
    company: str
    role: str
    role_family: Literal["new_grad", "intern"] = "new_grad"
    as_of: date  # cutoff: inputs existed by this date; later accounts supply the labels
    jd_text: str
    resume_text: str = ""
    sources: list[CaseSource]
    expected_topics: list[TopicId]  # hidden labels: never passed to the workflow


class Dataset(BaseModel):
    cases: list[Case]


class TopicGuess(BaseModel):
    topic_ids: list[TopicId]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score dataset cases on topic recall/precision at 10."
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    cases = Dataset.model_validate_json(args.dataset.read_text("utf-8")).cases
    dev_counts = Counter(
        t for c in cases if c.split == "dev" for t in c.expected_topics
    )
    generic = [
        t for t, _ in dev_counts.most_common(config.TOP_TOPICS)
    ]  # same list for every case

    rows, claims = [], []
    for case in [c for c in cases if c.split == args.split]:
        for variant in args.variants:
            usage.clear()
            started, report, error = time.monotonic(), None, ""
            try:
                if variant == "generic":
                    predicted = generic
                elif variant == "no_retrieval":
                    prompt = f"""Rank the {config.TOP_TOPICS} topics most likely asked in {case.role} interviews at {case.company}, most likely first.
Use only these topic IDs:
{TOPIC_LINES}"""
                    predicted = generate(prompt, TopicGuess).topic_ids
                else:
                    report = run(_request(case))
                    predicted = [t.topic_id for t in report.ranked_topics]
            except Exception as e:  # noqa: BLE001 - a failed case is counted, not fatal
                predicted, error = [], f"{type(e).__name__}: {e}"

            top = list(dict.fromkeys(predicted))[: config.TOP_TOPICS]
            expected = set(case.expected_topics)
            hits = len(set(top) & expected)
            rows.append(
                {
                    "case": case.id,
                    "variant": variant,
                    "recall_at_10": hits / len(expected)
                    if expected
                    else None,  # None: empty-label case
                    "precision_at_10": hits / config.TOP_TOPICS,
                    "returned": len(top),
                    "status": "failed"
                    if error
                    else (report.status if report else "complete"),
                    "latency_s": round(time.monotonic() - started, 1),
                    "model_calls": usage["calls"],
                    "error": error,
                }
            )
            if report:
                for claim in report.company_snapshot + [
                    c for r in report.interview_rounds for c in r.details
                ]:
                    excerpts = [report.evidence[i].excerpt for i in claim.citations]
                    claims.append(
                        {
                            "case": case.id,
                            **claim.model_dump(),
                            "excerpts": excerpts,
                            "human_label": None,
                        }
                    )

    summary = {}
    for variant in args.variants:
        mine = [r for r in rows if r["variant"] == variant]
        scored = [r for r in mine if r["recall_at_10"] is not None]
        summary[variant] = {
            "cases": len(mine),
            "empty_label_cases": len(mine) - len(scored),
            "failed": sum(r["status"] == "failed" for r in mine),
            "partial": sum(r["status"] == "partial" for r in mine),
            "recall_at_10": _mean([r["recall_at_10"] for r in scored]),
            "precision_at_10": _mean([r["precision_at_10"] for r in mine]),
            "mean_latency_s": _mean([r["latency_s"] for r in mine]),
            "mean_model_calls": _mean([r["model_calls"] for r in mine]),
        }

    agent_files = sorted(Path("backend").glob("agents/*.py")) + [
        Path("backend/models.py")
    ]
    output = {
        "split": args.split,
        "generic_from": "dev labels"
        + (" (in-sample for dev)" if args.split == "dev" else ""),
        "notes": NOTES,
        "summary": summary,
        "rows": rows,
        "citation_audit": random.Random(0).sample(
            claims, min(100, len(claims))
        ),  # label human_label by hand
        "record": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "prompt_hash": hashlib.sha1(
                b"".join(p.read_bytes() for p in agent_files)
            ).hexdigest()[:12],
            "settings": {
                k: v
                for k, v in vars(config).items()
                if k.isupper() and isinstance(v, int | float | str)
            },
        },
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"eval-{args.split}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(output, indent=2, default=str), "utf-8")

    print(
        f"{'variant':<14}{'cases':>6}{'recall@10':>11}{'prec@10':>9}{'failed':>8}{'latency_s':>11}"
    )
    for variant, s in summary.items():
        print(
            f"{variant:<14}{s['cases']:>6}{_fmt(s['recall_at_10']):>11}{_fmt(s['precision_at_10']):>9}"
            f"{s['failed']:>8}{_fmt(s['mean_latency_s']):>11}"
        )
    print(f"saved {path}")


def _request(case: Case) -> Request:
    """The workflow input for a case: frozen mode, supplied sources only, labels left out."""
    return Request(
        company=case.company,
        role=case.role,
        role_family=case.role_family,
        jd_text=case.jd_text,
        resume_text=case.resume_text,
        as_of=case.as_of,
        mode="frozen",
        supplied=[
            ingest.from_text(
                s.text, s.title, s.company, s.published, peer=s.company != case.company
            )
            for s in case.sources
        ],
    )


def _mean(values: list) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
