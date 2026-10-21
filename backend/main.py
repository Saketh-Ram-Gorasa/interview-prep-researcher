"""FastAPI entry point: validate requests, start, poll and cancel runs, health. All research goes through workflow.py.

Run: uvicorn backend.main:app --reload   (binds to 127.0.0.1 only)
"""

import base64
import binascii
import logging
import threading
import uuid
from datetime import date
from typing import Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from pypdf.errors import PdfReadError

from backend import config
from backend.rag import ingest
from backend.schemas import Report, Request
from backend.tools import resume, search
from backend.workflow import run

log = logging.getLogger("uvicorn.error")
app = FastAPI(title="Interview Prep Researcher")


class RunInput(BaseModel):
    company: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)
    role_family: Literal["new_grad", "intern"] = "new_grad"
    jd_text: str = Field("", max_length=20_000)
    jd_url: str = Field("", max_length=2_000)
    resume_pdf_base64: str = Field("", max_length=config.MAX_UPLOAD_BYTES * 4 // 3 + 4)
    interview_date: date | None = None
    daily_study_minutes: int = Field(90, ge=15, le=600)

    @model_validator(mode="after")
    def _one_job_description(self):
        if bool(self.jd_text) == bool(self.jd_url):
            raise ValueError("give exactly one of jd_text or jd_url")
        return self


class RunStatus(BaseModel):
    id: str
    status: Literal["running", "done", "failed"]
    report: Report | None = None
    error: str = ""


RUNS: dict[str, RunStatus] = {}  # in memory: a restart loses every run
CANCELS: dict[str, threading.Event] = {}
LOCK = threading.Lock()


@app.post("/runs", status_code=202)
def start_run(body: RunInput, tasks: BackgroundTasks) -> RunStatus:
    """Validate the input and start one background run; only one run may be active."""
    request = Request(
        company=body.company,
        role=body.role,
        role_family=body.role_family,
        jd_text=body.jd_text or _read_job_page(body.jd_url, body.company),
        resume_text=_read_resume(body.resume_pdf_base64),
        interview_date=body.interview_date,
        daily_study_minutes=body.daily_study_minutes,
    )
    run_id, cancel = uuid.uuid4().hex, threading.Event()
    with LOCK:
        if any(r.status == "running" for r in RUNS.values()):
            raise HTTPException(409, "a run is already in progress")
        RUNS[run_id] = RunStatus(id=run_id, status="running")
        CANCELS[run_id] = cancel
    tasks.add_task(_execute, run_id, request, cancel)
    return RUNS[run_id]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> RunStatus:
    if run_id not in RUNS:
        raise HTTPException(404, "run not found")
    return RUNS[run_id]


@app.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    """Cancel the run if it is still going and remove all of its data."""
    with LOCK:
        if run_id not in RUNS:
            raise HTTPException(404, "run not found")
        del RUNS[run_id]
        if cancel := CANCELS.pop(run_id, None):
            cancel.set()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chat_model": config.CHAT_MODEL,
        "running": sum(r.status == "running" for r in RUNS.values()),
    }


def _execute(run_id: str, request: Request, cancel: threading.Event) -> None:
    log.info("run %s started", run_id)
    try:
        update = {"status": "done", "report": run(request, cancel)}
    except Exception as e:  # noqa: BLE001 - the failure is reported on the run instead of lost
        log.error(
            "run %s failed: %s", run_id, type(e).__name__
        )  # type only: messages may hold private text
        update = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    with LOCK:
        if run_id in RUNS:  # a deleted run must not be recreated by a late finish
            RUNS[run_id] = RUNS[run_id].model_copy(update=update)
            CANCELS.pop(run_id, None)
    log.info("run %s finished", run_id)


def _read_resume(encoded: str) -> str:
    """Text of the uploaded resume PDF; the raw bytes are not kept."""
    if not encoded:
        return ""
    try:
        data = base64.b64decode(encoded, validate=True)
        if not data.startswith(b"%PDF-"):
            raise HTTPException(422, "resume must be a PDF")
        return resume.read_pdf(data)
    except (binascii.Error, PdfReadError) as e:
        raise HTTPException(
            422, f"could not read the resume PDF: {type(e).__name__}"
        ) from e


def _read_job_page(url: str, company: str) -> str:
    """Main text of the job description page, fetched under the same source policy as research."""
    try:
        html = search.fetch(url)
    except (OSError, httpx.HTTPError, httpx.InvalidURL) as e:
        raise HTTPException(422, f"could not fetch jd_url: {e}") from e
    page = ingest.from_html(url, html, company, search.tier(url, company), False)
    if not page:
        raise HTTPException(422, "no readable text at jd_url")
    return page.text
