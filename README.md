# Interview Prep Researcher

An agentic research pipeline that builds a cited interview briefing for **new-grad and intern software roles** at well-known companies. It takes a company, a role, the job description and your resume. It returns six sections: company snapshot, likely interview rounds, ranked prep topics, resume deep-dive questions, JD-vs-resume gaps and a study schedule.

Status: **scaffold only.** Modules hold a one-line description of what they will do; nothing runs yet.

## Setup (Windows, local only)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Install [Ollama](https://ollama.com), set `OLLAMA_CONTEXT_LENGTH=8192` before starting it, then:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Python 3.11 is used because it is already installed; the plan's 3.12 target works the same.

## File map

| Path | Owns |
|------|------|
| `backend/config.py` | Models, limits, source policy, target companies and roles |
| `backend/models.py` | Chat model client and structured-output helper |
| `backend/workflow.py` | Graph state, LangGraph loop, budgets, terminal entry point |
| `backend/agents/planner.py` | Queries and coverage check |
| `backend/agents/researcher.py` | Company, tech and interview-experience research |
| `backend/agents/synthesizer.py` | Six sections, citations, support checks, schedule |
| `backend/rag/ingest.py` | Normalize, deduplicate, chunk |
| `backend/rag/embeddings.py` | Local embeddings |
| `backend/rag/retrieval.py` | Hybrid search, ranking, support counts |
| `backend/tools/search.py` | Web discovery (ddgs) and page fetching |
| `backend/tools/resume.py` | Resume parsing |
| `backend/main.py` | FastAPI routes |
| `evals/` | Frozen evaluation runner and dataset |
| `web/` | Next.js frontend (built last) |

## Sources

- **Official** (highest weight): each company's own careers, students and engineering pages, listed in `COMPANIES` in `config.py`.
- **Community** (`COMMUNITY_DOMAINS`): GeeksforGeeks interview experiences, Medium, dev.to, Hashnode, GitHub, Tech Interview Handbook, interviewing.io.
- **Other:** any other domain, fetched only when its robots.txt allows it, at the lowest weight.
- **Blocked** (`BLOCKED_DOMAINS`): Glassdoor, Blind, LinkedIn, Reddit, LeetCode and Indeed. These have login walls, JavaScript-only pages, or terms that forbid automated collection. You can still paste text you read there as supplied evidence.

Search results are only leads. Claims are made only from fetched or pasted text.

## Companies and roles

Pilot (first evaluation set): Google, Amazon, Microsoft, Meta, Goldman Sachs.
Expansion: Apple, NVIDIA, Qualcomm, Adobe, Salesforce, Oracle, Atlassian, Uber, Bloomberg, JPMorgan Chase.
Role families: new grad (SWE New Grad, SDE I, ...) and intern (SWE Intern, SDE Intern, Summer Analyst).

## Commands (planned, not yet implemented)

```bash
python -m backend.workflow        # terminal run
python -m evals.evaluate          # frozen evaluation
uvicorn backend.main:app --reload # local API
```

## Results and limitations

None measured yet. Only measured numbers go here.
