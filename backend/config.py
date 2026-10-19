"""Models, limits, source policy and target companies. Every tunable lives here."""

import os

# "live" searches and fetches; "frozen" uses only the supplied corpus (evaluation).
MODE = os.getenv("MODE", "live")

# --- Models (Ollama's OpenAI-compatible endpoint; context size is set on the server, see .env.example)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b")  # confirm in the Day 0 speed test
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
MAX_OUTPUT_TOKENS = 2048
LLM_CONCURRENCY = 1  # 4 GB VRAM: one model call at a time
STRUCTURED_OUTPUT_RETRIES = 1

# --- Research loop budgets
MAX_PASSES = 3  # target role -> related roles at the company -> peer companies
QUERIES_PER_PASS = 4
RESULTS_PER_QUERY = 5
MAX_FETCHED_PAGES = 20
FETCH_TIMEOUT_S = 15
FETCH_RETRIES = 2
RUN_BUDGET_S = 300
SEARCH_REGION = os.getenv("SEARCH_REGION", "us-en")  # ddgs region, e.g. in-en
USER_AGENT = "InterviewPrepResearcher/0.1 (personal research tool)"

# --- Retrieval
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 80
CANDIDATES_PER_PATH = 20
TOP_PASSAGES = 8
RRF_K = 60
NEWS_WINDOW_DAYS = 183

# --- Source policy
# Tiers feed the credibility weight in rag/retrieval.py. A company's own domains are "official".
# Any other domain is fetched only if robots.txt allows it, at the "other" tier.
COMMUNITY_DOMAINS = {
    "geeksforgeeks.org",  # largest public archive of dated new-grad and intern interview experiences
    "medium.com",
    "dev.to",
    "hashnode.dev",
    "github.com",
    "techinterviewhandbook.org",
    "interviewing.io",
}
# Never fetched automatically: login walls, JS-only pages, or terms that forbid automated collection.
# Text the user copies from these sites can still be pasted in as supplied evidence.
BLOCKED_DOMAINS = {
    "glassdoor.com",
    "teamblind.com",
    "linkedin.com",
    "reddit.com",
    "leetcode.com",
    "indeed.com",
}

# --- Target roles (new grads and interns). The second family is the "related roles" widening step.
ROLE_FAMILIES = {
    "new_grad": [
        "Software Engineer, New Grad",
        "University Graduate Software Engineer",
        "Software Development Engineer I",
        "Software Engineer I",
        "Associate Software Engineer",
    ],
    "intern": [
        "Software Engineering Intern",
        "Software Development Engineer Intern",
        "Engineering Summer Analyst",
    ],
}

# --- Companies. "group" defines peers for the last widening pass; "pilot" marks the first eval set.
COMPANIES = {
    "Google": {"domains": ["google.com", "withgoogle.com"], "group": "big_tech", "pilot": True},
    "Amazon": {"domains": ["amazon.jobs", "aboutamazon.com", "amazon.science"], "group": "big_tech", "pilot": True},
    "Microsoft": {"domains": ["microsoft.com"], "group": "big_tech", "pilot": True},
    "Meta": {"domains": ["metacareers.com", "engineering.fb.com"], "group": "big_tech", "pilot": True},
    "Goldman Sachs": {"domains": ["goldmansachs.com"], "group": "finance", "pilot": True},
    "Apple": {"domains": ["apple.com"], "group": "big_tech", "pilot": False},
    "NVIDIA": {"domains": ["nvidia.com"], "group": "hardware", "pilot": False},
    "Qualcomm": {"domains": ["qualcomm.com"], "group": "hardware", "pilot": False},
    "Adobe": {"domains": ["adobe.com"], "group": "enterprise", "pilot": False},
    "Salesforce": {"domains": ["salesforce.com"], "group": "enterprise", "pilot": False},
    "Oracle": {"domains": ["oracle.com"], "group": "enterprise", "pilot": False},
    "Atlassian": {"domains": ["atlassian.com"], "group": "enterprise", "pilot": False},
    "Uber": {"domains": ["uber.com"], "group": "enterprise", "pilot": False},
    "Bloomberg": {"domains": ["bloomberg.com", "techatbloomberg.com"], "group": "finance", "pilot": False},
    "JPMorgan Chase": {"domains": ["jpmorganchase.com", "jpmorgan.com"], "group": "finance", "pilot": False},
}
