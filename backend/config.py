"""Models, limits, source policy and target companies. Every tunable lives here."""

import os

# "live" searches and fetches; "frozen" uses only the supplied corpus (evaluation).
MODE = os.getenv("MODE", "live")

# --- Models (Ollama native API: context size and thinking are set per request)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
NUM_CTX = 8192
MAX_OUTPUT_TOKENS = 2048
LLM_CONCURRENCY = 1  # 4 GB VRAM: one model call at a time
EXTRACT_CHARS = 12_000  # page text sent when reading an interview account (~3k tokens)

# --- Research loop budgets
MAX_PASSES = 3  # target role -> related roles at the company -> peer companies
QUERIES_PER_PASS = 4
RESULTS_PER_QUERY = 5
MAX_FETCHED_PAGES = 20
FETCH_TIMEOUT_S = 15
RUN_BUDGET_S = 300
SEARCH_REGION = os.getenv("SEARCH_REGION", "us-en")  # ddgs region, e.g. in-en
USER_AGENT = "InterviewPrepResearcher/0.1 (personal research tool)"
MAX_UPLOAD_BYTES = 5_000_000  # resume PDF limit for the API

# --- Retrieval
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 80
CANDIDATES_PER_PATH = 20
TOP_PASSAGES = 8
RRF_K = 60
TOP_TOPICS = 10  # scored as topic recall/precision at 10
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

# --- Topic list. The synthesizer ranks only these IDs and eval labels use the same IDs,
# so topic recall/precision is an exact match. Add an ID rather than renaming one.
TOPICS = {
    # Data structures and algorithms
    "dsa.arrays_hashing": "Arrays, strings and hashing",
    "dsa.two_pointers_sliding_window": "Two pointers and sliding window",
    "dsa.binary_search": "Binary search",
    "dsa.sorting": "Sorting and custom comparators",
    "dsa.stacks_queues": "Stacks, queues and monotonic stacks",
    "dsa.linked_lists": "Linked lists",
    "dsa.trees": "Binary trees, BSTs and traversals",
    "dsa.heaps": "Heaps, priority queues and top-k",
    "dsa.graphs_traversal": "Graph and grid traversal (BFS, DFS)",
    "dsa.graphs_advanced": "Shortest paths, topological sort and union-find",
    "dsa.dynamic_programming": "Dynamic programming",
    "dsa.greedy_intervals": "Greedy algorithms and intervals",
    "dsa.backtracking": "Recursion and backtracking",
    "dsa.tries": "Tries and string search",
    "dsa.bit_math": "Bit manipulation and math",
    "dsa.complexity": "Time and space complexity analysis",
    "dsa.design_data_structure": "Design a data structure (LRU cache, iterators)",
    # Coding practice
    "coding.online_assessment": "Timed online assessment (HackerRank, Codility)",
    "coding.edge_cases_testing": "Edge cases, clean code and testing",
    "coding.debugging": "Debugging and code reading",
    "coding.language": "Language fundamentals (Java, C++, Python)",
    "coding.object_oriented_design": "Object-oriented and low-level design",
    # CS fundamentals
    "cs.operating_systems": "Operating systems (processes, memory, scheduling)",
    "cs.concurrency": "Concurrency (threads, locks, deadlocks)",
    "cs.networking": "Networking (HTTP, TCP/IP, DNS)",
    "cs.databases_sql": "Databases and SQL (joins, indexes, transactions)",
    "cs.computer_architecture": "Computer architecture (caches, memory hierarchy)",
    "cs.c_low_level": "C/C++ low-level programming (pointers, memory)",
    # System design (new-grad depth)
    "design.scalability_basics": "Scalability basics (caching, load balancing, sharding)",
    "design.api_design": "API design (REST, data models)",
    "design.small_system": "Design a small system (URL shortener, rate limiter)",
    # Behavioral
    "behavioral.star_stories": "Behavioral stories in STAR format",
    "behavioral.amazon_leadership_principles": "Amazon Leadership Principles",
    "behavioral.googleyness": "Googleyness and leadership",
    "behavioral.motivation": "Why this company and role",
    "behavioral.project_deep_dive": "Resume project and internship deep dive",
    # Aptitude and domain
    "aptitude.quant_reasoning": "Quantitative aptitude and logical reasoning",
    "aptitude.probability_puzzles": "Probability, puzzles and brain teasers",
    "domain.finance_basics": "Markets and finance basics",
    "domain.ml_basics": "Machine learning fundamentals",
    "domain.web_frontend": "Web and frontend fundamentals (JavaScript, React)",
    "domain.cloud_linux_git": "Cloud, Linux and Git basics",
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
