"""Coverage agent: judges whether retrieved passages answer a research question."""

from pydantic import BaseModel

from backend.models import generate, passages_text
from backend.schemas import Evidence


class Coverage(BaseModel):
    answered: bool
    missing: str  # the evidence still needed; empty when answered


def check(question: str, passages: list[Evidence]) -> Coverage:
    """Whether the passages answer the question, and what is missing if not."""
    if not passages:
        return Coverage(answered=False, missing=question)
    prompt = f"""Do the passages answer this question with specific facts?
Question: {question}
If they do not, state in one sentence what evidence is still missing; otherwise leave missing empty."""
    return generate(prompt, Coverage, passages_text(passages))
