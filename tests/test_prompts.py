"""Offline tests for agent system prompts and the prompt loader.

No live API calls — pure filesystem reads and string assertions.
"""
import pytest

from app.llm.prompt_loader import load_prompt

# All 9 agent prompt names
ALL_PROMPT_NAMES = [
    "opportunity_scout",
    "client_insights",
    "planner",
    "intel_gathering",
    "macro",
    "portfolio",
    "news",
    "synthesizer",
    "chat",
]

# Each prompt must mention its output schema by name
SCHEMA_GUARDS: dict[str, str] = {
    "opportunity_scout": "OpportunitySignals",
    "client_insights": "ClientSnapshot",
    "planner": "Plan",
    "intel_gathering": "IntelFindings",
    "macro": "MacroFindings",
    "portfolio": "PortfolioFinding",
    "news": "NewsFindings",
    "synthesizer": "BriefSchema",
    "chat": "ChatResponse",
}


def test_all_nine_prompts_load():
    """Every prompt file loads, strips cleanly, and is substantive (>100 chars)."""
    for name in ALL_PROMPT_NAMES:
        text = load_prompt(name)
        assert isinstance(text, str), f"load_prompt('{name}') did not return a str"
        assert len(text) > 100, (
            f"Prompt '{name}' is suspiciously short ({len(text)} chars). "
            "Expected a real system instruction."
        )


def test_unknown_prompt_raises():
    """Requesting a non-existent prompt raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_prompts_name_their_schema():
    """Each prompt text must mention its corresponding Pydantic output schema.

    This is a cheap contract guard: if a prompt is accidentally pointed at the
    wrong schema (or the schema is renamed), this test will catch it.
    """
    for name, schema_name in SCHEMA_GUARDS.items():
        text = load_prompt(name)
        assert schema_name in text, (
            f"Prompt '{name}' does not mention its output schema '{schema_name}'. "
            "The prompt must instruct the model to produce that schema."
        )
