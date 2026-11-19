"""Human-readable metadata for every model input.

The frontend builds its form from ``GET /schema``, which merges this file with
the value ranges in ``src.data.schema``. Keeping labels and help text here
(and not in the UI) means the form can never drift from the API contract.
"""

from __future__ import annotations

FIELD_META: dict[str, dict[str, str]] = {
    "state_code": {
        "label": "State",
        "group": "Basics",
        "help": "US state where the startup is based.",
    },
    "category_code": {
        "label": "Industry",
        "group": "Basics",
        "help": "Main industry category.",
    },
    "age_first_funding_year": {
        "label": "Years from founding to first funding",
        "group": "Funding",
        "help": "Negative means funding arrived before the official founding date.",
    },
    "age_last_funding_year": {
        "label": "Years from founding to latest funding",
        "group": "Funding",
        "help": "Must be at least the value above.",
    },
    "funding_rounds": {
        "label": "Number of funding rounds",
        "group": "Funding",
        "help": "How many rounds the startup has raised so far.",
    },
    "funding_total_usd": {
        "label": "Total funding raised (USD)",
        "group": "Funding",
        "help": "Sum of all rounds, in US dollars.",
    },
    "avg_participants": {
        "label": "Average investors per round",
        "group": "Funding",
        "help": "Average number of participants across funding rounds.",
    },
    "has_VC": {
        "label": "Venture capital backing",
        "group": "Investors",
        "help": "Has the startup raised from a venture capital firm?",
    },
    "has_angel": {
        "label": "Angel investor backing",
        "group": "Investors",
        "help": "Has the startup raised from angel investors?",
    },
    "has_roundA": {"label": "Raised a Series A", "group": "Investors", "help": "Round A completed."},
    "has_roundB": {"label": "Raised a Series B", "group": "Investors", "help": "Round B completed."},
    "has_roundC": {"label": "Raised a Series C", "group": "Investors", "help": "Round C completed."},
    "has_roundD": {"label": "Raised a Series D", "group": "Investors", "help": "Round D completed."},
    "is_top500": {
        "label": "Among the top-500 startups",
        "group": "Investors",
        "help": "Listed in a top-500 ranking of startups.",
    },
    "relationships": {
        "label": "Key business relationships",
        "group": "Traction",
        "help": "Number of recorded people/organisation relationships.",
    },
    "milestones": {
        "label": "Milestones achieved",
        "group": "Traction",
        "help": "Number of recorded company milestones (0 if none).",
    },
    "age_first_milestone_year": {
        "label": "Years from founding to first milestone",
        "group": "Traction",
        "help": "Leave empty if there are no milestones.",
    },
    "age_last_milestone_year": {
        "label": "Years from founding to latest milestone",
        "group": "Traction",
        "help": "Leave empty if there are no milestones.",
    },
}

BINARY_FIELDS: frozenset[str] = frozenset(
    {"has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD", "is_top500"}
)
