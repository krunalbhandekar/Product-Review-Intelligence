"""Weekly pulse generation from aggregated review summaries.

This package turns one or more :class:`SummaryReport` objects (the
"aggregated summary" output of the chunk-summarisation pipeline) into a
concise, leadership-friendly weekly pulse:

* under ~250 words of executive narrative,
* the top 3 themes (ranked by prevalence and evidence),
* 3 representative user quotes,
* 3 action ideas.

Crucially, this module never touches raw reviews. It consumes only the
already-aggregated :class:`SummaryReport` contract, which keeps it cheap,
deterministic, and free of LLM dependencies.
"""

from app.services.pulse.aggregation import PulseAggregator, ThemeBucket
from app.services.pulse.formatting import PulseFormatter
from app.services.pulse.ranking import PulseRanker
from app.services.pulse.schemas import (
    PulseAction,
    PulseQuote,
    PulseTheme,
    WeeklyPulse,
    WeeklyPulseMeta,
)
from app.services.pulse.service import WeeklyPulseGenerator

__all__ = [
    "PulseAction",
    "PulseAggregator",
    "PulseFormatter",
    "PulseQuote",
    "PulseRanker",
    "PulseTheme",
    "ThemeBucket",
    "WeeklyPulse",
    "WeeklyPulseGenerator",
    "WeeklyPulseMeta",
]
