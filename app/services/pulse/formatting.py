"""Render a :class:`WeeklyPulse` into leadership-friendly markdown.

The markdown view is intentionally minimal: a short title, one
paragraph of executive context, three numbered themes, a quote block,
and a bulleted action list. It is designed to read well in:

* email clients (no nested lists or wide tables),
* Slack (markdown-ish rendering keeps the structure visible),
* and rendered docs (Notion, Google Docs paste).
"""

from __future__ import annotations

import re

from app.services.pulse.schemas import WeeklyPulse

_SENTIMENT_GLYPH: dict[str, str] = {
    "positive": "+",
    "negative": "-",
    "mixed": "~",
    "neutral": "·",
}

_WORD_RE = re.compile(r"\b\w[\w'-]*\b")


def count_words(text: str) -> int:
    """Stable word counter used for the 250-word budget check."""
    return len(_WORD_RE.findall(text))


def clip_to_word_budget(text: str, *, budget: int) -> str:
    """Trim ``text`` to at most ``budget`` words on a word boundary.

    Adds an ellipsis when content is dropped so readers know the
    paragraph was clipped rather than written that way. The 250-word
    target in the requirements is enforced against the whole pulse
    body — this helper is the primary lever for that.
    """
    if budget <= 0:
        return ""
    words = _WORD_RE.findall(text)
    if len(words) <= budget:
        return text.strip()
    # Find char offset of (budget+1)th word and cut before it.
    matches = list(_WORD_RE.finditer(text))
    cutoff = matches[budget].start()
    trimmed = text[:cutoff].rstrip().rstrip(",;:.—-")
    return f"{trimmed}…"


class PulseFormatter:
    """Render :class:`WeeklyPulse` instances into markdown.

    Pure function over the structured pulse — never reaches back into
    the aggregator or ranker. This separation keeps the JSON contract
    authoritative: anything the formatter shows must be in the JSON.
    """

    def render_markdown(self, pulse: WeeklyPulse) -> str:
        lines: list[str] = []
        lines.append(f"# {pulse.headline}")
        lines.append("")
        lines.extend(self._render_meta_line(pulse))
        lines.append("")

        if pulse.executive_summary:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(pulse.executive_summary)
            lines.append("")

        if pulse.themes:
            lines.append("## Top Themes")
            lines.append("")
            for theme in pulse.themes:
                glyph = _SENTIMENT_GLYPH.get(theme.sentiment, "·")
                lines.append(
                    f"{theme.rank}. **{theme.label}** "
                    f"_({glyph} {theme.sentiment} · {theme.prevalence} prevalence)_"
                )
                lines.append(f"   {theme.headline}")
            lines.append("")

        if pulse.quotes:
            lines.append("## What Users Are Saying")
            lines.append("")
            for quote in pulse.quotes:
                attribution = (
                    f" — _on {quote.theme_label}_" if quote.theme_label else ""
                )
                lines.append(f"> {quote.text}{attribution}")
                lines.append(">")
            # Drop the trailing empty blockquote marker.
            if lines[-1] == ">":
                lines.pop()
            lines.append("")

        if pulse.actions:
            lines.append("## Action Ideas")
            lines.append("")
            for action in pulse.actions:
                tag = f" _(re: {action.theme_label})_" if action.theme_label else ""
                lines.append(f"- {action.text}{tag}")
            lines.append("")

        # Trim trailing blank lines for a clean tail.
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"

    def _render_meta_line(self, pulse: WeeklyPulse) -> list[str]:
        meta = pulse.meta
        bits: list[str] = []
        if meta.window_start and meta.window_end:
            bits.append(
                f"{meta.window_start.date().isoformat()} → "
                f"{meta.window_end.date().isoformat()}"
            )
        bits.append(f"{meta.reports_count} report{'s' if meta.reports_count != 1 else ''}")
        bits.append(f"{meta.total_reviews:,} reviews")
        bits.append(f"generated {meta.generated_at.date().isoformat()}")
        return [f"_{' · '.join(bits)}_"]


__all__ = ["PulseFormatter", "clip_to_word_budget", "count_words"]
