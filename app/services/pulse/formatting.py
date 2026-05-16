"""Render a :class:`WeeklyPulse` into a leadership-ready markdown report.

The output is shaped for the eventual Google Doc surface: a clean cover
block, an executive summary, a small key-signals table, theme "cards"
with insight + quote + suggested action, then a Voice of Customer
roll-up and a Recommended Product Actions list. The intent is something
a PM could drop in front of leadership and have it skim in under two
minutes — not a raw markdown dump of the underlying JSON.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.services.pulse.schemas import (
    PulseAction,
    PulseQuote,
    PulseTheme,
    WeeklyPulse,
)

# Coloured circles are deliberately the only visual cue we use on theme
# cards. They scan faster than text glyphs in Google Docs and survive
# copy/paste into Slack or email without breaking layout.
_SENTIMENT_ICON: dict[str, str] = {
    "positive": "🟢",
    "negative": "🔴",
    "mixed": "🟡",
    "neutral": "⚪",
}

_SENTIMENT_LABEL: dict[str, str] = {
    "positive": "Positive",
    "negative": "Negative",
    "mixed": "Mixed",
    "neutral": "Neutral",
}

_PREVALENCE_LABEL: dict[str, str] = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
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
    matches = list(_WORD_RE.finditer(text))
    cutoff = matches[budget].start()
    trimmed = text[:cutoff].rstrip().rstrip(",;:.—-")
    return f"{trimmed}…"


def _fmt_date(dt: datetime) -> str:
    """Render a datetime as ``"May 16, 2026"`` — the leadership-friendly form."""
    return dt.strftime("%b %-d, %Y") if hasattr(dt, "strftime") else dt.date().isoformat()


def _fmt_range(start: datetime, end: datetime) -> str:
    """Render a date range. Drops the year on the start when both share one."""
    if start.year == end.year:
        return f"{start.strftime('%b %-d')} – {end.strftime('%b %-d, %Y')}"
    return f"{_fmt_date(start)} – {_fmt_date(end)}"


class PulseFormatter:
    """Render :class:`WeeklyPulse` instances into markdown.

    Pure function over the structured pulse — never reaches back into
    the aggregator or ranker. This separation keeps the JSON contract
    authoritative: anything the formatter shows must be in the JSON.
    """

    def render_markdown(self, pulse: WeeklyPulse) -> str:
        sections: list[str] = []

        sections.append(self._render_header(pulse))
        if pulse.executive_summary:
            sections.append(self._render_executive_summary(pulse))
        if pulse.themes:
            sections.append(self._render_key_signals(pulse))
            sections.append(self._render_themes(pulse))
        if pulse.quotes:
            sections.append(self._render_voice_of_customer(pulse))
        if pulse.actions:
            sections.append(self._render_actions(pulse))

        # Join with a horizontal-rule separator so each section has clear
        # breathing room in Google Docs. Trailing newline keeps appends clean.
        body = "\n\n---\n\n".join(s.strip() for s in sections if s.strip())
        return body + "\n"

    # -- header --------------------------------------------------------

    def _render_header(self, pulse: WeeklyPulse) -> str:
        meta = pulse.meta
        lines: list[str] = []
        lines.append("# 📊 Weekly Product Pulse")
        lines.append("")
        lines.append("#### User Review Intelligence Report")
        lines.append("")

        # Two-space trailing in each line preserves the line break when the
        # Google Docs renderer collapses single newlines into a paragraph.
        meta_lines: list[str] = []
        if meta.window_start and meta.window_end:
            meta_lines.append(
                f"**Reporting Window:** {_fmt_range(meta.window_start, meta.window_end)}"
            )
        meta_lines.append(f"**Reviews Analyzed:** {meta.total_reviews:,}")
        if meta.reports_count:
            meta_lines.append(
                f"**Reports:** {meta.reports_count} "
                f"({'platform' if meta.reports_count == 1 else 'platforms'})"
            )
        meta_lines.append(f"**Generated:** {_fmt_date(meta.generated_at)}")

        lines.append("  \n".join(meta_lines))
        return "\n".join(lines)

    # -- executive summary --------------------------------------------

    def _render_executive_summary(self, pulse: WeeklyPulse) -> str:
        return "## Executive Summary\n\n" + pulse.executive_summary.strip()

    # -- key signals table --------------------------------------------

    def _render_key_signals(self, pulse: WeeklyPulse) -> str:
        themes = pulse.themes
        positive = [t for t in themes if t.sentiment == "positive"]
        negative = [t for t in themes if t.sentiment == "negative"]

        dominant_concern = self._pick_by_prevalence(negative)
        most_praised = self._pick_by_prevalence(positive)

        rows: list[tuple[str, str]] = [
            ("Total Reviews", f"{pulse.meta.total_reviews:,}"),
            ("Positive Themes", str(len(positive))),
            ("Negative Themes", str(len(negative))),
            ("Overall Tone", _SENTIMENT_LABEL.get(pulse.overall_tone, "Neutral")),
        ]
        if dominant_concern:
            rows.append(("Dominant Concern", dominant_concern.label))
        if most_praised:
            rows.append(("Most Praised Area", most_praised.label))

        lines = ["## Key Signals", "", "| Metric | Value |", "| --- | --- |"]
        lines.extend(f"| {label} | {value} |" for label, value in rows)
        return "\n".join(lines)

    @staticmethod
    def _pick_by_prevalence(themes: list[PulseTheme]) -> PulseTheme | None:
        """Highest-prevalence theme, breaking ties by rank (lower is better)."""
        if not themes:
            return None
        order = {"high": 0, "medium": 1, "low": 2}
        return min(themes, key=lambda t: (order.get(t.prevalence, 3), t.rank))

    # -- themes --------------------------------------------------------

    def _render_themes(self, pulse: WeeklyPulse) -> str:
        quotes_by_theme = self._index_by_theme(pulse.quotes)
        actions_by_theme = self._index_by_theme(pulse.actions)

        blocks: list[str] = ["## Top Themes"]
        for theme in pulse.themes:
            blocks.append(
                self._render_theme_card(
                    theme,
                    quote=quotes_by_theme.get(theme.label),
                    action=actions_by_theme.get(theme.label),
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _index_by_theme(items: list):  # type: ignore[no-untyped-def]
        """Return ``{theme_label: first_item}`` so each card pairs cleanly.

        We use the *first* match because the ranker already orders items
        by relevance — later duplicates would be lower-signal repeats.
        """
        out: dict[str, object] = {}
        for item in items:
            label = getattr(item, "theme_label", None)
            if label and label not in out:
                out[label] = item
        return out

    def _render_theme_card(
        self,
        theme: PulseTheme,
        *,
        quote: PulseQuote | None,
        action: PulseAction | None,
    ) -> str:
        icon = _SENTIMENT_ICON.get(theme.sentiment, "⚪")
        sentiment = _SENTIMENT_LABEL.get(theme.sentiment, "Neutral")
        prevalence = _PREVALENCE_LABEL.get(theme.prevalence, theme.prevalence.title())

        lines = [
            f"### {icon} {theme.label}",
            "",
            f"_{sentiment} sentiment · {prevalence} prevalence_",
            "",
            "**Insight**",
            "",
            theme.headline.strip(),
        ]
        if quote is not None:
            lines.extend(
                [
                    "",
                    "**Representative Feedback**",
                    "",
                    f"> {quote.text.strip()}",
                ]
            )
        if action is not None:
            lines.extend(
                [
                    "",
                    "**Suggested Action**",
                    "",
                    action.text.strip(),
                ]
            )
        return "\n".join(lines)

    # -- voice of customer --------------------------------------------

    def _render_voice_of_customer(self, pulse: WeeklyPulse) -> str:
        lines = ["## Voice of Customer", ""]
        for quote in pulse.quotes:
            lines.append(f"> {quote.text.strip()}")
            if quote.theme_label:
                lines.append(f">  \n> — _on {quote.theme_label}_")
            lines.append("")
        # Drop trailing blank.
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    # -- recommended actions ------------------------------------------

    def _render_actions(self, pulse: WeeklyPulse) -> str:
        lines = ["## Recommended Product Actions", ""]
        for action in pulse.actions:
            text = action.text.strip().rstrip(".")
            if action.theme_label:
                lines.append(f"- **{action.theme_label}** — {text}.")
            else:
                lines.append(f"- {text}.")
        return "\n".join(lines)


__all__ = ["PulseFormatter", "clip_to_word_budget", "count_words"]
