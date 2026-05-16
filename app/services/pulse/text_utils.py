"""Bounded-text normalisation for pulse-model fields.

Pulse schemas enforce strict character limits so downstream renderers
(email, Google Docs, Slack) get predictable, well-shaped text. The LLM
upstream can — and does — produce longer strings, so every assignment
from an LLM-derived field into a Pulse model field must pass through
``clamp_text`` to guarantee schema compliance.

Why we still keep the schema constraint (rather than just relaxing it):
the cap is a *delivery contract*, not a guess. A 500-char "headline"
breaks email subject rendering, table-cell layouts, and Slack preview
truncation. Pydantic catching a too-long value is the safety net; this
module is the normal path.
"""

from __future__ import annotations

from app.core.logging import get_logger

log = get_logger("pulse.text")

ELLIPSIS = "…"

# Sentence-ending punctuation we'll try to preserve when truncating.
_SENTENCE_END = (". ", "! ", "? ")
# Trailing punctuation worth stripping after a hard cut so the
# ellipsis doesn't read as ", …" or "; …".
_TRAILING_STRIP = ",;:- –—"


def clamp_text(
    value: str | None,
    *,
    max_length: int,
    field: str,
    ellipsis: bool = True,
) -> str:
    """Return ``value`` normalised to fit within ``max_length`` chars.

    Behaviour is deterministic and side-effect-free apart from a single
    ``pulse.text.truncated`` log line when truncation actually occurs:

    * Collapses internal whitespace runs to a single space and strips
      surrounding whitespace.
    * If the cleaned value is within the limit, returns it unchanged.
    * Otherwise truncates on the latest sentence boundary that falls
      within the budget; failing that, the latest word boundary;
      failing that (single long token), a hard character cut.
    * Optionally appends a single-char ellipsis (counted against the
      budget) so readers can tell content was dropped.

    ``field`` is purely for structured logging — it identifies which
    schema field is being clamped (e.g. ``"PulseTheme.headline"``).
    """
    if value is None:
        return ""
    if max_length <= 0:
        return ""

    cleaned = " ".join(value.split())
    if len(cleaned) <= max_length:
        return cleaned

    target = max_length - len(ELLIPSIS) if ellipsis else max_length
    if target <= 0:
        # Limit too small for an ellipsis; fall back to a hard cut.
        return cleaned[:max_length]

    head = cleaned[:target]

    # Try to end on a sentence boundary if one falls in the back half
    # of the window. Anything earlier sacrifices too much content.
    floor = int(target * 0.6)
    sentence_cut = max(head.rfind(end) for end in _SENTENCE_END)
    if sentence_cut >= floor:
        # rfind returns the index of the separator; include the
        # terminating punctuation char itself.
        cut = head[: sentence_cut + 1]
    else:
        # Fall back to the last word boundary in the latter half.
        word_cut = head.rfind(" ")
        if word_cut >= int(target * 0.5):
            cut = head[:word_cut]
        else:
            cut = head

    cut = cut.rstrip(_TRAILING_STRIP)
    if not cut:
        # Pathological input (e.g. one giant word): hard cut.
        cut = cleaned[:target]

    result = f"{cut}{ELLIPSIS}" if ellipsis else cut

    log.warning(
        "pulse.text.truncated",
        field=field,
        original_length=len(value),
        cleaned_length=len(cleaned),
        clipped_length=len(result),
        max_length=max_length,
    )
    return result


__all__ = ["ELLIPSIS", "clamp_text"]
