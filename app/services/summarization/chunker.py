"""Token-aware review chunker.

The chunker is a greedy packer: it walks a sequence of reviews and emits
chunks that respect both a soft token budget and a hard review-count cap.
The token estimate is intentionally cheap (``len(text) // 4``) so chunking
never blocks on tokeniser imports or per-call API overhead — actual token
usage will be checked by Groq, but for *batching* this heuristic is more
than accurate enough and is the standard rule-of-thumb for English text.

A single oversized review (one whose own estimate already exceeds the
budget) is never split: it goes into its own chunk. Splitting a review
mid-sentence destroys the very signal we want to summarise.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.domain.models import Review

_CHARS_PER_TOKEN = 4  # rule-of-thumb for English; good enough for batch sizing
# Default per-review body cap. A handful of users write 5k-word essays;
# we don't lose useful signal by clipping them — themes are repetition,
# not length — and we protect chunk budgets from a single outlier.
DEFAULT_MAX_REVIEW_BODY_CHARS = 1500
_TRUNC_MARKER = "…[truncated]"


def estimate_tokens(text: str) -> int:
    """Return a fast char-based token estimate. Always >= 1 for non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_body(body: str, max_chars: int) -> str:
    """Clip ``body`` to ``max_chars`` on a word boundary where possible."""
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    head_budget = max_chars - len(_TRUNC_MARKER)
    if head_budget <= 0:
        return body[:max_chars]
    head = body[:head_budget]
    last_space = head.rfind(" ")
    # Only fall back to the word boundary if it doesn't sacrifice too
    # much content — otherwise a single long token would clip 90% off.
    if last_space >= head_budget * 0.6:
        head = head[:last_space]
    return f"{head.rstrip()}{_TRUNC_MARKER}"


def render_review(review: Review, *, max_body_chars: int = 0) -> str:
    """Render a single review into the compact form used inside chunks.

    Format: ``[review_id|⭐rating|country] title — body``. ``title`` is
    omitted if absent. The format is stable so that prompts can rely on
    extracting ``review_id`` for citation.

    When ``max_body_chars > 0``, the review body is clipped on a word
    boundary to prevent a single pathological review from blowing the
    chunk's token budget.
    """
    head = f"[{review.review_id}|{review.rating}*|{review.country}]"
    body = _truncate_body(review.body, max_body_chars) if max_body_chars else review.body
    if review.title:
        return f"{head} {review.title} — {body}"
    return f"{head} {body}"


@dataclass(frozen=True)
class ReviewChunk:
    """A batch of rendered reviews ready to be sent to the LLM."""

    chunk_id: int
    reviews: tuple[Review, ...]
    rendered: str
    token_estimate: int

    @property
    def review_count(self) -> int:
        return len(self.reviews)


class ReviewChunker:
    """Greedy token+count bounded chunker.

    Parameters
    ----------
    target_tokens:
        Soft upper bound on the per-chunk token estimate. The chunker stops
        adding reviews to the current chunk when adding the next one would
        push it over this number (unless the chunk is still empty, in which
        case the oversized review goes in alone).
    max_reviews:
        Hard cap on reviews per chunk. Prevents pathological cases where
        many tiny reviews produce a single hard-to-summarise mega-chunk.
    """

    def __init__(
        self,
        *,
        target_tokens: int = 3500,
        max_reviews: int = 50,
        max_body_chars: int = 0,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if max_reviews <= 0:
            raise ValueError("max_reviews must be positive")
        if max_body_chars < 0:
            raise ValueError("max_body_chars must be >= 0")
        self._target_tokens = target_tokens
        self._max_reviews = max_reviews
        self._max_body_chars = max_body_chars

    def chunk(self, reviews: Iterable[Review]) -> Iterator[ReviewChunk]:
        """Yield ``ReviewChunk`` instances greedily packed from ``reviews``."""
        buf: list[Review] = []
        buf_rendered: list[str] = []
        buf_tokens = 0
        chunk_id = 0

        for review in reviews:
            rendered = render_review(review, max_body_chars=self._max_body_chars)
            tokens = estimate_tokens(rendered)
            would_exceed_tokens = buf_tokens + tokens > self._target_tokens
            would_exceed_count = len(buf) + 1 > self._max_reviews

            if buf and (would_exceed_tokens or would_exceed_count):
                yield ReviewChunk(
                    chunk_id=chunk_id,
                    reviews=tuple(buf),
                    rendered="\n".join(buf_rendered),
                    token_estimate=buf_tokens,
                )
                chunk_id += 1
                buf, buf_rendered, buf_tokens = [], [], 0

            buf.append(review)
            buf_rendered.append(rendered)
            buf_tokens += tokens

        if buf:
            yield ReviewChunk(
                chunk_id=chunk_id,
                reviews=tuple(buf),
                rendered="\n".join(buf_rendered),
                token_estimate=buf_tokens,
            )


def split_if_oversized(
    chunk: ReviewChunk,
    *,
    token_threshold: int,
    max_body_chars: int = 0,
) -> list[ReviewChunk]:
    """Split ``chunk`` into smaller sub-chunks if it exceeds ``token_threshold``.

    Defensive layer in front of the Groq call: even with the upstream
    ``max_reviews``/``target_tokens`` knobs, an unusual batch of long
    reviews can still produce a chunk whose prompt would blow the
    model's context window or push it across the TPM ceiling. This
    helper re-packs such a chunk by halving the review list until
    every fragment is under the threshold, preserving order.

    A chunk already under the threshold is returned as-is (single
    element list), so this is safe to apply unconditionally.
    """
    if token_threshold <= 0 or chunk.token_estimate <= token_threshold:
        return [chunk]
    if len(chunk.reviews) <= 1:
        # Pathological single review larger than the threshold —
        # nothing useful we can split here; the body-char cap is the
        # right lever for this case.
        return [chunk]

    mid = max(1, len(chunk.reviews) // 2)
    halves = (chunk.reviews[:mid], chunk.reviews[mid:])
    out: list[ReviewChunk] = []
    next_id = chunk.chunk_id * 100  # deterministic, conflict-free id space
    for i, half in enumerate(halves):
        rendered_parts = [render_review(r, max_body_chars=max_body_chars) for r in half]
        rendered = "\n".join(rendered_parts)
        sub = ReviewChunk(
            chunk_id=next_id + i,
            reviews=half,
            rendered=rendered,
            token_estimate=estimate_tokens(rendered),
        )
        # Recurse — a single split may still leave one half oversized.
        out.extend(
            split_if_oversized(
                sub,
                token_threshold=token_threshold,
                max_body_chars=max_body_chars,
            )
        )
    return out
