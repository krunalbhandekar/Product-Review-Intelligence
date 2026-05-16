"""High-level pulse delivery via the MCP server.

:class:`PulseDeliveryService` is the one-call surface for shipping a
generated :class:`WeeklyPulseArtifacts` to leadership:

1. Append the rendered markdown to a Google Doc (``/append_to_doc``).
2. Create a Gmail draft that links to the resulting doc (``/create_email_draft``).

The Doc URL returned in step 1 is woven into the email body in step 2 so
the recipient can open the full report with one click — the email itself
stays short.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings, get_settings
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.services.mcp.client import MCPClient, MCPClientError
from app.services.pulse.service import WeeklyPulseArtifacts

log = get_logger("service.mcp.delivery")

DeliveryStatus = Literal["succeeded", "partial"]


@dataclass(frozen=True)
class PulseDeliveryResult:
    """Outcome of a pulse delivery attempt.

    ``status`` is ``"succeeded"`` when both the doc append and the email
    draft worked. It is ``"partial"`` when the doc was appended but the
    email draft step failed after retries — in that case the doc URL is
    still returned so an operator can manually send the email. A failed
    doc-append is not represented here: it raises ``MCPClientError``
    because there is nothing useful to return.
    """

    status: DeliveryStatus
    document_url: str
    doc_title: str
    email_to: str
    email_subject: str
    draft_id: str | None = None
    failure_stage: str | None = None
    error: str | None = None


class PulseDeliveryService:
    """Compose ``append_to_doc`` + ``create_email_draft`` into one call.

    The service is stateless apart from configuration; pass it the
    artifacts and an optional recipient override, and it returns the
    Doc URL + draft ID.
    """

    def __init__(
        self,
        *,
        client: MCPClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    async def deliver(
        self,
        artifacts: WeeklyPulseArtifacts,
        *,
        to: str | None = None,
        doc_title: str | None = None,
        subject: str | None = None,
        idempotency_key: str | None = None,
    ) -> PulseDeliveryResult:
        recipient = to or self._settings.email_to
        if not recipient:
            raise ValidationError(
                "No recipient configured: pass ``to=`` or set EMAIL_TO."
            )

        doc_id = self._settings.google_doc_id.strip()
        if not doc_id:
            raise ValidationError(
                "No Google Doc target configured: set GOOGLE_DOC_ID."
            )

        pulse = artifacts.pulse
        title = doc_title or pulse.headline
        email_subject = subject or self._compose_subject(pulse.headline)

        bound = log.bind(
            recipient=recipient,
            doc_id=doc_id,
            doc_title=title,
        )
        bound.info("mcp.delivery.start")

        client = self._client or MCPClient(settings=self._settings)
        owns_client = self._client is None

        try:
            if owns_client:
                await client.__aenter__()

            # Suffix per-endpoint so the doc and draft don't collide on
            # the server's idempotency cache.
            doc_idem = f"{idempotency_key}:doc" if idempotency_key else None
            draft_idem = (
                f"{idempotency_key}:draft" if idempotency_key else None
            )
            doc = await client.append_to_doc(
                doc_id=doc_id,
                content=artifacts.markdown,
                idempotency_key=doc_idem,
            )
            bound.info("mcp.delivery.doc_appended", document_url=doc.document_url)

            email_body = self._compose_email_body(
                pulse_headline=pulse.headline,
                executive_summary=pulse.executive_summary,
                document_url=doc.document_url,
            )
            try:
                draft = await client.create_email_draft(
                    to=recipient,
                    subject=email_subject,
                    body=email_body,
                    idempotency_key=draft_idem,
                )
            except MCPClientError as exc:
                # Doc is already published — degrade to partial success so the
                # caller keeps the doc URL and an operator can send manually.
                bound.error(
                    "mcp.delivery.draft_failed",
                    document_url=doc.document_url,
                    status_code=exc.status_code,
                    error=str(exc),
                )
                return PulseDeliveryResult(
                    status="partial",
                    document_url=doc.document_url,
                    doc_title=title,
                    email_to=recipient,
                    email_subject=email_subject,
                    draft_id=None,
                    failure_stage="create_email_draft",
                    error=str(exc),
                )

            bound.info("mcp.delivery.draft_created", draft_id=draft.draft_id)
            return PulseDeliveryResult(
                status="succeeded",
                document_url=doc.document_url,
                draft_id=draft.draft_id,
                doc_title=title,
                email_to=recipient,
                email_subject=email_subject,
            )
        finally:
            if owns_client:
                await client.__aexit__(None, None, None)

    def _compose_subject(self, headline: str) -> str:
        prefix = self._settings.pipeline.EMAIL_SUBJECT_PREFIX.strip()
        if not prefix:
            return headline
        # Avoid double-prefixing if the headline already starts with it.
        if headline.startswith(prefix):
            return headline
        return f"{prefix} {headline}".strip()

    def _compose_email_body(
        self,
        *,
        pulse_headline: str,
        executive_summary: str,
        document_url: str,
    ) -> str:
        summary = executive_summary.strip() or "(no executive summary)"
        return (
            f"{pulse_headline}\n\n"
            f"{summary}\n\n"
            f"Full report: {document_url}\n"
        )


__all__ = ["PulseDeliveryResult", "PulseDeliveryService"]
