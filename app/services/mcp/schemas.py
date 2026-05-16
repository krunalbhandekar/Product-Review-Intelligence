"""Pydantic schemas for the external MCP server contract.

Schemas mirror the server's contract. ``extra="ignore"`` so forward-
compatible server-side field additions don't break us.

Response normalisation
----------------------
The MCP server currently returns responses keyed as
``{"status": "success", "doc_id": "...", "message": "..."}`` for
``append_to_doc``. Older/preferred clients expect
``{"success": true, "doc_id": "...", "document_url": "...", "message": "..."}``.

Rather than redesigning the contract or branching at the call site, the
response schemas accept *both* shapes via a ``mode="before"`` validator
that:

* coerces a string ``status`` into a boolean ``success``,
* derives the canonical Google Docs ``document_url`` from ``doc_id`` if
  the server didn't include one.

This keeps the typed surface stable for callers (they always see
``response.success`` and ``response.document_url``) and lets the server
evolve its wire format without another round of breakage.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Canonical Google Docs "edit" URL. Apps Script / Drive both render
# the same doc under this pattern, so it's safe to fabricate when the
# MCP server returns only a doc_id.
_DOCS_URL_TEMPLATE = "https://docs.google.com/document/d/{doc_id}/edit"
# String values the MCP server uses for a successful status. Anything
# else (incl. ``error``, ``failed``, missing) coerces to False.
_TRUTHY_STATUS = frozenset({"success", "ok", "succeeded", "true"})


def _coerce_success(data: dict[str, Any]) -> None:
    """If only ``status`` is present, mirror it into a boolean ``success``."""
    if "success" in data:
        return
    raw = data.get("status")
    if isinstance(raw, bool):
        data["success"] = raw
    elif isinstance(raw, str):
        data["success"] = raw.strip().lower() in _TRUTHY_STATUS
    elif raw is None:
        # Leave ``success`` unset so pydantic raises a clear "missing
        # required field" rather than silently masking a malformed body.
        return
    else:
        data["success"] = bool(raw)


class AppendDocRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Matches the MCP server's ``AppendDocInput``: the server expects the
    # target Google Doc ID (not a human title), plus the markdown body.
    doc_id: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class AppendDocResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool
    document_url: str = Field(min_length=1)
    # Optional fields the server may or may not return. Kept on the
    # model so callers can opt into richer logging / traceability.
    doc_id: str | None = Field(default=None, max_length=200)
    message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        _coerce_success(d)
        # Derive the canonical Docs URL from ``doc_id`` when the server
        # omits ``document_url``. Only do this on success — fabricating
        # a URL for a failed call would be misleading.
        if not d.get("document_url") and d.get("success") and d.get("doc_id"):
            d["document_url"] = _DOCS_URL_TEMPLATE.format(doc_id=d["doc_id"])
        return d


class CreateEmailDraftRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Keep as str (not EmailStr) — avoids the optional ``email-validator``
    # dependency. The MCP server validates the address upstream.
    to: str = Field(min_length=3, max_length=320, pattern=r".+@.+\..+")
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1)


class CreateEmailDraftResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool
    draft_id: str = Field(min_length=1)
    message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        _coerce_success(d)
        return d


__all__ = [
    "AppendDocRequest",
    "AppendDocResponse",
    "CreateEmailDraftRequest",
    "CreateEmailDraftResponse",
]
