"""MCP (Google Docs + Gmail) integration layer.

This package is a thin async HTTP client over the external MCP server at
``${MCP_SERVER_URL}`` that already implements the Google Docs + Gmail
integrations. We deliberately do not re-implement those integrations
here — we only call the server's two endpoints:

* ``POST /append_to_doc``       — append a markdown report to a Google Doc
* ``POST /create_email_draft``  — create a Gmail draft

The higher-level :class:`PulseDeliveryService` wires the weekly pulse
output into "publish to doc, then draft an email pointing to the doc".
"""

from app.services.mcp.client import MCPClient, MCPClientError
from app.services.mcp.delivery import PulseDeliveryResult, PulseDeliveryService
from app.services.mcp.schemas import (
    AppendDocRequest,
    AppendDocResponse,
    CreateEmailDraftRequest,
    CreateEmailDraftResponse,
)

__all__ = [
    "AppendDocRequest",
    "AppendDocResponse",
    "CreateEmailDraftRequest",
    "CreateEmailDraftResponse",
    "MCPClient",
    "MCPClientError",
    "PulseDeliveryResult",
    "PulseDeliveryService",
]
