"""Example: deliver a generated weekly pulse via the MCP server.

Run with::

    MCP_SERVER_URL=https://google-docs-gmail-mcp-server-u06s.onrender.com \
    EMAIL_TO=you@example.com \
    python -m examples.mcp_delivery_demo

The demo:

1. Reuses the two hand-built reports from ``weekly_pulse_demo`` to
   produce a :class:`WeeklyPulseArtifacts`.
2. Hands those artifacts to :class:`PulseDeliveryService`, which calls
   the external MCP server's ``append_to_doc`` + ``create_email_draft``
   endpoints.
3. Prints the resulting Google Doc URL and Gmail draft ID.

The Google integrations themselves live in the external MCP server — we
only call it over HTTP.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.services.mcp import PulseDeliveryService
from app.services.pulse import WeeklyPulseGenerator
from examples.weekly_pulse_demo import _android_report, _ios_report


async def _amain() -> int:
    configure_logging()

    if not os.getenv("MCP_SERVER_URL") and not os.getenv("MCP_SERVER_URL"):
        print(
            "ERROR: set MCP_SERVER_URL to the MCP server, "
            "e.g. https://google-docs-gmail-mcp-server-u06s.onrender.com",
            file=sys.stderr,
        )
        return 2

    if not os.getenv("EMAIL_TO"):
        print("ERROR: set EMAIL_TO to the recipient address.", file=sys.stderr)
        return 2

    window_end = datetime.now(tz=UTC)
    window_start = window_end - timedelta(days=7)

    artifacts = WeeklyPulseGenerator().generate(
        [_ios_report(), _android_report()],
        window_start=window_start,
        window_end=window_end,
        generated_at=window_end,
    )

    try:
        result = await PulseDeliveryService().deliver(artifacts)
    except AppError as exc:
        print(f"ERROR: pulse delivery failed: {exc.message}", file=sys.stderr)
        return 1

    print("=== Delivery result ===")
    print(f"Doc title:     {result.doc_title}")
    print(f"Doc URL:       {result.document_url}")
    print(f"Email to:      {result.email_to}")
    print(f"Email subject: {result.email_subject}")
    print(f"Draft ID:      {result.draft_id}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
