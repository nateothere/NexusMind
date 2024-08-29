"""Custom tool registration - run: python examples/custom_tool.py"""

import asyncio

from nexusmind import NexusMind
from nexusmind.tools import tool


@tool
def get_ticket_status(ticket_id: str) -> dict:
    """Fetch the current status of a Jira ticket."""
    # Replace with your real integration.
    return {"ticket_id": ticket_id, "status": "in_progress"}


async def main() -> None:
    engine = NexusMind()
    engine.register_tool(get_ticket_status)
    print(engine.tool_schemas())


if __name__ == "__main__":
    asyncio.run(main())