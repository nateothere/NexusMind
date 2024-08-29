"""Runnable examples - ingest, ask, custom tools, MCP client."""

from pathlib import Path

from nexusmind import NexusMind

def main() -> None:
    engine = NexusMind()
    engine.ingest("./docs")
    print(engine.collections())
    answer = engine.ask("What does the API expose?", top_k=6)
    print(answer.text)

if __name__ == "__main__":
    main()