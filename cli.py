"""Typer CLI - ingest, ask, chat, serve, eval, mcp."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import typer
import uvicorn
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .engine import NexusMind
from .observability.otel import configure_logging

app = typer.Typer(name="nexusmind", help="Local-first Agentic RAG engine.", no_args_is_help=True)
console = Console()

_ENGINE: NexusMind | None = None


def _engine(config: str) -> NexusMind:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NexusMind.from_config(config)
    return _ENGINE


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File, folder or URL to ingest"),
    recursive: bool = typer.Option(False, "--recursive", "-r"),
    collection: str = typer.Option("default", "--collection"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Parse -> chunk -> embed -> graph."""
    engine = _engine(config)
    chunk_ids = engine.ingest(path, recursive=recursive, collection=collection)
    console.print(f"[green]ingested[/green] {len(chunk_ids)} chunks into '{collection}'")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to answer"),
    citations: bool = typer.Option(False, "--citations"),
    top_k: int | None = typer.Option(None, "--top-k"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Ask a single question (streams the answer)."""
    engine = _engine(config)
    answer = asyncio.run(engine.ask(query, top_k=top_k))
    with Live(console=console, refresh_per_second=12) as live:
        for token in answer.text.split():
            live.update(token + " ")
            time.sleep(0.01)
    console.print()
    if citations:
        table = Table(title="Citations")
        table.add_column("#")
        table.add_column("Source")
        table.add_column("Score")
        for i, citation in enumerate(answer.citations):
            table.add_row(str(i), citation.source, f"{citation.score:.2f}")
        console.print(table)


@app.command()
def chat(config: str = typer.Option("config.yaml", "--config")) -> None:
    """Interactive TUI chat session."""
    engine = _engine(config)
    console.print(Panel("NexusMind chat — type 'exit' to quit", border_style="cyan"))
    while True:
        question = Prompt.ask("[cyan]you[/cyan]")
        if question.strip().lower() in ("exit", "quit"):
            break
        answer = asyncio.run(engine.ask(question))
        console.print(Panel(answer.text, title="nexusmind", border_style="green"))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Start the FastAPI server."""
    from .api.app import create_app

    engine = _engine(config)
    uvicorn.run(create_app(engine), host=host, port=port)


@app.command("eval")
def eval_cmd(
    suite: str = typer.Option("rag"),
    dataset: str = typer.Option("evals/golden.jsonl", "--dataset"),
    output: str = typer.Option("report.html", "--output"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Run the eval harness against golden data."""
    from .eval.harness import load_golden, run_suite, write_report

    engine = _engine(config)
    samples = load_golden(dataset)
    result = run_suite(engine._retriever, samples, top_k=engine.retriever_cfg.top_k)  # noqa: SLF001
    write_report(result, output)
    console.print(json.dumps(result.to_dict(), indent=2))
    console.print(f"[green]report written to {output}[/green]")


@app.command("mcp")
def mcp_cmd(
    http: bool = typer.Option(False, "--http", help="serve over streamable HTTP instead of stdio"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Serve the knowledge base as an MCP server."""
    from .mcp.server import MCPServer

    engine = _engine(config)
    server = MCPServer("nexusmind", "0.3.0")
    for entry in engine.tool_schemas():
        name = entry["name"]
        schema: dict[str, Any] = entry.get("parameters", {"type": "object"})
        description = entry.get("description", "")

        def handler(args: dict[str, Any], _name: str = name) -> Any:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(engine._tools.call(_name, args))  # noqa: SLF001
            finally:
                loop.close()

        server.add_tool(name, description, schema, handler)
    if http:
        console.print("[red]streamable HTTP requires uvicorn; run via the API /mcp route[/red]")
        raise typer.Exit(1)
    server.serve_stdio()


@app.command("tools")
def tools_cmd(
    connect: str | None = typer.Option(None, "--server", help="Attach an external MCP server"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """List or connect agent tools."""
    engine = _engine(config)
    if connect:
        from .tools.registry import MCPToolClient

        client = MCPToolClient(connect)
        attached = client.attach(engine._tools)  # noqa: SLF001
        console.print(f"[green]attached[/green] {len(attached)} tools from {connect}")
        return
    for name in engine._tools.names():  # noqa: SLF001
        console.print(f"- {name}")


def main() -> None:
    configure_logging("INFO")
    app()


if __name__ == "__main__":
    main()