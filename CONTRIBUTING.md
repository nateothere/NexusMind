# Contributing to NexusMind

Thanks for considering a contribution. NexusMind is a local-first project —
no telemetry, no cloud dependencies — and we keep the contribution bar
high on purpose: retrieval changes are treated like production code.

## Development setup

```bash
make setup            # python venv + deps + pre-commit + ollama models
make test             # pytest (unit + integration + eval smoke)
make lint             # ruff check + ruff format --check + mypy --strict
```

## Branch naming

- `feat/<slug>` — new capability
- `fix/<slug>` — bug fix
- `docs/<slug>` — documentation only
- `perf/<slug>` — performance work
- `chore/<slug>` — tooling, CI, packaging

## Commit style

Conventional Commits: `type(scope): subject` in lowercase, e.g.
`feat(retrieval): add rrf normalization`. Keep commits small and
single-purpose.

## Pull request checklist

- [ ] `make lint` passes (ruff + mypy --strict)
- [ ] `make test` passes, coverage >= 85%
- [ ] New retrieval logic includes a golden-dataset eval run
- [ ] CI is green on Python 3.11 and 3.12
- [ ] README updated if user-facing behavior changed

## The eval gate

Any PR that changes retrieval behavior (tokenization, chunking, fusion,
reranking, graph expansion) must include the before/after numbers from
`nexusmind eval run --suite rag`. If faithfulness drops below the
threshold in `.codecov.yml`, the PR is blocked until it recovers.

## Local models

Tests never require network access or model downloads. The unit suite
runs against stub clients; the integration suite (marked `@pytest.mark.integration`)
needs Ollama running locally with `bge-m3` and `qwen2.5:14b-instruct`.

## Questions

Open a discussion or hop into Discord (see SUPPORT.md). We respond
faster than the CI queue drains.