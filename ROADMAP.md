# Roadmap

This is the public roadmap. Items move between quarters based on
community demand and maintainer capacity. Proposals go through the
RFC process (see below) before they land in a quarter.

## Legend

- [x] shipped
- [ ] planned
- (RFC) under discussion

## 2026 Q1 — Retrieval depth

- [ ] Leiden community summaries over the Kuzu graph (GraphRAG v2) (RFC)
- [ ] Late chunking + ColBERT late-interaction index
- [ ] Query decomposition with sub-question re-fusion
- [ ] Hybrid quantized embeddings (int8) for smaller RAM footprints

## 2026 Q2 — Serving

- [ ] Speculative decoding profiles for vLLM
- [ ] Web UI v2 (Next.js) — design RFC
- [ ] Distributed ingestion workers (Ray) (RFC)
- [ ] RBAC for multi-tenant collections

## 2026 Q3 — Voice & tuning

- [ ] Voice mode (faster-whisper + XTTS)
- [ ] LoRA fine-tuning loop on your own corpus (RFC)
- [ ] Self-hosted model registry with pinning

## 2026 Q4 — Hardening

- [ ] Fuzz harness for parsers
- [ ] Offline evaluation packs (no model needed to smoke-test retrieval)
- [ ] 1.0 stabilization pass: API freeze, migration tooling

## RFC process

1. Open a discussion with `[RFC]` in the title.
2. Maintainers label it `rfc`, community review runs 2 weeks.
3. Accepted RFCs are linked from this roadmap and scheduled into a quarter.

## How to influence the roadmap

- Upvote or comment on the relevant RFC
- Bring benchmark numbers (the eval harness makes this cheap)
- Sponsor an item via GitHub Sponsors (FUNDING.yml)

Status page: see [ROADMAP.md](ROADMAP.md) is the source of truth; the
project board mirrors it.