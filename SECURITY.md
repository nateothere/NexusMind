# Security Policy

NexusMind is local-first by design: no telemetry, no data egress, no
cloud round-trips. That reduces the attack surface, but local tools still
have real security considerations.

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a vulnerability

Please report vulnerabilities privately — do not open a public issue.

1. Email security@nexusmind.dev with a description, affected versions,
   and a minimal reproducer.
2. You will receive an acknowledgement within 48 hours.
3. We will confirm the issue, assess impact, and ship a fix.

We follow a 90-day disclosure policy: after a fix is released, details are
disclosed publicly.

## Security considerations for local deployments

- `NEXUSMIND_API_KEY` is **required** in production; without it the API
  refuses to serve requests from non-loopback hosts.
- Bind the API to localhost unless you are behind a reverse proxy that
  terminates TLS.
- The ingestion pipeline executes parsing libraries on untrusted input —
  run the worker container as a non-root user (the default Dockerfile
  already does).
- Model weights are downloaded over HTTPS; pin your Ollama registry if
  you require supply-chain guarantees.
- No secrets are stored by NexusMind itself; `.env` should be
  git-ignored and permissions-restricted (600).