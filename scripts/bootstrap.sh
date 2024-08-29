# NexusMind build bootstrap.
# Creates the venv, installs deps and pulls the default local models.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo "==> Creating venv at $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -U pip

echo "==> Installing NexusMind (gpu + eval + dev extras)"
"$VENV/bin/pip" install -e ".[gpu,eval,dev]"

echo "==> Installing pre-commit hooks"
"$VENV/bin/pre-commit" install

echo "==> Pulling default models (best effort)"
ollama pull qwen2.5:14b-instruct || echo "warn: ollama not available, skipping model pulls"
ollama pull bge-m3 || true

echo "==> Done. Run 'make dev' to start api + ui + worker."