#!/usr/bin/env bash
# Release helper: tag the current version from VERSION and push.
# GitHub Actions (docker.yml) builds the image on the tag.

set -euo pipefail

VERSION="$(cat VERSION)"
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
  echo "tag v$VERSION already exists" >&2
  exit 1
fi

git tag -a "v$VERSION" -m "NexusMind v$VERSION"
git push origin "v$VERSION"
echo "tagged and pushed v$VERSION"