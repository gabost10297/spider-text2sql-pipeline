#!/usr/bin/env bash
# Convenience wrappers for Phase 1 Docker workflows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmd="${1:-help}"
shift || true

case "$cmd" in
  build)
    docker compose build phase1
    ;;
  download)
    docker compose run --rm phase1 python -m src.phase1.download_spider "$@"
    ;;
  explore)
    docker compose run --rm phase1 python -m src.phase1.explore_spider "$@"
    ;;
  shell)
    docker compose --profile tools run --rm shell
    ;;
  all)
    docker compose build phase1
    docker compose run --rm phase1 python -m src.phase1.download_spider
    docker compose run --rm phase1 python -m src.phase1.explore_spider
    ;;
  help|*)
    cat <<'EOF'
Phase 1 Docker helpers

  ./scripts/phase1.sh build       Build the phase1 image
  ./scripts/phase1.sh download    Download Spider (HF + SQLite DBs)
  ./scripts/phase1.sh explore     Run EDA report
  ./scripts/phase1.sh shell       Interactive bash in the container
  ./scripts/phase1.sh all         build → download → explore

Equivalent PowerShell:
  docker compose build phase1
  docker compose run --rm phase1 python -m src.phase1.download_spider
  docker compose run --rm phase1 python -m src.phase1.explore_spider
EOF
    ;;
esac
