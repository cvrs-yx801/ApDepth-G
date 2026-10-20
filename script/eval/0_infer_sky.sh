#!/usr/bin/env bash
# Unified full-frame prediction + sky metrics; no base checkpoint files modified.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../.."
exec "${PYTHON_BIN:-python}" -B script/sky_infer_eval.py "$@"
