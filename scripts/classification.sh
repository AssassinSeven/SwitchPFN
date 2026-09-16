#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run.py --dataset "${1:-Heartbeat}" --data_root "${2:-./dataset/UEA}" "${@:3}"
