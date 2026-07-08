#!/bin/bash
# π0.5 (e6) 추론 자원(GPU/CPU/RAM) 벤치마크 실행기.
#   사전조건: 다른 터미널에서 정책 서버가 떠 있어야 함 (ws://127.0.0.1:8000)
#       cd ~/E6-VLA_INFERENCE && bash run_server_v23.sh
#
#   1) move-one venv 로 측정 (websocket 클라 → 서버) → JSON
#   2) 시스템 python3(matplotlib) 로 막대그래프 PNG  (smolvla 와 동일 형식)
#
# 사용:
#   ./run_infer_benchmark.sh                      # 10회, warmup 3
#   ./run_infer_benchmark.sh --runs 20 --warmup 3
#   ./run_infer_benchmark.sh --host 127.0.0.1 --port 8000 --prompt "..."
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$HOME/move-one/min-imum/move-one/bin/python"

JSON="$HERE/infer_resource_result.json"
PNG="$HERE/infer_resource_bar.png"

# ── 1) 측정 (move-one venv; CUDA 불필요 — 클라이언트라서) ─────────────────────
echo "[run] measuring with move-one venv (websocket client) ..."
PYTHONNOUSERSITE=1 \
  "$VENV_PYTHON" "$HERE/infer_resource_benchmark.py" --out-json "$JSON" "$@"

# ── 2) 그래프 (시스템 python3 + matplotlib) ──────────────────────────────────
echo "[run] plotting with system python3 ..."
/usr/bin/python3 "$HERE/plot_infer_resource.py" --json "$JSON" --out "$PNG"

echo "[run] done."
echo "  JSON : $JSON"
echo "  PNG  : $PNG"
