#!/bin/bash
# E7 정책 서버 — 터미널 1.
#
#   bash run_server_e7.sh [번들경로] [config이름] [포트]
#
# ⚠️ **경로를 하드코딩하지 않는다.** E6 의 `run_server_v*.sh` 14개가 전부
#    `/media/billye6/새 볼륨1/...` 를 박아뒀고, HDD 를 UUID 마운트로 바꾼 뒤 **전부
#    깨졌다**(TASKS.md 미해결). 같은 실수를 반복하지 않는다.
#
# 🔴 `--policy.config` 는 체크포인트 안이 아니라 **openpi 소스의 TrainConfig 이름**을
#    참조한다(`_config.get_config(name)`). 그 정의가 이 트리에 없으면 서빙이 안 된다 —
#    E6 v20 이 orphan 이었던 이유다. `pi05_e7_grounded_lora` / `pi05_e7_grounded_v2_lora`
#    는 2026-08-12 에 설치했다.
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"

# 2026-08-19: v2 번들. **08-15 카메라 이동 이후** 코퍼스(59 에피소드 / 24,964 프레임)로
# 학습한 첫 번들이다. 카메라가 옮겨졌으므로 **v1 번들을 이 리그에 쓰면 안 된다** —
# 관측 분포가 다르고, norm_stats 도 다른 리그·다른 divisor(train 47개)로 계산됐다.
#   v1     /mnt/robotdata/e7_fix_v2/e7_bundle_v1_merged        pi05_e7_grounded_lora
#   v2_60  /mnt/robotdata/Dobot/e7_v2_60_bundle                pi05_e7_v2_60_lora
#   v2_120 /mnt/robotdata/Dobot/e7_v2_120_step20000_bundle     pi05_e7_v2_120_lora  ← 지금
#          119 에피소드 · **간판 배치 4개** · 홀드아웃 24개로 과적합 없음 확인
# 둘 다 848텐서·attn_vec_merge=head_summed 라 **구조로는 구분되지 않는다.**
# policy_node 의 지문 핀과 asset_id 검사가 갈라놓는다.
#   STEP 11 A/B (2026-09-02) — 아래 둘을 번갈아 돈다
#   C      /mnt/robotdata/e7_bundle_C_20260901       pi05_e7_v2_120_sg     frame_delta
#   sgrel  /mnt/robotdata/e7_bundle_sgrel_20260901   pi05_e7_v2_120_sgrel  chunk_relative
BUNDLE="${1:-/mnt/robotdata/e7_bundle_C_20260901}"

# 🔴 CONFIG 는 **번들의 manifest 에서 읽는다.** 인자로 따로 받으면 번들과 config 가
#    어긋난 조합이 성립하고, A/B 처럼 둘을 번갈아 도는 실험에서 그건 한 번의 오타로
#    조용히 일어난다. manifest 는 번들이 자기 자신에 대해 적은 값이라 갈릴 수 없다.
#
#    ⚠️ 2번째 인자로 넘기면 그것이 이긴다 — 다만 manifest 와 다르면 **경고**한다.
#       (구버전 번들이나 디버깅용 탈출구)
_MANIFEST_CONFIG=""
if [ -f "$BUNDLE/manifest.json" ]; then
    _MANIFEST_CONFIG=$(python3 -c "
import json,sys
try: print(json.load(open('$BUNDLE/manifest.json')).get('config_name') or '')
except Exception: print('')" 2>/dev/null)
fi
CONFIG="${2:-$_MANIFEST_CONFIG}"
if [ -z "$CONFIG" ]; then
    echo "[오류] config 를 정할 수 없다 — $BUNDLE/manifest.json 에 config_name 이 없고"
    echo "       2번째 인자도 안 왔다. 번들이 온전한지 확인할 것."
    exit 1
fi
if [ -n "$_MANIFEST_CONFIG" ] && [ "$CONFIG" != "$_MANIFEST_CONFIG" ]; then
    echo "[경고] config 가 manifest 와 다르다"
    echo "       manifest : $_MANIFEST_CONFIG"
    echo "       사용값   : $CONFIG   ← 인자로 덮어썼다"
    echo "       ⚠️ asset_id 가 다르면 norm_stats 를 못 찾아 죽는다. 의도한 것인지 확인할 것."
fi
# CAG(Semantic Action Guidance).
# 🔴 **값의 truth 는 번들의 `cag_config.json`** 이다. 여기 넘기는 값은 덮어쓰지 않고
#    **대조만** 한다 — 다르면 서버가 기동을 거부한다. 안 넘기면 파일 값으로 돈다.
#    ω 는 번들의 속성이고, 24/30 은 "이 가중치 + 이 ω" 조합에서 나온 값이다.
# ⚠️ 3번째 인자는 원래 PORT 다(아래). CAG 는 **4번째**로 받는다 — 3번을 뺏으면
#    기존 호출이 포트를 3.0 으로 읽고 죽는다(실제로 겪었다).
CAG_OMEGA="${4:-}"
PORT="${3:-8000}"

if [ ! -d "$BUNDLE" ]; then
    echo "[오류] 번들 폴더 없음: $BUNDLE"
    echo "       HDD 가 마운트돼 있는지 확인:  findmnt /mnt/robotdata"
    exit 1
fi
# model.safetensors 의 **존재**가 PyTorch 경로를 고른다(policy_config.py).
# 없으면 openpi 가 JAX 경로(`params/`)로 가서 엉뚱한 에러를 낸다.
if [ ! -f "$BUNDLE/model.safetensors" ]; then
    echo "[오류] $BUNDLE/model.safetensors 없음 — PyTorch 번들이 아니다"
    exit 1
fi
# norm_stats 는 루트가 아니라 asset_id 아래. 없으면 FileNotFoundError 로 죽는다
# (조용히 degrade 하지 않는다 — 학습서버 확인).
# ⚠️ asset_id 는 `--policy.config` 가 가리키는 TrainConfig 가 정한다. 여기를 고치면
#    openpi `config.py` 의 AssetsConfig(asset_id=...) 와 `policy_node.EXPECTED_ASSET_ID`
#    도 같이 고쳐야 한다. 셋이 갈라지면 로드에서 FileNotFoundError 다.
#
# 🔴 **번들에서 직접 찾는다** (2026-08-20). 경로를 박아두면 옛 번들을 되돌려 띄울 때
#    "norm_stats 없음"으로 막힌다(실제로 겪음 — v2_60 을 다시 띄우려다 걸렸다).
#    `assets/local/*/norm_stats.json` 이 정확히 하나여야 한다. 여러 개면 어느 것을
#    쓸지 우리가 정할 문제가 아니라 번들이 잘못된 것이므로 거부한다.
mapfile -t _NSCAND < <(find "$BUNDLE/assets" -name norm_stats.json 2>/dev/null | sort)
if [ "${#_NSCAND[@]}" -ne 1 ]; then
    echo "[오류] norm_stats 가 ${#_NSCAND[@]}개 — 정확히 1개여야 한다"
    printf '        %s\n' "${_NSCAND[@]}"
    echo "        찾은 위치: $BUNDLE/assets/local/<asset_id>/norm_stats.json"
    exit 1
fi
NS="${_NSCAND[0]}"
echo " norm_stats: ${NS#$BUNDLE/}"

source ~/move-one/min-imum/move-one/bin/activate
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"
_NV="$HOME/move-one/min-imum/move-one/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$_NV/cusparselt/lib:$_NV/nccl/lib:$_NV/nvshmem/lib:$_NV/cu12/lib:$LD_LIBRARY_PATH"
# ⚠️ 이게 없으면 `import torch` 자체가 libcudss.so.0 로 실패한다.
export TORCHDYNAMO_DISABLE=1
# ⚠️ 코드 드롭(2026-08-12)이 `pytorch_compile_mode` 기본값을 None → "max-autotune" 으로
#    바꿨다. 첫 추론이 4.4초, 이후 2.1초다(실측). 지연을 잴 때 워밍업을 분리할 것.

# 지금 서빙하는 번들의 지문을 고정 경로에 남긴다. inference_bridge 가 이걸 읽어
# 롤아웃 로그에 싣는다 — 그게 없어서 run→번들 매핑이 사후 추론이 됐다(08-12~13).
# 헤더 + expert o_proj 18개만 읽으므로 75MB, 기동을 지연시키지 않는다.
FP=/tmp/e7_policy_bundle.json
PYTHONPATH="$HOME/xarm_quest_ws/src/xarm_vla_collector:$PYTHONPATH" \
    python3 -m xarm_vla_collector.bundle_fingerprint "$BUNDLE" > "$FP" 2>/dev/null \
    && echo " 지문   : $(python3 -c "import json;d=json.load(open('$FP'));print(d['discriminant_sha256'][:16], '·', d['n_tensors'], '텐서 ·', d['attn_vec_merge'])")" \
    || { echo " ⚠️ 지문 계산 실패 — 롤아웃에 번들 식별자가 안 남는다"; rm -f "$FP"; }

echo "=============================="
echo " E7 정책 서버"
echo " config : $CONFIG"
echo " bundle : $BUNDLE"
echo " port   : $PORT"
echo " 계약   : 이미지 3슬롯(HIK/ZED/라벨) · state 7D 원시 각도(도) · action (16,7) delta"
echo " 실측   : 워밍업 4449ms → 정상상태 2095 ± 70ms (n=10, Jetson)"
echo "=============================="
echo ""

exec python "$REPO/scripts/serve_policy.py" \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$BUNDLE" \
    ${CAG_OMEGA:+--policy.cag-omega "$CAG_OMEGA"}
