#!/usr/bin/env python3
"""
eval_session.py — E6 실기 평가 세션 기록 검증·집계 (프로토콜 v1)

사양: docs/E6_EVAL_PROTOCOL.md
로봇·ROS2 불필요. 순수 파일 검사라 언제든 돌릴 수 있다.

  validate  <session_dir>...   스키마/필수필드/enum/단위 검사
  aggregate <session_dir>...   조건별 집계 (시도수와 시간표본수를 분리 보고)

집계는 집단(arm / control_mode / inference_location / policy_source_commit)이 섞이면
거부한다 — 서로 다른 arm 을 평균해 Tracking RMSE 를 잘못 게시한 사고(2026-08-07)를 막기 위함.
--allow-mixed 로 명시적으로만 허용한다.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

SCHEMA_VERSION = 1

OUTCOMES = (
    "grasped_and_placed",
    "grasped_only",
    "no_grasp",
    "abort_safety",
    "abort_max_steps",
    "abort_operator",
    "error",
)
GRASPED = ("grasped_and_placed", "grasped_only")
EVIDENCE_GRADES = ("measured", "visual")

TRIAL_REQUIRED = (
    "schema_version", "trial_index", "condition_id",
    "checkpoint_cli_label", "checkpoint_reported",
    "arm", "control_mode", "outcome", "evidence_grade",
)
SESSION_REQUIRED = (
    "schema_version", "session_id", "operator", "robot",
    "policy_source_commit", "inference_location",
)
# 집단을 정의하는 키 — 이게 다르면 평균을 섞지 않는다
POPULATION_KEYS = ("arm", "control_mode", "inference_location", "policy_source_commit")


def _load_session(d: Path) -> tuple[dict, list[str]]:
    p = d / "session.json"
    if not p.exists():
        return {}, [f"{d.name}: session.json 없음"]
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"{d.name}/session.json 파싱 실패: {exc}"]
    errs = [f"{d.name}/session.json: 필수 필드 없음 '{k}'"
            for k in SESSION_REQUIRED if k not in s]
    if s.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"{d.name}/session.json: schema_version="
                    f"{s.get('schema_version')} (기대 {SCHEMA_VERSION})")
    return s, errs


def _load_trials(d: Path) -> tuple[list[dict], list[str]]:
    p = d / "trials.jsonl"
    if not p.exists():
        return [], [f"{d.name}: trials.jsonl 없음"]
    trials, errs = [], []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            trials.append(json.loads(line))
        except Exception as exc:
            errs.append(f"{d.name}/trials.jsonl:{i} 파싱 실패: {exc}")
    return trials, errs


def _check_trial(d: Path, t: dict, i: int) -> list[str]:
    tag = f"{d.name}/trials.jsonl:{i}"
    errs = [f"{tag}: 필수 필드 없음 '{k}'" for k in TRIAL_REQUIRED if k not in t]
    if t.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"{tag}: schema_version={t.get('schema_version')} (기대 {SCHEMA_VERSION})")
    if t.get("outcome") not in OUTCOMES:
        errs.append(f"{tag}: outcome='{t.get('outcome')}' 은 enum 밖 {OUTCOMES}")
    if t.get("evidence_grade") not in EVIDENCE_GRADES:
        errs.append(f"{tag}: evidence_grade='{t.get('evidence_grade')}' 은 enum 밖")

    # 단위 규칙: 시간 필드는 _s 또는 _ms 로 끝나야 한다
    for k in t:
        if any(w in k for w in ("time", "elapsed", "duration", "interval", "rtt", "tick")):
            if not (k.endswith("_s") or k.endswith("_ms") or k.endswith("_wall")
                    or k.endswith("_monotonic")):
                errs.append(f"{tag}: 시간 필드 '{k}' 에 단위 접미사(_s/_ms)가 없다")

    tso = t.get("time_to_suction_on_s")
    if t.get("outcome") in GRASPED and tso is None:
        errs.append(f"{tag}: outcome={t['outcome']} 인데 time_to_suction_on_s 가 null")
    if t.get("outcome") == "no_grasp" and tso is not None:
        errs.append(f"{tag}: outcome=no_grasp 인데 time_to_suction_on_s={tso} (모순)")
    if tso is not None and (not isinstance(tso, (int, float)) or not math.isfinite(tso) or tso <= 0):
        errs.append(f"{tag}: time_to_suction_on_s={tso} 가 유효하지 않다")

    if t.get("outcome") == "abort_safety" and not t.get("outcome_reason"):
        errs.append(f"{tag}: abort_safety 인데 outcome_reason 이 비었다 "
                    f"(FAIL_SAFETY 원문을 넣을 것)")
    if t.get("evidence_grade") == "measured" and tso is None and t.get("outcome") in GRASPED:
        errs.append(f"{tag}: evidence_grade=measured 인데 계측값이 없다")
    return errs


def _pop_key(sess: dict, t: dict) -> tuple:
    return tuple(t.get(k, sess.get(k)) for k in POPULATION_KEYS)


def cmd_validate(dirs: list[Path]) -> int:
    total_err, total_trials = [], 0
    for d in dirs:
        sess, e1 = _load_session(d)
        trials, e2 = _load_trials(d)
        errs = e1 + e2
        idx = [t.get("trial_index") for t in trials]
        if len(set(idx)) != len(idx):
            errs.append(f"{d.name}: trial_index 중복 {sorted(idx)}")
        for i, t in enumerate(trials, 1):
            errs.extend(_check_trial(d, t, i))
        total_trials += len(trials)
        total_err.extend(errs)
        mark = "OK " if not errs else "FAIL"
        print(f"[{mark}] {d}  trials={len(trials)}  문제={len(errs)}")
        for e in errs[:20]:
            print(f"        - {e}")
        if len(errs) > 20:
            print(f"        ... {len(errs)-20}건 더")
    print(f"\n총 {len(dirs)} 세션 / {total_trials} trial / 문제 {len(total_err)}건")
    return 1 if total_err else 0


def cmd_aggregate(dirs: list[Path], allow_mixed: bool) -> int:
    by_cond: dict[str, list[tuple[dict, dict]]] = {}
    for d in dirs:
        sess, _ = _load_session(d)
        trials, _ = _load_trials(d)
        for t in trials:
            by_cond.setdefault(t.get("condition_id", "(condition_id 없음)"), []).append((sess, t))

    mixed = []
    for cond, rows in sorted(by_cond.items()):
        pops = {_pop_key(s, t) for s, t in rows}
        if len(pops) > 1:
            mixed.append((cond, pops))

    if mixed and not allow_mixed:
        print("거부: 한 조건 안에 서로 다른 집단이 섞여 있다. "
              f"집단 키 = {POPULATION_KEYS}")
        for cond, pops in mixed:
            print(f"  condition_id='{cond}' 에 집단 {len(pops)}종:")
            for p in sorted(pops, key=str):
                print(f"    {p}")
        print("\n의도한 것이면 --allow-mixed 를 명시할 것. "
              "(서로 다른 arm 을 평균해 RMSE 를 잘못 게시한 사고가 있었다)")
        return 2

    for cond, rows in sorted(by_cond.items()):
        ts = [t for _, t in rows]
        n = len(ts)
        n_grasped = sum(1 for t in ts if t.get("outcome") in GRASPED)
        n_done = sum(1 for t in ts if t.get("outcome") == "grasped_and_placed")
        times = [t["time_to_suction_on_s"] for t in ts
                 if isinstance(t.get("time_to_suction_on_s"), (int, float))]
        n_visual = sum(1 for t in ts if t.get("evidence_grade") == "visual")

        print(f"\n=== {cond} ===")
        print(f"  n_attempts   {n}")
        print(f"  n_grasped    {n_grasped}   (grasped_* / 집기 성공)")
        print(f"  n_completed  {n_done}   (grasped_and_placed / 완주)")
        print(f"  n_timed      {len(times)}   ← 시간 통계의 실제 분모")
        if n_visual:
            print(f"  n_visual     {n_visual}   ⚠️ 육안 관찰 — 정량 주장에 쓰지 말 것")
        if times:
            sd = st.stdev(times) if len(times) > 1 else 0.0
            print(f"  time_to_suction_on_s  mean={st.mean(times):.2f}  sd={sd:.2f}  "
                  f"p50={st.median(times):.2f}  min={min(times):.2f}  max={max(times):.2f}")
        else:
            print("  time_to_suction_on_s  (표본 없음)")
        from collections import Counter
        for k, v in sorted(Counter(t.get("outcome") for t in ts).items()):
            print(f"    outcome {k:20s} {v}")
        if n != len(times):
            print(f"  ⚠️ n_attempts({n}) != n_timed({len(times)}) — "
                  f"보고 시 반드시 분리해 쓸 것. 비율을 성공률로 환산하지 말 것")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="E6 실기 평가 기록 검증·집계 (프로토콜 v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument("dirs", nargs="+", type=Path)
        if name == "aggregate":
            sp.add_argument("--allow-mixed", action="store_true",
                            help="한 조건에 여러 집단이 섞인 것을 명시적으로 허용")
    a = ap.parse_args()
    dirs = [d for d in a.dirs if d.is_dir()]
    if not dirs:
        print("세션 디렉터리가 없다", file=sys.stderr)
        return 2
    if a.cmd == "validate":
        return cmd_validate(dirs)
    return cmd_aggregate(dirs, a.allow_mixed)


if __name__ == "__main__":
    raise SystemExit(main())
