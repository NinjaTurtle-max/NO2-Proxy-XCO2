#!/bin/bash
# Stop hook: src/·scripts/ 의 .py 가 마지막 스모크 이후 바뀐 경우에만 빠른 스모크(≈5 s)를 돌린다.
# 실패하면 exit 2 + 이유 → Claude 에게 되돌아가 수정하도록 함. 변경 없으면 아무 것도 하지 않음.
cd "$(dirname "$0")/../.." || exit 0
PY=/opt/miniconda3/envs/NO2_Proxy_XCO2/bin/python; MARK=.claude/.last_smoke
[ -f "$MARK" ] || touch -t 200001010000 "$MARK"
changed=$(find src scripts -name '*.py' -newer "$MARK" 2>/dev/null | head -1)
[ -z "$changed" ] && exit 0
out=$("$PY" -m pytest --color=no -m "not slow" tests/smoke 2>&1 | tail -15)
if [ "${PIPESTATUS[0]}" != "0" ] && ! echo "$out" | grep -qE "passed|no tests ran"; then :; fi
if echo "$out" | grep -qE "[0-9]+ failed|error"; then
  printf '{"decision":"block","reason":"스모크 테스트 실패 — 수정 전 완료 보고 금지 (리트라이 3회 초과 시 사용자에게 보고)\\n%s"}' "$(echo "$out" | sed 's/"/\\"/g' | tr '\n' ' ' | cut -c1-1500)"
  exit 0
fi
touch "$MARK"; printf '{"systemMessage":"스모크 통과: %s"}' "$(echo "$out" | grep -E "passed|skipped" | tail -1 | tr -d '"')"
