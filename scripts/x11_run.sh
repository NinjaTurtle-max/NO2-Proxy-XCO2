#!/bin/bash
# 결정 11 ① 실행 wrapper (11-09 주 트리 복구판). PYTHONPATH 는 건드리지 않는다 — 패키지는 이 저장소 src 를 가리키는 editable 설치.
# 실행 전 no2xco2.__file__ 이 <저장소>/src/no2xco2/ 아래가 아니면 rc 4 로 중단하고, 통과하면 첫 줄에 경로를 출력한다 (QA 확인용 로그 첫 줄).
# 사용: scripts/x11_run.sh scripts/run_train5.py --months 202001 ... > train.log 2>&1   ·   scripts/x11_run.sh -m pytest -q -m "not slow" tests/smoke
WT=$(cd "$(dirname "$0")/.." && pwd -P) || exit 1
cd "$WT" || exit 1
PY=${PY:-/opt/miniconda3/envs/NO2_Proxy_XCO2/bin/python}
f=$("$PY" -c "import no2xco2; print(no2xco2.__file__)") || { echo "중단: no2xco2 import 실패" >&2; exit 4; }
case "$f" in
  "$WT/src/no2xco2/"*) ;;
  *) echo "중단: no2xco2.__file__=$f (저장소 $WT/src 아님)" >&2; exit 4 ;;
esac
echo "no2xco2.__file__=$f"
exec "$PY" "$@"
