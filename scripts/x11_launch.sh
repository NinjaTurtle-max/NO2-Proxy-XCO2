#!/bin/bash
# 결정 11 ① W6 런 실행기 (연구책임자 지시 10-27: dr_launch.sh 는 주 트리로 cd 하므로 쓰지 않음). 인자: $1 = out 디렉토리(experiments/x11_*), 나머지 = run_train5 추가 인자.
# 예) X0s: scripts/x11_launch.sh experiments/x11_X0s_20261027 --configs scene:0
#     X1s: … --configs scene:0 --x11   ·   X1-Ls: … --configs scene:0 --x11-leak   ·   X1t: … --configs time:2 --x11   (X0t = 기존 R0)
# 고정 인자 = DR 설정 (--avg-epoch --month-order random --avg-val clean --val-px-frac 1.0, 에폭 상한 200, patience 10), 물리만(--ablation 없음), seed 0, 2020-01~12.
# 게이트: 우리 패턴 python 0 (ps ucomm) ∧ 스왑 < 80 % 1분 간격 2회 ∧ AC (S-12). 최대 30분. 평균풍 12개월 파일이 없으면 --x11 계열은 시작하지 않음.
# 실행은 x11_run.sh 경유 (11-09 주 트리 복구판: PYTHONPATH 없이 editable 설치 경로 확인, 로그 첫 줄 = no2xco2.__file__) + /usr/bin/time -l (끝에 최대 RSS) + caffeinate -i. X11_DRY=1 이면 게이트·실행 없이 명령만 출력.
WT=$(cd "$(dirname "$0")/.." && pwd -P) || exit 1; cd "$WT" || exit 1
OUT=$1; shift; [ -n "$OUT" ] || { echo "사용: $0 <out> [run_train5 인자]"; exit 1; }
swu() { sysctl -n vm.swapusage | awk '{gsub("M","",$3); gsub("M","",$6); printf "%.0f/%.0f MB (%.1f %%)", $6, $3, 100*$6/$3}'; }
sw() { sysctl -n vm.swapusage | awk '{gsub("M","",$3); gsub("M","",$6); printf "%.1f", 100*$6/$3}'; }
ac() { pmset -g batt | head -1 | grep -q "AC Power" && echo 1 || echo 0; }
jobs_n() { ps -axo ucomm=,args= | awk '$1 ~ /^[Pp]ython/ && $0 ~ /run_train5|pytest|ct_compare|build_index|physics_rollout|diag|post_pair|dr_report|seed3_table|plot_results|p7_|post_|build_sif|index_add_bg|era5_tcwv|era5_wmean|x11_|experiments\/[^ ]*_20[0-9][0-9][0-9][0-9][0-9][0-9]\.py/' | wc -l | tr -d ' '; }
CMD=(scripts/x11_run.sh scripts/run_train5.py --months 202001-202012 --seeds 0 --epochs 200 --patience 10 --avg-epoch --month-order random --avg-val clean --val-px-frac 1.0 "$@" --out "$OUT")
if [ -n "$X11_DRY" ]; then echo "[dry] ${CMD[*]}"; exit 0; fi
[ -e "$OUT/summary.csv" ] && { echo "중단: $OUT/summary.csv 존재 (N-2)"; exit 2; }
case " $* " in *" --x11 "*|*" --x11-leak "*)
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do [ -s data/raw/era5_wmean/era5_wmean_2020$m.nc ] || { echo "중단: 평균풍 era5_wmean_2020$m.nc 없음"; exit 5; }; done;; esac
ok=0; for i in $(seq 1 30); do if [ "$(jobs_n)" = 0 ] && [ "$(echo "$(sw) < 80" | bc)" = 1 ] && [ "$(ac)" = 1 ]; then ok=$((ok+1)); [ $ok -ge 2 ] && break; else ok=0; fi; sleep 60; done
[ $ok -ge 2 ] || { echo "$(date '+%F %T') 게이트 30분 미충족 (스왑 $(swu) · python $(jobs_n) · AC $(ac))"; exit 3; }
mkdir -p "$OUT"; echo "착수 $(date '+%F %T') · 스왑 $(swu) · python 작업 $(jobs_n) · 전원 $(pmset -g batt | head -1 | sed "s/Now drawing from //") · caffeinate -i · 인자 $* · 저장소 $WT · 브랜치 $(git rev-parse --abbrev-ref HEAD) · 커밋 $(git rev-parse HEAD) · 추적 파일 변경 $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') 건 (QA §36-R: 런 중 checkout·stash·reset 금지)" | tee "$OUT/launch.txt"
caffeinate -i /usr/bin/time -l "${CMD[@]}" > "$OUT/train.log" 2>&1
rc=$?; echo "$(date '+%F %T') 종료 rc $rc · 스왑 $(swu)" | tee -a "$OUT/launch.txt"; exit $rc
