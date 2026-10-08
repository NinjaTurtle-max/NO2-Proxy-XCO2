#!/bin/bash
# PostToolUse(Edit|Write) hook: 편집된 파일이 src/ 또는 scripts/ 의 .py 면 ruff check. 위반 시 결과를 Claude 컨텍스트로 되돌림.
export NO_COLOR=1
cd "$(dirname "$0")/../.." || exit 0
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
case "$f" in *.py) ;; *) exit 0 ;; esac
case "$f" in */src/*|*/scripts/*|*/tests/*) ;; *) exit 0 ;; esac
out=$(/opt/miniconda3/envs/NO2_Proxy_XCO2/bin/python -m ruff check --quiet --no-fix --output-format concise "$f" 2>&1 | sed "s/\x1b\[[0-9;]*m//g")
[ -z "$out" ] && exit 0
printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"ruff 위반 (수정 후 진행):\\n%s"}}' "$(echo "$out" | sed 's/"/\\"/g' | tr '\n' ' ' | cut -c1-1500)"
