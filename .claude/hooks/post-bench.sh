#!/usr/bin/env bash
# PostToolUse hook: reminder after running benchmark
# Fires after Bash commands containing run_local or run_modal

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except:
    print('')
" 2>/dev/null)

if [[ "$COMMAND" == *"run_local"* ]] || [[ "$COMMAND" == *"run_modal"* ]]; then
    # Check if benchmark output contains results
    HAS_RESULTS=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    resp = d.get('tool_response', '')
    if 'PASSED' in resp or 'FAILED' in resp:
        print('yes')
    else:
        print('no')
except:
    print('no')
" 2>/dev/null)

    if [[ "$HAS_RESULTS" == "yes" ]]; then
        echo "[KERNEL-OPT] 벤치마크 완료. optimize.md에 결과를 기록하려면 /project:track-opt 을 실행하세요."
    fi
fi
