#!/usr/bin/env bash
# PostToolUse hook: AST check after editing kernel.py
# Receives JSON on stdin with tool_input.filePath

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # Handle both Edit and Write tool input formats
    ti = d.get('tool_input', {})
    print(ti.get('filePath', ti.get('file_path', '')))
except:
    print('')
" 2>/dev/null)

# Only check if the edited file is kernel.py
if [[ "$FILE_PATH" == *"kernel.py"* ]]; then
    KERNEL_FILE="solution/triton/kernel.py"
    if [[ -f "$KERNEL_FILE" ]]; then
        if python3 -c "import ast; ast.parse(open('$KERNEL_FILE').read())" 2>/dev/null; then
            echo "✅ kernel.py AST syntax OK"
        else
            echo "❌ kernel.py has Python syntax errors — fix before benchmarking"
            python3 -c "import ast; ast.parse(open('$KERNEL_FILE').read())" 2>&1 || true
        fi
    fi
fi
