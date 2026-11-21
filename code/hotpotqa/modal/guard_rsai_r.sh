#!/bin/zsh
# Launch guard: workspace must be rsai-r and metered MTD < $25 (user cap $30).
export HOME=/Users/nilan MODAL_PROFILE=rsai-r; M=/Users/nilan/.local/bin/modal
W=$($M token info --profile rsai-r 2>/dev/null | grep -o "Workspace: [a-z0-9-]*" | awk '{print $2}')
[ "$W" = "rsai-r" ] || { echo "GUARD_FAIL workspace=$W"; exit 2; }
C=$($M billing summary --profile rsai-r --json | python3 -c "import sys,json;print(float(json.load(sys.stdin)['metered_cost']))")
echo "GUARD workspace=$W metered_usd=$C utc=$(date -u +%FT%TZ)"
python3 -c "import sys;sys.exit(0 if $C+${1:-0} < 25.0 else 3)" || { echo "GUARD_FAIL metered+projected(${1:-0})>=25"; exit 3; }
