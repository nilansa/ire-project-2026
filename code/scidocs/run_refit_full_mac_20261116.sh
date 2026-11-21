#!/bin/bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
PY=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-ann-learning-pilot-20260907/.venv/bin/python
F=/Users/nilan/Documents/ire-project-2026/experiments/refit-beir-scidocs-20261116
T=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-ceous-training-free-rerank-20261116-0524/shared/beir-scidocs-original
OUT="$T/refit-full-mac-mps-20261116-1029"
mkdir "$OUT"
printf 'Started: %s\n' "$(date '+%F %T %Z')" > "$OUT/source-manifest.txt"
shasum -a 256 "$F/sharded.py" "$F/benchmark.py" "$T/scidocs.zip" "$T/raw-mac-mps-20261116-1013/sharded-frozen-cache.pt" "$T/models/contriever-msmarco/pytorch_model.bin" "$T/models/ms-marco-MiniLM-L6-v2/model.safetensors" >> "$OUT/source-manifest.txt"
for i in 0 1 2 3; do
    printf 'RUN_SHARD %s START %s\n' "$i" "$(date '+%F %T %Z')"
    "$PY" -u "$F/sharded.py" shard \
      --archive "$T/scidocs.zip" \
      --retriever "$T/models/contriever-msmarco" \
      --teacher "$T/models/ms-marco-MiniLM-L6-v2" \
      --cache "$T/raw-mac-mps-20261116-1013/sharded-frozen-cache.pt" \
      --shard "$i" --out "$OUT/shard-$i.json" --device mps \
      > "$OUT/shard-$i.log" 2>&1
    printf 'RUN_SHARD %s COMPLETE %s\n' "$i" "$(date '+%F %T %Z')"
    shasum -a 256 "$OUT/shard-$i.json" >> "$OUT/source-manifest.txt"
done
"$PY" "$F/sharded.py" merge --archive "$T/scidocs.zip" \
  --inputs "$OUT/shard-0.json" "$OUT/shard-1.json" "$OUT/shard-2.json" "$OUT/shard-3.json" \
  --out "$OUT/merged-predictions.json"
"$PY" "$F/benchmark.py" score --archive "$T/scidocs.zip" \
  --predictions "$OUT/merged-predictions.json" --out "$OUT/recall-at-100.json"
shasum -a 256 "$OUT/merged-predictions.json" "$OUT/recall-at-100.json" >> "$OUT/source-manifest.txt"
printf 'COMPLETE_ALL_1000 %s\n' "$(date '+%F %T %Z')"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("MEASURED_RECALL_AT100",{k:round(v["mean_recall"],8) for k,v in d["methods"].items()})' "$OUT/recall-at-100.json"
