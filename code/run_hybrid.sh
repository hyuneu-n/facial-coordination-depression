#!/bin/bash
# 제안 모델 v2 검증: 베이스라인 + coordination 브랜치가 베이스라인 단독을 넘는가.
# 핵심 비교는 depdetector(이미 측정됨) vs depcoord. 같은 프로토콜·같은 seed.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
GPU="$1"; shift
export CUDA_VISIBLE_DEVICES="$GPU"

for spec in "$@"; do
  c="${spec%%:*}"; m="${spec##*:}"
  echo "##### $c / $m"
  $PY -m bench.run --corpus "$c" --model "$m" --seeds 5
done
echo "########## HYBRID GPU$GPU ALLDONE"
