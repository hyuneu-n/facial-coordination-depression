#!/bin/bash
# SPD 곡면 층을 제거한 변형(nospd)을 제안 설계로 놓고 전 격자 검증.
# 근거: full 대비 6번 중 5번 우세, D-Vlog 에서 full 의 손해가 사라짐.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=$1
shift
for c in "$@"; do
  for h in blstm tfn depdetector tamfn; do
    echo "##### $c / $h+nospd"
    $PY -m bench.run --corpus $c --model "$h+nospd" --seeds 5 --tag _ns
  done
done
echo "########## NOSPD ALLDONE ($*)"
