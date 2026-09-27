#!/bin/bash
# Task 5 — 공정 학습률 탐색. baseline 과 이식 모델에 동일 범위를 적용한다.
# 지금까지 모든 비교가 7e-4 고정이었는데, 이식 모델은 이 학습률에서 불안정하다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=$1
C=$2
for lr in 7e-4 3e-4 1e-4; do
  for m in depdetector depdetector+full blstm blstm+full; do
    echo "##### $C / $m / lr=$lr"
    $PY -m bench.run --corpus $C --model "$m" --seeds 5 --lr $lr --tag _lr$lr
  done
done
echo "########## LRFAIR $C ALLDONE"
