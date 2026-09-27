#!/bin/bash
# Task 3-재실행 — 반복 교차검증으로 ablation. 공식 분할(시험 66명) 대신
# 전체 650명을 5겹x3반복으로 돌려 검정력을 확보한다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
R=/home/hyuneun/disk_b/🟡facial-prodrome/results/bench
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
for h in depdetector blstm; do
  echo "########## host=$h (MUD3, 반복교차검증)"
  $PY -m bench.repeat_cv --corpus mud3 --repeats 3 \
    --models "$h,$h+full,$h+nospd,$h+notime,$h+nocov" \
    --out $R/ablcv_mud3_$h.csv
done
echo '########## ABLCV ALLDONE'
