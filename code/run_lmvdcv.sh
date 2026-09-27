#!/bin/bash
# LMVD 는 4개 host 전부에서 하락 칸이 0 인 유일한 코퍼스이고 n=1556 으로 가장 크다.
# 효과가 작으므로(+0.7~1.7%p) 반복 교차검증으로 유의성을 확인한다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
R=/home/hyuneun/disk_b/🟡facial-prodrome/results/bench
export CUDA_VISIBLE_DEVICES=$1
for h in $2 $3; do
  [ -z "$h" ] && continue
  echo "########## LMVD host=$h"
  $PY -m bench.repeat_cv --corpus lmvd --repeats 3 \
    --models "$h,$h+full" --out $R/lmvdcv_$h.csv
done
echo "########## LMVDCV ALLDONE ($2 $3)"
