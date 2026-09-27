#!/bin/bash
# ablation 을 두 번째 데이터(D-Vlog)와 confound 제거 조건에서 재확인.
# MUD3 에서 nospd > full 이 나왔으므로, 다른 조건에서도 일관된지 본다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
R=/home/hyuneun/disk_b/🟡facial-prodrome/results/bench
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=$1
C=$2
for h in depdetector blstm; do
  echo "########## $C / host=$h"
  $PY -m bench.repeat_cv --corpus $C --repeats 3 \
    --models "$h,$h+full,$h+nospd,$h+notime,$h+nocov" \
    --out $R/ablcv_${C}_$h.csv
done
echo "########## ABLCV $C ALLDONE"
