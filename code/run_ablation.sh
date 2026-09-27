#!/bin/bash
# Task 3 — 모듈 내부 ablation. 설계 요소를 하나씩 제거해 각 부품의 기여를 분리한다.
# full(제안) / nospd(곡면기하 제거) / notime(시간모델 제거) / nocov(coordination 제거)
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0

for c in mud3 mud3_aligned; do
  for h in depdetector blstm; do
    echo "##### $c / $h (baseline)"
    $PY -m bench.run --corpus "$c" --model "$h" --seeds 5 --tag _abl
    for m in full nospd notime nocov; do
      echo "##### $c / $h+$m"
      $PY -m bench.run --corpus "$c" --model "$h+$m" --seeds 5 --tag _abl
    done
  done
done
echo "########## ABLATION ALLDONE"
