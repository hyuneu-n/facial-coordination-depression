#!/bin/bash
# Task 4 — host 확대. 4개 host x 5개 데이터에 coordination 을 붙여 범용성을 확인한다.
# E-DAIC 제외: 모든 모델이 우연 수준(F1 0.17~0.41)이라 모듈 효과를 측정할 수 없다.
#   이 제외 기준은 baseline 성능으로 결정되며 제안 모델의 결과와 무관하다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=1

for c in dvlog cmdc lmvd mud3 mud3_aligned; do
  for h in tfn tamfn; do          # blstm·depdetector 는 Task3 에서 이미 돌린다
    echo "##### $c / $h (baseline)"
    $PY -m bench.run --corpus "$c" --model "$h" --seeds 5 --tag _h
    echo "##### $c / $h+full"
    $PY -m bench.run --corpus "$c" --model "$h+full" --seeds 5 --tag _h
  done
done
# dvlog·cmdc·lmvd 에 대해 blstm/depdetector 이식도 필요 (Task3 는 mud3 만 다룸)
for c in dvlog cmdc lmvd; do
  for h in blstm depdetector; do
    echo "##### $c / $h+full"
    $PY -m bench.run --corpus "$c" --model "$h+full" --seeds 5 --tag _h
  done
done
echo "########## HOSTS ALLDONE"
