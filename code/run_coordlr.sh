#!/bin/bash
# 가설 검증 2: coordination 브랜치에만 높은 학습률을 주면 저학습률 붕괴가 해결되는가?
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=1
for mult in 5 10; do
  for lr in 1e-4 3e-4; do
    echo "##### depdetector+full lr=$lr coord_mult=$mult"
    $PY -m bench.run --corpus dvlog --model 'depdetector+full' --seeds 5 \
      --lr $lr --coord_lr_mult $mult --tag _cm${mult}lr$lr
  done
done
echo '########## COORDLR ALLDONE'
