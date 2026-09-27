#!/bin/bash
# 가설 검증: 이식 브랜치가 저학습률에서 수렴 부족인가?
# 맞다면 에폭을 2배로 늘리면 저학습률에서도 recall 이 회복되어야 한다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
for m in depdetector depdetector+full; do
  for lr in 1e-4 3e-4; do
    echo "##### $m lr=$lr ep=120"
    $PY -m bench.run --corpus dvlog --model "$m" --seeds 5 --lr $lr --ep 120 --tag _ep120lr$lr
  done
done
echo '########## EPOCHTEST ALLDONE'
