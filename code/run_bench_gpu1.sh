#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=1
for c in cmdc edaic lmvd; do
  for m in blstm tfn depdetector tamfn ours; do
    echo "##### $c / $m"
    $PY -m bench.run --corpus $c --model $m --seeds 5
  done
done
echo '########## GPU1 ALLDONE'
