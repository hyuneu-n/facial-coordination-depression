#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
for m in blstm tfn depdetector ours; do
  echo "##### EARLY dvlog / $m"
  $PY -m bench.run --corpus dvlog --model $m --seeds 5 --early 0.2,0.4,0.6,0.8,1.0 --tag _e
done
echo '########## EARLY ALLDONE'
