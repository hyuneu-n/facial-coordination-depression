#!/bin/bash
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
for m in blstm tfn depdetector tamfn; do
  echo "##### $m"
  $PY -m bench.run --corpus dvlog --model $m --seeds 5 --tag _refac
done
echo '########## T1CHECK ALLDONE'
