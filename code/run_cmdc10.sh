#!/bin/bash
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
for m in depdetector depcoord blstm blstmcoord tfn ours; do
  echo "##### cmdc / $m (10seed)"
  $PY -m bench.run --corpus cmdc --model $m --seeds 10 --tag _s10
done
echo '########## CMDC10 ALLDONE'
