#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
echo '########## ablation: norm=False (원본 landmark)'
$PY -m bench.run --corpus dvlog --model blstm --seeds 5 --norm 0 --tag _nonorm
for m in tfn depdetector tamfn ours; do
  echo "########## $m"
  $PY -m bench.run --corpus dvlog --model $m --seeds 5
done
echo '########## ALLDONE'
