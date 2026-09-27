#!/bin/bash
# MUD3 rawcoord — 정규화 없음 = 위치·크기 confound 완전 보존. confound 3단 비교의 첫 단계.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0

echo "########## MUD3 rawcoord (정규화 없음)"
for m in blstm tfn depdetector tamfn ours; do
  echo "##### mud3_rawcoord / $m"
  $PY -m bench.run --corpus mud3_rawcoord --model "$m" --seeds 5
done
echo "########## RAWCOORD ALLDONE"
