#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
echo '########## TAMFN 공정화: lr 스윕'
for lr in 3e-4 1e-4; do
  echo "##### tamfn lr=$lr"
  $PY -m bench.run --corpus dvlog --model tamfn --seeds 5 --lr $lr --tag _lr$lr
done
echo '########## D-Vlog 10seed 유의성용'
for m in blstm ours; do
  echo "##### $m 10seed"
  $PY -m bench.run --corpus dvlog --model $m --seeds 10 --tag _s10
done
echo '########## GPU0 ALLDONE'
