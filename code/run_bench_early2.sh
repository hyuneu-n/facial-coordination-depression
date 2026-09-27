#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
while pgrep -f 'run_bench_fid.sh' > /dev/null; do sleep 30; done
$PY patch_segment.py || exit 1
$PY patch_varlen.py || exit 1
rm -f /home/hyuneun/disk_b/🟡facial-prodrome/data/cache_dvlog.npz
mv /home/hyuneun/disk_b/🟡facial-prodrome/results/bench/early_dvlog.csv \
   /home/hyuneun/disk_b/🟡facial-prodrome/results/bench/early_dvlog_BUGGY.csv 2>/dev/null
for m in blstm tfn depdetector ours; do
  echo "##### EARLY2 $m (원본시퀀스 절단 + 구간통제)"
  $PY -m bench.run --corpus dvlog --model $m --seeds 5 \
      --early 0.2,0.4,0.6,0.8,1.0 --seg 0:0.4,0.3:0.7,0.6:1.0 --tag _e2
done
echo '########## EARLY2 ALLDONE'
