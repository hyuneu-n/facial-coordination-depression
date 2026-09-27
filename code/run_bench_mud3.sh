#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
# aligned 캐시가 만들어질 때까지 대기
while [ ! -f /home/hyuneun/disk_b/🟡facial-prodrome/data/cache_mud3_aligned.npz ]; do sleep 30; done
sleep 10
# GPU0 큐가 끝날 때까지 대기
while pgrep -f 'run_bench_gpu0.sh' > /dev/null; do sleep 30; done
for c in mud3 mud3_aligned; do
  for m in blstm tfn depdetector tamfn ours; do
    echo "##### $c / $m"
    $PY -m bench.run --corpus $c --model $m --seeds 5
  done
done
echo '########## MUD3 ALLDONE'
