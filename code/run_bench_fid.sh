#!/bin/bash
PY=/home/hyuneun/disk_b/miniconda3/bin/python
cd /home/hyuneun/disk_b/🟡facial-prodrome/code
export CUDA_VISIBLE_DEVICES=0
# 조기탐지 큐가 끝난 뒤에 패치 적용 (실행 중 조건 혼입 방지)
while pgrep -f 'run_bench_early.sh' > /dev/null; do sleep 30; done
cp bench/data.py bench/data.py.pre_fid; cp bench/run.py bench/run.py.pre_fid
$PY patch_fidelity.py || exit 1
rm -f /home/hyuneun/disk_b/🟡facial-prodrome/data/cache_dvlog.npz
echo '##### fidelity: ours 10seed'
$PY -m bench.run --corpus dvlog --model ours --seeds 10 --tag _fid
echo '##### fidelity: ours early'
$PY -m bench.run --corpus dvlog --model ours --seeds 5 --early 0.2,0.4,0.6,0.8,1.0 --tag _fid_e
echo '########## FID ALLDONE'
