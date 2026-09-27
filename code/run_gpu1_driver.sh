#!/bin/bash
# GPU1 — MUD3 진짜 raw(정규화 없음) 조건. confound 의존도 측정용 세 번째 조건.
# 파일 대기 없이 즉시 실행 (GPU1 큐는 LMVD 종료 후 비게 되므로 lmvd 완료만 파일로 확인).
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
CODE=/home/hyuneun/disk_b/🟡facial-prodrome/code
cd "$CODE"
export CUDA_VISIBLE_DEVICES=1

# LMVD 큐가 끝날 때까지 로그 마커로 대기 (pgrep 자기매칭 회피)
until grep -q 'GPU1 ALLDONE' bench_gpu1.log 2>/dev/null; do sleep 45; done

echo "########## MUD3 rawcoord (z-score 없음 = 위치·크기 confound 보존)"
for m in blstm tfn depdetector tamfn ours; do
  echo "##### mud3_rawcoord / $m"
  $PY -m bench.run --corpus mud3_rawcoord --model "$m" --seeds 5
done
echo "########## GPU1 DRIVER ALLDONE"
