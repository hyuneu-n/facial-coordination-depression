#!/bin/bash
# GPU0 순차 드라이버 — pgrep 대기 없이 한 프로세스에서 전부 실행한다.
# (이전 체인 스크립트들은 pgrep -f 패턴이 자기 래퍼 명령줄과 매칭돼 데드락이 났다.)
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
CODE=/home/hyuneun/disk_b/🟡facial-prodrome/code
DATA=/home/hyuneun/disk_b/🟡facial-prodrome/data
RES=/home/hyuneun/disk_b/🟡facial-prodrome/results/bench
cd "$CODE"
export CUDA_VISIBLE_DEVICES=0

echo "########## STAGE 1: MUD3 (패치 전 코드로 실행)"
for c in mud3 mud3_aligned; do
  for m in blstm tfn depdetector tamfn ours; do
    echo "##### $c / $m"
    $PY -m bench.run --corpus "$c" --model "$m" --seeds 5
  done
done

echo "########## STAGE 2: 패치 적용 (fidelity + segment + varlen)"
cp bench/data.py bench/data.py.pre_patch
cp bench/run.py  bench/run.py.pre_patch
$PY patch_fidelity.py || exit 1
$PY patch_segment.py  || exit 1
$PY patch_varlen.py   || exit 1
rm -f "$DATA/cache_dvlog.npz"
mv "$RES/early_dvlog.csv" "$RES/early_dvlog_BUGGY.csv" 2>/dev/null

echo "########## STAGE 3: fidelity 복원판 (ours 10seed)"
$PY -m bench.run --corpus dvlog --model ours --seeds 10 --tag _fid

echo "########## STAGE 4: 조기탐지 재측정 + 구간통제 (원본 시퀀스 절단)"
for m in blstm tfn depdetector ours; do
  echo "##### EARLY2 $m"
  $PY -m bench.run --corpus dvlog --model "$m" --seeds 5 \
      --early 0.2,0.4,0.6,0.8,1.0 --seg 0:0.4,0.3:0.7,0.6:1.0 --tag _e2
done

echo "########## GPU0 DRIVER ALLDONE"
