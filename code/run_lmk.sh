#!/bin/bash
# 2x2 설계 완성: LMVD(일상)·CMDC(임상)를 landmark 입력으로 재측정.
# 현재 '일상 vs 임상' 축과 'landmark vs AU' 축이 교란되어 있어 이를 분리한다.
set -u
PY=/home/hyuneun/disk_b/miniconda3/bin/python
DATA=/home/hyuneun/disk_b/🟡facial-prodrome/data
cd /home/hyuneun/disk_b/🟡facial-prodrome/code

GPU="$1"; CORPUS="$2"
export CUDA_VISIBLE_DEVICES="$GPU"

# 캐시 선생성 (LMVD landmark 로딩이 길다)
$PY - <<PYEOF
import numpy as np, warnings, time
warnings.filterwarnings('ignore')
from bench.data import LOADERS
import os
c = "$CORPUS"
p = "$DATA/cache_%s.npz" % c
if not os.path.exists(p):
    t0 = time.time()
    d = LOADERS[c]()
    print(c, 'N', len(d['y']), 'dep', int(d['y'].sum()), d['Xv'].shape, d['Xa'].shape,
          round(time.time()-t0, 1), 's', flush=True)
    np.savez_compressed(p, Xv=d['Xv'], Xa=d['Xa'], y=d['y'])
PYEOF

for m in blstm tfn depdetector tamfn ours; do
  echo "##### $CORPUS / $m"
  $PY -m bench.run --corpus "$CORPUS" --model "$m" --seeds 5
done
echo "########## ${CORPUS} ALLDONE"
