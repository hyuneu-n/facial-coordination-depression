"""bench/models.py — 공개 베이스라인 재구현 + 제안 모델.
모든 모델: forward(xv, xa) -> (B,) logit.   xv:(B,T,Dv)  xa:(B,T,Da)
"""
import torch, torch.nn as nn, numpy as np
from sklearn.covariance import ledoit_wolf


# ------------------------------------------------------------------ BLSTM
class BLSTM(nn.Module):
    """모달별 Bidirectional LSTM → 시간평균 → concat → FC. (2019 baseline)"""

    def __init__(self, dv, da, h=64):
        super().__init__()
        self.lv = nn.LSTM(dv, h, batch_first=True, bidirectional=True)
        self.la = nn.LSTM(da, h, batch_first=True, bidirectional=True)
        self.feat_dim = 4 * h
        self.cls = nn.Sequential(nn.Linear(4 * h, 64), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(64, 1))

    def features(self, xv, xa):
        ov, _ = self.lv(xv)
        oa, _ = self.la(xa)
        return torch.cat([ov.mean(1), oa.mean(1)], 1)

    def forward(self, xv, xa):
        return self.cls(self.features(xv, xa)).squeeze(-1)


# -------------------------------------------------------------------- TFN
class TFN(nn.Module):
    """Tensor Fusion Network (Zadeh et al., EMNLP 2017).
    모달 임베딩에 1을 붙여 외적 → (h+1)^2 상호작용 텐서 → FC."""

    def __init__(self, dv, da, h=16):
        super().__init__()
        self.lv = nn.LSTM(dv, h, batch_first=True)
        self.la = nn.LSTM(da, h, batch_first=True)
        self.feat_dim = (h + 1) ** 2
        self.cls = nn.Sequential(nn.Linear((h + 1) ** 2, 64), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(64, 1))

    def features(self, xv, xa):
        ov, _ = self.lv(xv)
        oa, _ = self.la(xa)
        one = torch.ones(ov.size(0), 1, device=ov.device)
        ev = torch.cat([ov.mean(1), one], 1)
        ea = torch.cat([oa.mean(1), one], 1)
        return torch.bmm(ev.unsqueeze(2), ea.unsqueeze(1)).flatten(1)

    def forward(self, xv, xa):
        return self.cls(self.features(xv, xa)).squeeze(-1)


# ------------------------------------------------------------ DepDetector
class DepDetector(nn.Module):
    """D-Vlog 원논문 (Yoon et al., AAAI 2022): 모달별 Transformer encoder
    + cross-modal attention (v<-a, a<-v) + detector."""

    def __init__(self, dv, da, h=64, nhead=4, nlayer=2, maxT=256):
        super().__init__()
        self.pv, self.pa = nn.Linear(dv, h), nn.Linear(da, h)
        self.pos = nn.Parameter(torch.randn(1, maxT, h) * 0.02)
        mk = lambda: nn.TransformerEncoderLayer(h, nhead, h * 2, 0.1,
                                                batch_first=True)
        self.ev = nn.TransformerEncoder(mk(), nlayer)
        self.ea = nn.TransformerEncoder(mk(), nlayer)
        self.xva = nn.MultiheadAttention(h, nhead, batch_first=True)
        self.xav = nn.MultiheadAttention(h, nhead, batch_first=True)
        self.feat_dim = 2 * h
        self.cls = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(64, 1))

    def features(self, xv, xa):
        v = self.ev(self.pv(xv) + self.pos[:, :xv.size(1)])
        a = self.ea(self.pa(xa) + self.pos[:, :xa.size(1)])
        v2, _ = self.xva(v, a, a)
        a2, _ = self.xav(a, v, v)
        return torch.cat([v2.mean(1), a2.mean(1)], 1)

    def forward(self, xv, xa):
        return self.cls(self.features(xv, xa)).squeeze(-1)


# ------------------------------------------------------------------ TAMFN
class _GTCN(nn.Module):
    """Global-information TCN: dilated TCN(지역) + 전역 요약 결합."""

    def __init__(self, din, h=64):
        super().__init__()
        self.c1 = nn.Conv1d(din, h, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(h, h, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(h, h, 3, padding=4, dilation=4)
        self.g = nn.Linear(din, h)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = torch.relu(self.c1(z))
        z = torch.relu(self.c2(z))
        z = torch.relu(self.c3(z))
        return z.transpose(1, 2) + self.g(x.mean(1)).unsqueeze(1)


class TAMFN(nn.Module):
    """Time-Aware Attention Multimodal Fusion Network (Zhou et al., TNSRE 2023).
    GTCN(전역결합 TCN) + IFE(모달간 초기 상호작용) + TAMF(시간인지 attention 융합)."""

    def __init__(self, dv, da, h=64):
        super().__init__()
        self.gv, self.ga = _GTCN(dv, h), _GTCN(da, h)
        self.ife = nn.Conv1d(dv + da, h, 3, padding=1)
        self.tw = nn.Linear(3 * h, 3)
        self.feat_dim = 3 * h
        self.cls = nn.Sequential(nn.Linear(3 * h, 64), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(64, 1))

    def features(self, xv, xa):
        v, a = self.gv(xv), self.ga(xa)
        e = torch.relu(self.ife(torch.cat([xv, xa], 2).transpose(1, 2))).transpose(1, 2)
        cat = torch.cat([v, a, e], 2)
        w = torch.softmax(self.tw(cat), 2)
        fused = torch.cat([v * w[..., 0:1], a * w[..., 1:2], e * w[..., 2:3]], 2)
        return fused.mean(1)

    def forward(self, xv, xa):
        return self.cls(self.features(xv, xa)).squeeze(-1)


# ------------------------------------------------------------------- Ours
def make_covs(Xv, pca, W=64, STR=32, eps=1e-3):
    """(N,T,Dv) -> (N,K,d,d) 윈도우 coordination(공분산). numpy 경로, 사전계산용."""
    d = pca.n_components_
    out = []
    for s in Xv:
        Z = pca.transform(s)
        sd = Z.std(0); sd = np.where(sd < 1e-8, 1.0, sd)
        Z = (Z - Z.mean(0)) / sd
        cs = []
        for i in range(0, Z.shape[0] - W + 1, STR):
            seg = Z[i:i + W]
            sd2 = seg.std(0); sd2 = np.where(sd2 < 1e-8, 1.0, sd2)
            seg = (seg - seg.mean(0)) / sd2
            c, _ = ledoit_wolf(seg)
            cs.append(c + eps * np.eye(d))
        out.append(np.asarray(cs))
    return np.asarray(out, np.float32)


class Ours(nn.Module):
    """제안 모델: 윈도우 coordination -> SPD 임베딩 -> GRU(시간) + 음성 1D-CNN -> concat.
    공분산은 run.py 가 사전계산해 covs 인자로 넘긴다(속도)."""

    def __init__(self, dv, da, d=20, out=16):
        super().__init__()
        self.d, self.out = d, out
        self.Wp = nn.Parameter(torch.randn(out, d) * 0.1)
        self.pre = nn.Sequential(nn.Linear(out * (out + 1) // 2, 64), nn.ReLU())
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.a = nn.Sequential(
            nn.Conv1d(da, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(48, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.cls = nn.Sequential(nn.Linear(64 + 96, 64), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(64, 1))

    def _logeig(self, M):
        I = torch.eye(self.out, device=M.device, dtype=torch.float64)
        B = (self.Wp.double() @ M.double() @ self.Wp.double().transpose(0, 1)) + 1e-2 * I
        ev, U = torch.linalg.eigh(B)
        ev = ev.clamp(min=1e-3)
        M2 = U @ torch.diag_embed(ev) @ U.transpose(1, 2)
        ev2, U2 = torch.linalg.eigh(M2 + 1e-4 * I)
        L = U2 @ torch.diag_embed(torch.log(ev2.clamp(min=1e-4))) @ U2.transpose(1, 2)
        idx = torch.triu_indices(self.out, self.out)
        return L[:, idx[0], idx[1]].float()

    def forward(self, xv, xa, covs=None):
        if covs is None:
            raise RuntimeError('Ours 는 사전계산된 covs 가 필요하다 (run.py prep_ours)')
        B, K = covs.shape[0], covs.shape[1]
        emb = self.pre(self._logeig(covs.reshape(B * K, self.d, self.d)).reshape(B, K, -1))
        o, _ = self.gru(emb)
        af = self.a(xa.transpose(1, 2)).squeeze(-1)
        return self.cls(torch.cat([o.mean(1), af], 1)).squeeze(-1)


class CoordBranch(nn.Module):
    """coordination 인코더. mode 로 설계 요소를 하나씩 제거해 각 부품의 기여를 분리한다.

    full   : 윈도우 공분산 -> SPD LogEig -> GRU          (제안)
    nospd  : 윈도우 공분산 상삼각을 그대로 Linear -> GRU   (곡면 기하 제거)
    notime : 윈도우 공분산 -> SPD LogEig -> 평균          (시간 모델링 제거)
    nocov  : 공분산 대신 윈도우별 특징 평균 -> GRU         (coordination 자체 제거)
    """

    MODES = ('full', 'nospd', 'notime', 'nocov')

    def __init__(self, d=20, out=16, h=32, mode='full'):
        super().__init__()
        assert mode in self.MODES, mode
        self.d, self.out, self.mode = d, out, mode
        if mode == 'nocov':
            in_dim = d
        elif mode == 'nospd':
            in_dim = d * (d + 1) // 2
        else:
            self.Wp = nn.Parameter(torch.randn(out, d) * 0.1)
            in_dim = out * (out + 1) // 2
        self.pre = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        if mode != 'notime':
            self.gru = nn.GRU(64, h, batch_first=True, bidirectional=True)
            self.odim = 2 * h
        else:
            self.odim = 64

    def _logeig(self, M):
        I = torch.eye(self.out, device=M.device, dtype=torch.float64)
        B = (self.Wp.double() @ M.double() @ self.Wp.double().transpose(0, 1)) + 1e-2 * I
        ev, U = torch.linalg.eigh(B)
        ev = ev.clamp(min=1e-3)
        M2 = U @ torch.diag_embed(ev) @ U.transpose(1, 2)
        ev2, U2 = torch.linalg.eigh(M2 + 1e-4 * I)
        L = U2 @ torch.diag_embed(torch.log(ev2.clamp(min=1e-4))) @ U2.transpose(1, 2)
        idx = torch.triu_indices(self.out, self.out)
        return L[:, idx[0], idx[1]].float()

    def forward(self, covs, wfeat=None):
        """covs:(B,K,d,d) 윈도우 공분산.  wfeat:(B,K,d) 윈도우별 특징 평균 (nocov 전용)"""
        if self.mode == 'nocov':
            if wfeat is None:
                raise RuntimeError('nocov 모드는 wfeat 이 필요하다')
            emb = self.pre(wfeat)
        elif self.mode == 'nospd':
            B, K = covs.shape[0], covs.shape[1]
            idx = torch.triu_indices(self.d, self.d)
            v = covs.reshape(B * K, self.d, self.d)[:, idx[0], idx[1]]
            emb = self.pre(v.reshape(B, K, -1))
        else:
            B, K = covs.shape[0], covs.shape[1]
            v = self._logeig(covs.reshape(B * K, self.d, self.d))
            emb = self.pre(v.reshape(B, K, -1))
        if self.mode == 'notime':
            return emb.mean(1)
        o, _ = self.gru(emb)
        return o.mean(1)


class Graft(nn.Module):
    """임의의 host 모델에 coordination 브랜치를 더한다.
    host 내부는 전혀 수정하지 않고 표현(features)만 가져다 쓰므로,
    성능 변화를 coordination 의 기여로 귀속할 수 있다."""

    needs_covs = True

    def __init__(self, dv, da, host='depdetector', mode='full', d=20, out=16):
        super().__init__()
        self.host = REGISTRY_BASE[host](dv, da)
        # host 의 분류기는 쓰지 않는다. 남겨두면 학습되지 않는 파라미터가 집계에 섞여
        # '추가 파라미터'가 실제보다 커 보이므로 제거한다.
        self.host.cls = nn.Identity()
        self.coord = CoordBranch(d=d, out=out, mode=mode)
        self.cls = nn.Sequential(
            nn.Linear(self.host.feat_dim + self.coord.odim, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, xv, xa, covs=None, wfeat=None):
        h = self.host.features(xv, xa)
        c = self.coord(covs, wfeat)
        return self.cls(torch.cat([h, c], 1)).squeeze(-1)


Ours.needs_covs = True

REGISTRY_BASE = {'blstm': BLSTM, 'tfn': TFN,
                 'depdetector': DepDetector, 'tamfn': TAMFN}


def _mk(host, mode):
    def f(dv, da, **kw):
        return Graft(dv, da, host=host, mode=mode, **kw)
    f.needs_covs = True
    return f


REGISTRY = dict(REGISTRY_BASE)
REGISTRY['ours'] = Ours
for _h in list(REGISTRY_BASE):
    for _m in CoordBranch.MODES:
        REGISTRY[f'{_h}+{_m}'] = _mk(_h, _m)
# 하위 호환 (기존 결과 재현용)
REGISTRY['depcoord'] = _mk('depdetector', 'full')
REGISTRY['blstmcoord'] = _mk('blstm', 'full')


def build(name, dv, da, **kw):
    return REGISTRY[name](dv, da, **kw)
