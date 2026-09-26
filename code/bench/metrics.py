import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score)

KEYS = ['acc', 'prec', 'rec', 'f1', 'auc']


def report(y, p, thr=0.5):
    yh = (p > thr).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float('nan')
    return dict(acc=accuracy_score(y, yh),
                prec=precision_score(y, yh, zero_division=0),
                rec=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0),
                auc=auc)


def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
