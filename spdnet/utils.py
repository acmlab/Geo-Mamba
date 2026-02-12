import torch

def orthogonal_projection(A, B):

    out = A - B @ A.transpose(-2, -1) @ B
    return out


def retraction(A, ref=None):
    data = A if ref is None else A + ref
    m, n = data.shape
    if m >= n:
        Q, R = data.qr()
        sign = (R.diagonal().sign() + 0.5).sign().diag_embed()
        out = Q @ sign
    else:
        Q, R = data.T.qr()
        sign = (R.diagonal().sign() + 0.5).sign().diag_embed()
        out = (Q @ sign).T
    return out