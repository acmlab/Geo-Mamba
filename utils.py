import torch
import torch.nn as nn
import torch.nn.functional as F
from spdnet import StiefelParameter

def symmetric_part(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X + X.transpose(-2, -1))

def ReEig(X: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    w, V = torch.linalg.eigh(symmetric_part(X))
    w = torch.clamp(w, min=epsilon)
    return (V @ torch.diag_embed(w) @ V.transpose(-2, -1)).contiguous()

def LogEig(X: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    w, V = torch.linalg.eigh(X)
    w = torch.clamp(w, min=epsilon)
    return (V @ torch.diag_embed(torch.log(w)) @ V.transpose(-2, -1)).contiguous()

def ExpEig(X: torch.Tensor) -> torch.Tensor:
    w, V = torch.linalg.eigh(X)
    return (V @ torch.diag_embed(torch.exp(w)) @ V.transpose(-2, -1)).contiguous()

def MatPow(X: torch.Tensor, p: float, epsilon: float = 1e-8) -> torch.Tensor:
    w, V = torch.linalg.eigh(X)
    w = torch.clamp(w, min=epsilon)
    return (V @ torch.diag_embed(w**p) @ V.transpose(-2, -1)).contiguous()

def MatInvSqrt(X: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return MatPow(X, -0.5, epsilon)

def MatSqrt(X: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return MatPow(X, 0.5, epsilon)

def stiefel_retract(W: torch.Tensor) -> torch.Tensor:
    if W.dim() == 2:
        Q, _ = torch.linalg.qr(W)
        return Q[:, :W.shape[1]]
    elif W.dim() == 3:
        Qs = []
        for b in range(W.shape[0]):
            Q, _ = torch.linalg.qr(W[b])
            Qs.append(Q[:, :W.shape[2]])
        return torch.stack(Qs, dim=0)
    else:
        return W

def geo_mix_logeuclid(S1: torch.Tensor, S2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    L1, L2 = LogEig(S1), LogEig(S2)
    return ExpEig((1 - alpha) * L1 + alpha * L2)

def geo_mix_bures(S1: torch.Tensor, S2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    Xh = MatSqrt(S1)
    invXh = MatInvSqrt(S1)
    M = ReEig(invXh @ S2 @ invXh)
    w, V = torch.linalg.eigh(M)
    w = torch.clamp(w, min=1e-8)
    if isinstance(alpha, torch.Tensor):
        if alpha.dim() == 0:
            alpha = alpha.view(1, 1, 1)
        a = alpha
    else:
        a = torch.tensor(alpha, device=S1.device, dtype=S1.dtype).view(1, 1, 1)
    w_a = w ** a
    M_a = V @ torch.diag_embed(w_a) @ V.transpose(-2, -1)
    return ReEig(Xh @ M_a @ Xh.transpose(-2, -1))

def geodesic_mix(S1: torch.Tensor, S2: torch.Tensor, alpha, metric: str = 'logeuclid') -> torch.Tensor:
    if metric == 'logeuclid':
        return geo_mix_logeuclid(S1, S2, alpha)
    elif metric == 'bures':
        return geo_mix_bures(S1, S2, alpha)
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")

def tril_vectorize(S: torch.Tensor) -> torch.Tensor:
    if S.dim() == 3:
        B, m, _ = S.shape
        idx = torch.tril_indices(m, m, offset=0, device=S.device)
        return S[:, idx[0], idx[1]]
    elif S.dim() == 4:
        B, L, m, _ = S.shape
        idx = torch.tril_indices(m, m, offset=0, device=S.device)
        return S[:, :, idx[0], idx[1]]
    else:
        raise ValueError("Unsupported dims for tril_vectorize")

def sym_from_tril_vec(v: torch.Tensor, m: int) -> torch.Tensor:
    dim = v.dim()
    if dim == 2:
        B, _ = v.shape
        S = torch.zeros(B, m, m, device=v.device, dtype=v.dtype)
    elif dim == 3:
        B, L, _ = v.shape
        S = torch.zeros(B, L, m, m, device=v.device, dtype=v.dtype)
    else:
        raise ValueError("Unsupported dims for sym_from_tril_vec")

    tril_idx = torch.tril_indices(m, m, device=v.device)
    if dim == 2:
        S[:, tril_idx[0], tril_idx[1]] = v
    else:
        S[:, :, tril_idx[0], tril_idx[1]] = v
    S = S + S.transpose(-1, -2) - torch.diag_embed(S.diagonal(dim1=-2, dim2=-1))
    return S

class BiMap(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_channels: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_channels > 1:
            self.W = StiefelParameter(torch.Tensor(in_channels, in_dim, out_dim), requires_grad=True)
        else:
            self.W = StiefelParameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
        nn.init.orthogonal_(self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        W = self.W
        Y = W.transpose(-2, -1) @ X @ W

        return Y

def SubSec_multi(X: torch.Tensor, window_sizes: list[int]):
    B, n, _ = X.shape
    outputs = []
    for w in window_sizes:
        if w > n:
            raise ValueError("window size cannot exceed matrix size")
        subs = []
        for i in range(0, n - w + 1):
            subs.append(X[:, i:i + w, i:i + w])
        outputs.append(torch.stack(subs, dim=1))
    return outputs

class SPDToVector(nn.Module):

    def __init__(self, s: int, out_dim: int):
        super().__init__()
        dim_vec = (s * (s + 1)) // 2
        self.fc = nn.Linear(dim_vec, out_dim)
        self.s = s

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        if S.dim() == 3:
            L = LogEig(S)
            v = tril_vectorize(L)
            return self.fc(v)
        elif S.dim() == 4:
            B, K, s, _ = S.shape
            L = LogEig(S.view(-1, s, s)).view(B, K, s, s)
            v = tril_vectorize(L)  # (B,K,dim_vec)
            B, K, _ = v.shape
            return self.fc(v.view(B * K, -1)).view(B, K, -1)
        else:
            raise ValueError("Unsupported dims for SPDToVector")

class RiemannianBatchNormSPD(nn.Module):
    def __init__(self, s: int, momentum: float = 0.1, affine: bool = True, eps: float = 1e-5, k1 = 5):
        super().__init__()
        self.s = s
        self.momentum = momentum
        self.affine = affine
        self.eps = eps
        self.register_buffer('running_mean_log', torch.zeros(s, s))

        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        if X.dim() == 4:
            B, K, s, _ = X.shape
            X_flat = X.view(B * K, s, s)
        else:
            B, s, _ = X.shape
            X_flat = X
            K = 1

        L = LogEig(X_flat)


        if self.training:
            mean_log = L.mean(dim=0)
            self.running_mean_log.mul_(1 - self.momentum).add_(self.momentum * mean_log.detach())

        else:
            mean_log = self.running_mean_log


        Lc = L - mean_log
        if self.affine:
            Lc = self.gamma * Lc + self.beta

        Y_flat = ExpEig(Lc)
        if X.dim() == 4:
            return Y_flat.view(B, K, s, s)
        else:
            return Y_flat

class GeodesicSelfAttentionSPD(nn.Module):
    def __init__(self, s: int, dropout: float = 0.1):
        super().__init__()
        self.s = s
        self.dropout = nn.Dropout(dropout)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, L, s, _ = S.shape
        assert s == self.s
        Llogs = LogEig(S.view(-1, s, s)).view(B, L, s, s)
        Li = Llogs.unsqueeze(2)
        Lj = Llogs.unsqueeze(1)
        diff = Li - Lj
        score = -torch.sum(diff * diff, dim=(-1, -2)) / float(s)
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        attn_ = attn.view(B, L, L, 1, 1)
        Lout = torch.sum(attn_ * Lj, dim=2)
        Sout = ExpEig(Lout)
        return Sout

class SPDVectorSelfAttention(nn.Module):
    def __init__(self, s: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.s = s
        dim_vec = (s * (s + 1)) // 2
        self.mha = nn.MultiheadAttention(dim_vec, num_heads, dropout=dropout, batch_first=True)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, L, s, _ = S.shape
        assert s == self.s
        L_logs = LogEig(S.view(-1, s, s)).view(B, L, s, s)
        vec_in = tril_vectorize(L_logs)
        vec_out, _ = self.mha(vec_in, vec_in, vec_in)
        vec_out = vec_out + vec_in
        sym_out = sym_from_tril_vec(vec_out, s)
        spd_out = ExpEig(sym_out)
        return spd_out

class SPDAttentionPool(nn.Module):
    def __init__(self, s: int, eps: float = 1e-4, metric: str = 'logeuclid'):
        super().__init__()
        self.s = s
        self.eps = eps
        self.metric = metric
        self.Q = nn.Parameter(torch.randn(s, s) * 0.02)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, K, s, _ = S.shape
        assert s == self.s
        S_log = LogEig(S.view(-1, s, s)).view(B, K, s, s)
        Q = symmetric_part(self.Q)
        diff = S_log - Q.view(1, 1, s, s)
        score = -torch.sum(diff * diff, dim=(-1, -2)) / float(s)
        alpha = F.softmax(score, dim=1).view(B, K, 1, 1)
        Y_log = torch.sum(alpha * S_log, dim=1)
        Y = ExpEig(Y_log)
        Y = Y + self.eps * torch.eye(s, device=Y.device).unsqueeze(0)
        return Y

def pad_or_truncate_spd(X: torch.Tensor, target: int, eps: float = 1e-6) -> torch.Tensor:

    single = False
    if X.dim() == 2:
        X = X.unsqueeze(0)
        single = True
    B, s, _ = X.shape
    if s == target:
        return X.squeeze(0) if single else X
    if s < target:
        print('s < target')
        pad_size = target - s
        pads = eps * torch.eye(pad_size, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(B, 1, 1)
        top = torch.cat([X, torch.zeros(B, s, pad_size, device=X.device, dtype=X.dtype)], dim=2)
        bottom = torch.cat([torch.zeros(B, pad_size, s, device=X.device, dtype=X.dtype), pads], dim=2)
        Y = torch.cat([top, bottom], dim=1)
        return Y.squeeze(0) if single else Y
    else:
        print(f's > target')
        Y = X[:, :target, :target]
        return Y.squeeze(0) if single else Y