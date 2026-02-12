from  utils import *

class SingleSelectiveSSM(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, gate_hidden: int = 64,
                 return_sequence: bool = False, stable_ssm: bool = True,
                 delta_max: float = 0.1, metric: str = 'logeuclid', use_rbn: bool = True, k1=5):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.return_sequence = return_sequence
        self.dim_reduced = (input_dim != state_dim)
        self.stable_ssm = stable_ssm
        self.delta_max = delta_max
        self.metric = metric
        self.use_rbn = use_rbn
        self.alpha = nn.Parameter(torch.tensor(0.0))

        if self.dim_reduced:
            self.bimap = BiMap(in_dim=input_dim, out_dim=state_dim)

        self.lam = nn.Parameter(torch.ones(state_dim))

        dim_vec = (state_dim * (state_dim + 1)) // 2
        self.gru = nn.GRU(dim_vec, gate_hidden, num_layers=1, batch_first=True)
        self.fc_delta = nn.Linear(gate_hidden, 1)
        self.fc_B = nn.Linear(gate_hidden, state_dim * state_dim)
        self.fc_C = nn.Linear(gate_hidden, state_dim * state_dim)

        if use_rbn:
            self.rbn_in = RiemannianBatchNormSPD(state_dim, k1=k1)
            self.rbn_out = RiemannianBatchNormSPD(state_dim, k1=k1)
        else:
            self.rbn_in = self.rbn_out = None

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        alpha_all = torch.sigmoid(self.alpha)
        if U.dim() == 3:
            U = U.unsqueeze(1)
        B, K, s, _ = U.shape
        assert s == self.input_dim

        if self.dim_reduced:
            U_flat = U.view(B * K, self.input_dim, self.input_dim)
            projected_flat = self.bimap(U_flat)
            projected_U = projected_flat.view(B, K, self.state_dim, self.state_dim)
        else:
            projected_U = ReEig(U)

        if self.use_rbn:
            projected_U = self.rbn_in(projected_U)

        U_logs = LogEig(projected_U.view(-1, self.state_dim, self.state_dim)).view(B, K, self.state_dim, self.state_dim)
        gate_in = tril_vectorize(U_logs)
        gru_out, _ = self.gru(gate_in)
        deltas = F.softplus(self.fc_delta(gru_out)).squeeze(-1)
        deltas = self.delta_max * torch.sigmoid(deltas)

        x = torch.eye(self.state_dim, device=U.device).unsqueeze(0).repeat(B, 1, 1) * 1e-3
        Y_seq = []

        lam = F.softplus(self.lam)

        for t in range(K):
            u_t = projected_U[:, t]
            delta_t = deltas[:, t].view(B, 1, 1)

            gru_t = gru_out[:, t]
            B_flat = self.fc_B(gru_t)
            W_B_t = B_flat.view(B, self.state_dim, self.state_dim)
            W_B_t = stiefel_retract(W_B_t)

            C_flat = self.fc_C(gru_t)
            W_C_t = C_flat.view(B, self.state_dim, self.state_dim)
            W_C_t = stiefel_retract(W_C_t)

            if self.stable_ssm:
                scales = torch.exp(-delta_t * lam.view(1, -1).unsqueeze(-1))
                A = torch.diag_embed(scales.squeeze(-1))
                Bu = W_B_t.transpose(-1, -2) @ u_t @ W_B_t
                x = A @ x @ A.transpose(-1, -2) + Bu + 1e-6 * torch.eye(self.state_dim, device=U.device)

            else:
                scales = torch.exp(-F.softplus(self.lam))
                D = torch.diag_embed(torch.sqrt(scales)).to(U.device)
                Bu = W_B_t.transpose(-1, -2) @ u_t @ W_B_t
                x = x + delta_t * (D @ x @ D.transpose(-1, -2) + Bu)

            Cx = W_C_t.transpose(-1, -2) @ x @ W_C_t
            y_t = geodesic_mix(u_t, Cx, alpha=alpha_all, metric=self.metric)
            Y_seq.append(y_t)

        Y_stack = torch.stack(Y_seq, dim=1)
        if self.use_rbn:
            Y_stack = self.rbn_out(Y_stack)

        if self.return_sequence:
            return Y_stack
        else:
            w = torch.softmax(deltas, dim=1).view(B, K, 1, 1)
            return ReEig(torch.sum(w * Y_stack, dim=1))

class SelectiveSSM_SPD_Riemann(nn.Module):
    def __init__(self, s: int = None, reduction_dims: list[int] = None, gate_hidden: int = 64,
                 return_sequence: bool = False, stable_ssm: bool = True, metric: str = 'logeuclid',
                 use_rbn: bool = True, k1 = 5):
        super().__init__()
        self.reduction_dims = reduction_dims
        self.gate_hidden = gate_hidden
        self.return_sequence = return_sequence
        self.metric = metric
        self.stable_ssm = stable_ssm
        self.use_rbn = use_rbn

        if reduction_dims is None:
            assert s is not None, "s must be provided when reduction_dims is None"
            self.single = SingleSelectiveSSM(
                input_dim=s, state_dim=s, gate_hidden=gate_hidden,
                return_sequence=return_sequence, stable_ssm=stable_ssm,
                metric=metric, use_rbn=use_rbn, k1=k1
            )
            self.is_stacked = False
        else:
            self.is_stacked = True
            self.reduction_dims = list(reduction_dims)
            self.ssms = nn.ModuleList()
            prev_dim = s
            count = 0
            for out_dim in self.reduction_dims:
                # print(count)
                if count > 0 and return_sequence is False:
                    k1 = 1
                ssm = SingleSelectiveSSM(
                    input_dim=prev_dim, state_dim=out_dim, gate_hidden=gate_hidden,
                    return_sequence=return_sequence, stable_ssm=stable_ssm,
                    metric=metric, use_rbn=use_rbn, k1=k1
                )
                self.ssms.append(ssm)
                prev_dim = out_dim
                count = count+1

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        if not self.is_stacked:
            return self.single(U)

        current = U
        if current.dim() == 3:
            current = current.unsqueeze(1)

        for ssm in self.ssms:
            current = ssm(current)

        if self.return_sequence and current.dim() == 4:
            return current
        elif not self.return_sequence and current.dim() == 4:
            return current.squeeze(1)
        else:
            return current


class MambaSPD_Attn_then_SSM(nn.Module):
    def __init__(self, n: int, window_sizes: list[int], d_state: int, bottleneck: int = 64, num_classes: int = 2,
                 print_shapes: bool = False, down_dims: list[int] = None,
                 metric: str = 'logeuclid',
                 stable_ssm: bool = True,
                 use_rbn: bool = True,
                 attn_mode: str = 'geo'):
        super().__init__()
        self.n = n
        self.window_sizes = sorted(window_sizes)
        self.s = d_state
        self.print_shapes = print_shapes
        self.metric = metric
        self.stable_ssm = stable_ssm
        self.use_rbn = use_rbn
        self.attn_mode = attn_mode

        self.local_bimaps = nn.ModuleList([BiMap(w, d_state) for w in self.window_sizes])
        self.local_pools = nn.ModuleList([SPDAttentionPool(d_state, metric=metric) for _ in self.window_sizes])

        self.global_bimap = BiMap(n, d_state)

        self.ssm1 = SelectiveSSM_SPD_Riemann(
            s=d_state, gate_hidden=64, return_sequence=True,
            stable_ssm=stable_ssm, metric=metric, use_rbn=use_rbn
        )

        if attn_mode == 'geo':
            self.self_attn = GeodesicSelfAttentionSPD(d_state)
        else:
            self.self_attn = SPDVectorSelfAttention(d_state, num_heads=4)

        if down_dims is None:
            down_dims = [64, 32, 16]
        self.down_dims = down_dims
        final_dim = down_dims[-1]
        if d_state > final_dim:
            intermediate_dim = (d_state + final_dim) // 2
            self.attn_reduction_dims = [intermediate_dim, final_dim]
        else:
            self.attn_reduction_dims = [final_dim]

        self.ssm2 = SelectiveSSM_SPD_Riemann(
            s=d_state, reduction_dims=self.attn_reduction_dims, gate_hidden=64,
            return_sequence=False, stable_ssm=stable_ssm, metric=metric, use_rbn=use_rbn
        )

        self.ssm_down = SelectiveSSM_SPD_Riemann(
            s=n, reduction_dims=down_dims, gate_hidden=64,
            return_sequence=False, stable_ssm=stable_ssm, metric=metric, use_rbn=use_rbn, k1=1
        )

        dim_vec = (final_dim * (final_dim + 1)) // 2
        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim_vec * 2, max(32, dim_vec)),
            nn.ReLU(),
            nn.Linear(max(32, dim_vec), 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_dim * final_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )

        self.rbn_pre = RiemannianBatchNormSPD(d_state) if use_rbn else None
        self.rbn_final = RiemannianBatchNormSPD(final_dim, k1=1) if use_rbn else None

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _map_blocks(self, blocks: torch.Tensor, bimap: nn.Module):

        B, K, w, _ = blocks.shape
        blocks_flat = blocks.view(B * K, w, w)
        Y_flat = bimap(blocks_flat)
        d_out = Y_flat.size(-1)
        Y = Y_flat.view(B, K, d_out, d_out)
        if d_out != self.s:
            Y_fixed = pad_or_truncate_spd(Y.view(-1, d_out, d_out), self.s)
            Y_fixed = Y_fixed.view(B, K, self.s, self.s)
            return Y_fixed
        return Y

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        alpha_all = torch.sigmoid(self.alpha)

        if X.ndim == 2:
            X = X.unsqueeze(0)
        B, n, _ = X.shape

        assert n == self.n

        local_spds = []
        prev_Y = None
        for i, w in enumerate(self.window_sizes):
            subs = SubSec_multi(X, [w])[0]
            tokens = self._map_blocks(subs, self.local_bimaps[i])
            pool = self.local_pools[i]
            if tokens.shape[-1] != pool.s:

                tokens_fixed = pad_or_truncate_spd(tokens.view(-1, tokens.shape[-1], tokens.shape[-1]), pool.s)
                tokens_fixed = tokens_fixed.view(B, tokens.shape[1], pool.s, pool.s)
                Y_i = pool(tokens_fixed)

                if pool.s != self.s:
                    Y_i = pad_or_truncate_spd(Y_i, self.s)
            else:
                Y_i = pool(tokens)
            if prev_Y is not None:
                Y_i = geodesic_mix(prev_Y, Y_i, alpha=alpha_all, metric=self.metric)

            local_spds.append(Y_i)
            prev_Y = Y_i

        G = self.global_bimap(X)
        if prev_Y is not None:

            if G.shape[-1] != self.s:
                G = pad_or_truncate_spd(G, self.s)
            G = geodesic_mix(G, prev_Y, alpha=alpha_all, metric=self.metric)


        if self.print_shapes:
            print("local_spds shapes:", [t.shape for t in local_spds])
            print("global_spd shape:", G.shape)

        if self.rbn_pre is not None:
            S_set = torch.stack(local_spds + [G], dim=1)
            S_set = self.rbn_pre(S_set)
        else:
            S_set = torch.stack(local_spds + [G], dim=1)

        seq = S_set
        seq = geodesic_mix(seq, self.ssm1(seq), alpha=alpha_all, metric=self.metric)
        seq = geodesic_mix(seq, self.self_attn(seq), alpha=alpha_all, metric=self.metric)

        y_attn_spd = self.ssm2(seq)

        if self.print_shapes:
            print("attn ssm output shape:", y_attn_spd.shape)

        G_down = self.ssm_down(X)

        if self.print_shapes:
            print("G_down (ssm_down) shape:", G_down.shape)

        v1 = tril_vectorize(LogEig(G_down))
        v2 = tril_vectorize(LogEig(y_attn_spd))
        concat = torch.cat([v1, v2], dim=1)
        gate_logits = self.fuse_mlp(concat).squeeze(-1)
        alpha = torch.sigmoid(gate_logits).view(B, 1, 1)

        y_spd = geodesic_mix(G_down, y_attn_spd, alpha=alpha, metric=self.metric)


        if self.rbn_final is not None:
            y_spd = self.rbn_final(y_spd)

        if self.print_shapes:
            print("final fused y_spd shape:", y_spd.shape)

        log_matrix = LogEig(y_spd)
        x = log_matrix.view(log_matrix.size(0), -1)
        logits = self.classifier(x)
        return logits


# ==========================================
# quick smoke test
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    n = 116
    window_sizes = [16, 32, 64, 96]
    d_state = 32
    num_classes = 2
    down_dims = [96, 64, 32, 16]

    R = torch.randn(B, n, n)
    X = R @ R.transpose(-2, -1) + torch.eye(n).unsqueeze(0) * 1e-3
    device = torch.device("cpu")

    model = MambaSPD_Attn_then_SSM(
        n=n, window_sizes=window_sizes, d_state=d_state,
        num_classes=num_classes, print_shapes=True, down_dims=down_dims,
        metric='logeuclid', stable_ssm=True, use_rbn=True, attn_mode='geo'
    ).to(device)

    X.to(device)

    logits = model(X)
    print("logits shape:", logits.shape)