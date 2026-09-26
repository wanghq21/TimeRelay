

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


# ============================================================================
# Triton affine scan kernels
# ============================================================================

if _HAS_TRITON:

    @triton.jit
    def _affine_combine(a_l, b_l, a_r, b_r):
        """Compose two affine transforms: R o L."""
        a = a_r * a_l
        b = a_r * b_l + b_r
        return a, b


    @triton.jit
    def _affine_scan_fwd_kernel(
        A,
        Bv,
        H,
        C,
        D,
        stride_a0,
        stride_a1,
        stride_a2,
        stride_b0,
        stride_b1,
        stride_b2,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        REVERSE: tl.constexpr,
    ):
        """Inclusive affine scan along C for one traversal direction."""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        r = tl.arange(0, BLOCK_C)[:, None]
        d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]

        valid = (r < C) & (d < D)

        if REVERSE:
            pos = C - 1 - r
        else:
            pos = r

        offs_a = (
            pid_b * stride_a0
            + pos * stride_a1
            + d * stride_a2
        )
        offs_b = (
            pid_b * stride_b0
            + pos * stride_b1
            + d * stride_b2
        )

        # Padding positions are the identity affine transform (1, 0).
        a = tl.load(A + offs_a, mask=valid, other=1.0).to(tl.float32)
        b = tl.load(Bv + offs_b, mask=valid, other=0.0).to(tl.float32)

        _, h = tl.associative_scan(
            (a, b),
            axis=0,
            combine_fn=_affine_combine,
        )

        # H is allocated contiguous as [B, C, D].
        offs_h = (pid_b * C + pos) * D + d
        tl.store(H + offs_h, h, mask=valid)


    @triton.jit
    def _affine_scan_bwd_kernel(
        GRAD_H,
        A,
        H,
        GRAD_A,
        GRAD_B,
        C,
        D,
        stride_gh0,
        stride_gh1,
        stride_gh2,
        stride_a0,
        stride_a1,
        stride_a2,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        REVERSE: tl.constexpr,
    ):
        """Backward of h_i = a_i * h_prev + b_i.

        Let q_i be the upstream gradient for h_i.  Along traversal order,

            lambda_i = q_i + a_{i+1} * lambda_{i+1}
            dL/db_i   = lambda_i
            dL/da_i   = lambda_i * h_{i-1}

        The lambda recurrence is itself an affine scan, evaluated in reverse
        traversal order.
        """
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        s = tl.arange(0, BLOCK_C)[:, None]
        d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]

        valid = (s < C) & (d < D)

        # s walks in reverse traversal order.
        if REVERSE:
            # Forward traversal was C-1 -> 0, so backward traversal is 0 -> C-1.
            pos = s
            next_pos = pos - 1
            prev_state_pos = pos + 1
            has_next = valid & (s > 0)
            has_prev_state = valid & (pos < C - 1)
        else:
            # Forward traversal was 0 -> C-1, so backward traversal is C-1 -> 0.
            pos = C - 1 - s
            next_pos = pos + 1
            prev_state_pos = pos - 1
            has_next = valid & (s > 0)
            has_prev_state = valid & (pos > 0)

        offs_q = (
            pid_b * stride_gh0
            + pos * stride_gh1
            + d * stride_gh2
        )
        q = tl.load(GRAD_H + offs_q, mask=valid, other=0.0).to(tl.float32)

        offs_a_next = (
            pid_b * stride_a0
            + next_pos * stride_a1
            + d * stride_a2
        )

        # First element of the reverse scan has coefficient 0, giving
        # lambda_last = q_last.  Padding after C uses identity coefficient 1.
        coeff = tl.load(A + offs_a_next, mask=has_next, other=0.0).to(tl.float32)
        coeff = tl.where(valid, coeff, 1.0)

        _, lam = tl.associative_scan(
            (coeff, q),
            axis=0,
            combine_fn=_affine_combine,
        )

        # h_{i-1} in traversal order. H is contiguous [B, C, D].
        offs_h_prev = (pid_b * C + prev_state_pos) * D + d
        h_prev = tl.load(
            H + offs_h_prev,
            mask=has_prev_state,
            other=0.0,
        ).to(tl.float32)

        grad_a = lam * h_prev
        grad_b = lam

        # Gradient outputs are contiguous [B, C, D].
        offs_out = (pid_b * C + pos) * D + d
        tl.store(GRAD_A + offs_out, grad_a, mask=valid)
        tl.store(GRAD_B + offs_out, grad_b, mask=valid)


def _scan_launch_meta(C: int):
    block_c = 1 << (C - 1).bit_length()

    # A conservative heuristic that keeps the scan tile reasonably sized while
    # still vectorizing across D for coalesced memory access.
    if block_c >= 1024:
        block_d = 4
        num_warps = 8
    elif block_c >= 512:
        block_d = 4
        num_warps = 8
    elif block_c >= 256:
        block_d = 8
        num_warps = 8
    elif block_c >= 128:
        block_d = 8
        num_warps = 4
    else:
        block_d = 16
        num_warps = 4

    return block_c, block_d, num_warps


def _launch_affine_forward(a, b, reverse: bool):
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not installed.")
    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("Triton affine scan requires CUDA tensors.")
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must have the same [B, C, D] shape.")

    B, C, D = a.shape
    h = torch.empty((B, C, D), device=a.device, dtype=a.dtype)

    block_c, block_d, num_warps = _scan_launch_meta(C)
    grid = (B, triton.cdiv(D, block_d))

    _affine_scan_fwd_kernel[grid](
        a,
        b,
        h,
        C,
        D,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        REVERSE=reverse,
        num_warps=num_warps,
    )
    return h


def _launch_affine_backward(grad_h, a, h, reverse: bool):
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not installed.")

    B, C, D = a.shape
    grad_a = torch.empty((B, C, D), device=a.device, dtype=a.dtype)
    grad_b = torch.empty((B, C, D), device=a.device, dtype=a.dtype)

    block_c, block_d, num_warps = _scan_launch_meta(C)
    grid = (B, triton.cdiv(D, block_d))

    _affine_scan_bwd_kernel[grid](
        grad_h,
        a,
        h,
        grad_a,
        grad_b,
        C,
        D,
        grad_h.stride(0),
        grad_h.stride(1),
        grad_h.stride(2),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        REVERSE=reverse,
        num_warps=num_warps,
    )
    return grad_a, grad_b


class _TritonBidirectionalAffineScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_f, b_f, a_b, b_b):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is not installed.")

        if not (a_f.is_cuda and b_f.is_cuda and a_b.is_cuda and b_b.is_cuda):
            raise RuntimeError("Triton bidirectional affine scan requires CUDA tensors.")

        if not (a_f.shape == b_f.shape == a_b.shape == b_b.shape):
            raise ValueError("a_f, b_f, a_b, and b_b must have identical shapes.")

        # Two kernel launches total: one per direction.  Each kernel performs the
        # entire associative scan internally, avoiding Python-level tree stages,
        # torch.stack, torch.flip, and intermediate prefix tensors.
        h_f = _launch_affine_forward(a_f, b_f, reverse=False)
        h_b = _launch_affine_forward(a_b, b_b, reverse=True)

        ctx.save_for_backward(a_f, a_b, h_f, h_b)
        return h_f, h_b

    @staticmethod
    def backward(ctx, grad_h_f, grad_h_b):
        a_f, a_b, h_f, h_b = ctx.saved_tensors

        if grad_h_f is None:
            grad_h_f = torch.zeros_like(h_f)
        if grad_h_b is None:
            grad_h_b = torch.zeros_like(h_b)

        grad_a_f, grad_b_f = _launch_affine_backward(
            grad_h_f, a_f, h_f, reverse=False
        )
        grad_a_b, grad_b_b = _launch_affine_backward(
            grad_h_b, a_b, h_b, reverse=True
        )

        return grad_a_f, grad_b_f, grad_a_b, grad_b_b


def triton_bidirectional_affine_scan(a_f, b_f, a_b, b_b):
    """Training-capable GPU bidirectional affine scan.

    All tensors are [B, C, D].  h_f is aligned with the original variable order;
    h_b is also returned aligned with the original variable order even though its
    recurrence is evaluated from C-1 to 0.
    """
    return _TritonBidirectionalAffineScanFn.apply(a_f, b_f, a_b, b_b)


# ============================================================================
# PyTorch reference scans (correctness / ablation / fallback)
# ============================================================================


def _sequential_affine_scan(a, b):
    states = []
    h = torch.zeros_like(b[:, 0, :])
    for i in range(b.size(1)):
        h = torch.addcmul(b[:, i, :], a[:, i, :], h)
        states.append(h)
    return torch.stack(states, dim=1)


def _hillis_steele_affine_scan(a, b):
    C = a.size(1)
    if C == 1:
        return b

    A = a
    Bv = b
    offset = 1

    while offset < C:
        A_prev = A
        B_prev = Bv

        A_head = A_prev[:, :offset, :]
        B_head = B_prev[:, :offset, :]

        A_right = A_prev[:, offset:, :]
        A_left = A_prev[:, :-offset, :]
        B_right = B_prev[:, offset:, :]
        B_left = B_prev[:, :-offset, :]

        A_tail = A_right * A_left
        B_tail = torch.addcmul(B_right, A_right, B_left)

        A = torch.cat((A_head, A_tail), dim=1)
        Bv = torch.cat((B_head, B_tail), dim=1)
        offset *= 2

    return Bv


def _work_efficient_affine_scan_torch(a, b):
    B, C, D = a.shape
    if C == 1:
        return b

    N = 1 << (C - 1).bit_length()
    num_levels = N.bit_length() - 1
    pad = N - C

    if pad > 0:
        a_pad = F.pad(a, (0, 0, 0, pad), value=1.0)
        b_pad = F.pad(b, (0, 0, 0, pad), value=0.0)
    else:
        a_pad = a
        b_pad = b

    levels_a = [a_pad]
    levels_b = [b_pad]
    cur_a = a_pad
    cur_b = b_pad

    for _ in range(num_levels):
        left_a = cur_a[:, 0::2, :]
        left_b = cur_b[:, 0::2, :]
        right_a = cur_a[:, 1::2, :]
        right_b = cur_b[:, 1::2, :]

        next_a = right_a * left_a
        next_b = torch.addcmul(right_b, right_a, left_b)

        levels_a.append(next_a)
        levels_b.append(next_b)
        cur_a = next_a
        cur_b = next_b

    prefix_a = torch.ones_like(cur_a)
    prefix_b = torch.zeros_like(cur_b)

    for level_idx in range(num_levels, 0, -1):
        child_a = levels_a[level_idx - 1]
        child_b = levels_b[level_idx - 1]

        left_agg_a = child_a[:, 0::2, :]
        left_agg_b = child_b[:, 0::2, :]

        right_prefix_a = left_agg_a * prefix_a
        right_prefix_b = torch.addcmul(left_agg_b, left_agg_a, prefix_b)

        prefix_a = torch.stack((prefix_a, right_prefix_a), dim=2).flatten(1, 2)
        prefix_b = torch.stack((prefix_b, right_prefix_b), dim=2).flatten(1, 2)

    inclusive_b = torch.addcmul(b_pad, a_pad, prefix_b)
    return inclusive_b[:, :C, :]


def _torch_bidirectional_scan(scan_fn, a_f, b_f, a_b, b_b):
    h_f = scan_fn(a_f, b_f)
    h_b = torch.flip(
        scan_fn(torch.flip(a_b, dims=(1,)), torch.flip(b_b, dims=(1,))),
        dims=(1,),
    )
    return h_f, h_b




class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(*normalized_shape))  # 可学习的缩放参数

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=(-2, -1), keepdim=True) + self.eps)
        return self.scale * x / rms


class MLP_block(nn.Module):
    def __init__(
        self,
        norm_dim,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            RMSNorm([norm_dim, input_dim]),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)




class CompactMultiscale_MLP(nn.Module):
    """Compact multi-scale temporal encoder.

    Input:
        x: [B, L, C]
    Output:
        out: [B, D, C]
    """

    def __init__(self, configs):
        super(CompactMultiscale_MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout_p = configs.dropout
        self.core = 64
        self.use_decomp = getattr(configs, "use_decomp", False)
        self.patch = list(getattr(configs, "patch_sizes", [1,4,12,24]))

        if not self.patch:
            raise ValueError("patch_sizes must not be empty")

        invalid = [
            p for p in self.patch
            if p < 1 or self.seq_len % p != 0
        ]
        if invalid:
            raise ValueError(
                f"seq_len={self.seq_len} must be divisible by every "
                f"patch size; invalid={invalid}"
            )

        self.patch_num = [self.seq_len // p for p in self.patch]
        self.scale_dim = max(16, self.d_model // (len(self.patch)+1))

        self.decomp = nn.ModuleList([
            series_decomp(p + 1)
            for p in self.patch
        ])

        self.t_intra = nn.ModuleList()
        self.s_intra = nn.ModuleList()
        self.t_inter_proj = nn.ModuleList()
        self.s_inter_proj = nn.ModuleList()

        for p, n_patch in zip(self.patch, self.patch_num):
            self.t_intra.append(
                MLP_block(
                    norm_dim=n_patch,
                    input_dim=p,
                    hidden_dim=self.core,
                    output_dim=p,
                    dropout=self.dropout_p,
                )
            )
            self.s_intra.append(
                MLP_block(
                    norm_dim=n_patch,
                    input_dim=p,
                    hidden_dim=self.core,
                    output_dim=p,
                    dropout=self.dropout_p,
                )
            )
            self.t_inter_proj.append(
                MLP_block(
                    norm_dim=p,
                    input_dim=n_patch,
                    hidden_dim=self.core,
                    output_dim=self.scale_dim,
                    dropout=self.dropout_p,
                )
            )
            self.s_inter_proj.append(
                MLP_block(
                    norm_dim=p,
                    input_dim=n_patch,
                    hidden_dim=self.core,
                    output_dim=self.scale_dim,
                    dropout=self.dropout_p,
                )
            )

        fusion_dim = (len(self.patch)+1) * self.scale_dim
        self.scale_fusion = MLP_block(
            norm_dim=self.enc_in,
            input_dim=fusion_dim,
            hidden_dim=self.d_model,
            output_dim=self.d_model,
            dropout=self.dropout_p,
        )
        self.base_proj = nn.Linear(self.seq_len, self.d_model)
        self.scale1_proj = MLP_block(
            norm_dim=self.enc_in,
            input_dim=self.seq_len,
            hidden_dim=self.core,
            output_dim=self.scale_dim,
            dropout=self.dropout_p,
        )


    def forward(self, x, x_mark_enc=None):
        B, L, C = x.shape
        if L != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {L}")

        x_var = x.transpose(1, 2)
        base = self.base_proj(x_var)
        scale1 = self.scale1_proj(x_var)
        scale_features = [scale1]

        for i, (p, n_patch) in enumerate(zip(self.patch, self.patch_num)):
            if self.use_decomp:
                season, trend = self.decomp[i](x)
            else:
                trend = x

            trend = trend.transpose(1, 2).reshape(B, C, n_patch, p)
            trend = trend + self.t_intra[i](trend)
            trend = trend.permute(0, 1, 3, 2)
            trend = self.t_inter_proj[i](trend).mean(dim=2)
            scale_feature = trend

            if self.use_decomp:
                season = season.transpose(1, 2).reshape(B, C, n_patch, p)
                season = season + self.s_intra[i](season)
                season = season.permute(0, 1, 3, 2)
                season = self.s_inter_proj[i](season).mean(dim=2)
                scale_feature = scale_feature + season

            scale_features.append(scale_feature)

        multi_scale = torch.cat(scale_features, dim=-1)
        out = base + self.scale_fusion(multi_scale)
        return out.transpose(1, 2).contiguous()




class RelationAwareRecurrentMixer(nn.Module):
    """Relation-aware bidirectional recurrent variable mixer.
    Input:
        x: [B, D, C]
    Output:
        y: [B, D, C]
    Recurrent update:
        h_i = g_i * h_{i-1} + (1 - g_i) * u_i
    Supported scan modes:
        sequential:
            O(C) work, O(C) dependency depth.
        parallel / hillis_steele:
            O(C log C) work, O(log C) dependency depth.
        work_efficient:
            O(C) arithmetic work, O(log C) dependency depth.
    """

    def __init__(self, configs, relation_aware=True):
        super(RelationAwareRecurrentMixer, self).__init__()

        self.d_model = configs.d_model
        self.dropout_p = configs.dropout
        self.relation_aware = relation_aware

        self.scan_mode = str(getattr(configs, "scan_mode", "work_efficient")).lower()
        self.relation_gate = str(getattr(configs, "relation_gate", "additive")).lower()    # dense  additive
        self.context_type = str(getattr(configs, "context_type", "learnable")).lower()   # learnable  mean
        self.use_triton = True
        self.norm = nn.LayerNorm(self.d_model)

        if self.relation_aware:
            if self.context_type == "learnable":
                self.context_phi = nn.Linear(self.d_model,self.d_model)
                self.context_rho = nn.Linear(self.d_model,self.d_model)

            elif self.context_type != "mean":
                raise ValueError('context_type must be either ''"mean" or "learnable"')

            if self.relation_gate == "dense":
                self.gate_dense = nn.Linear(4 * self.d_model,2 * self.d_model)

            elif self.relation_gate == "additive":
                self.gate_e = nn.Linear(self.d_model,2 * self.d_model)
                self.gate_c = nn.Linear(self.d_model,2 * self.d_model)
            else:
                raise ValueError('relation_gate must be either ''"additive" or "dense"')
        else:
            self.gate_plain = nn.Linear(self.d_model,2 * self.d_model)

        self.candidate = nn.Linear(self.d_model, 2 * self.d_model)
        self.out_proj = nn.Linear(2 * self.d_model, self.d_model)

    def _leave_one_out_context(self, e):
        """
        Args:
            e: [B, C, D]
        Returns:
            c: [B, C, D]
        """
        C = e.size(1)
        sum_e = e.sum(dim=1, keepdim=True)
        c_mean = (sum_e - e) / (C - 1)

        if self.context_type == "mean":
            return c_mean

        z = self.context_phi(e)
        sum_z = z.sum(dim=1, keepdim=True)
        c_refine = (sum_z - z) / (C - 1)
        c_refine = self.context_rho(c_refine)

        return c_mean + c_refine


    def _compute_gate_and_candidate(self, e):
        """
        Compute gate and candidate representations for both directions.
        Args:
            e: [B, C, D]
        Returns:
            g_f, g_b, u_f, u_b: [B, C, D]
        """
        B, C, D = e.shape


        if self.relation_aware:
            c = self._leave_one_out_context(e)
            if self.relation_gate == "dense":
                delta = torch.abs(e - c)
                prod = e * c
                relation = torch.cat((e, c, delta, prod),dim=-1)
                gate_logits = self.gate_dense(relation)
                gate_logits = gate_logits.view(B,C,2,D)

            else:
                gate_logits = (self.gate_e(e).view(B, C, 2, D) + self.gate_c(c).view(B, C, 2, D))

                delta = torch.abs(e - c)
                gate_logits = gate_logits + delta.unsqueeze(2)

                prod = e * c
                gate_logits = gate_logits + prod.unsqueeze(2)

        else:
            gate_logits = self.gate_plain(e).view(B, C, 2, D)

        g = torch.sigmoid(gate_logits)
        u = torch.tanh(self.candidate(e)).view(B, C, 2, D)

        g_f = g[:, :, 0, :]
        g_b = g[:, :, 1, :]
        u_f = u[:, :, 0, :]
        u_b = u[:, :, 1, :]

        return g_f, g_b, u_f, u_b


    @staticmethod
    def _sequential_affine_scan(a, b):
        """Reference inclusive scan.
        h_i = a_i * h_{i-1} + b_i
        Args:
            a, b: [B, C, D]
        Returns:
            h: [B, C, D]
        """
        states = []
        h = torch.zeros_like(b[:, 0, :])

        for i in range(b.size(1)):
            h = a[:, i, :] * h + b[:, i, :]
            states.append(h)

        return torch.stack(states, dim=1)

    # ==================================================================
    # Hillis-Steele parallel scan
    # ==================================================================
    @staticmethod
    def _hillis_steele_affine_scan(a, b):
        """
        Hillis-Steele associative scan.

        Complexity:
            O(C log C) arithmetic work
            O(log C) dependency depth
        """
        C = a.size(1)
        if C == 1:
            return b
        A = a
        Bv = b
        offset = 1
        while offset < C:
            A_prev = A
            B_prev = Bv
            A_head = A_prev[:, :offset, :]
            B_head = B_prev[:, :offset, :]
            A_right = A_prev[:, offset:, :]
            A_left = A_prev[:, :-offset, :]
            B_right = B_prev[:, offset:, :]
            B_left = B_prev[:, :-offset, :]
            A_tail = (A_right* A_left)
            B_tail = torch.addcmul(B_right,A_right,B_left)

            A = torch.cat((A_head, A_tail),dim=1)
            Bv = torch.cat((B_head, B_tail),dim=1)
            offset *= 2
        return Bv

    # ==================================================================
    # Work-efficient affine scan
    # ==================================================================
    @staticmethod
    def _work_efficient_affine_scan(a, b):
        """
        The implementation uses
            1. tree reduction (up-sweep),
            2. exclusive-prefix propagation (down-sweep),
            3. exclusive-to-inclusive conversion.

        Complexity:
            O(C) arithmetic work
            O(log C) dependency depth
        """
        B, C, D = a.shape
        N = 1 << (C - 1).bit_length()
        num_levels = (N.bit_length() - 1)
        pad = N - C

        if pad > 0:
            a_pad = torch.nn.functional.pad(a, (0, 0,0, pad), mode="constant", value=1.0)
            b_pad = torch.nn.functional.pad(b, (0, 0,0, pad), mode="constant", value=0.0)
        else:
            a_pad = a
            b_pad = b

        levels_a = [a_pad]
        levels_b = [b_pad]

        cur_a = a_pad
        cur_b = b_pad

        for _ in range(num_levels):
            left_a = cur_a[:, 0::2, :]
            left_b = cur_b[:, 0::2, :]
            right_a = cur_a[:, 1::2, :]
            right_b = cur_b[:, 1::2, :]

            next_a = (right_a * left_a)
            next_b = torch.addcmul(right_b, right_a, left_b)

            levels_a.append(next_a)
            levels_b.append(next_b)

            cur_a = next_a
            cur_b = next_b


        prefix_a = torch.ones_like(cur_a)
        prefix_b = torch.zeros_like(cur_b)

        for level_idx in range(num_levels, 0, -1):
            child_a = levels_a[level_idx - 1]
            child_b = levels_b[level_idx - 1]

            left_agg_a = child_a[:, 0::2, :]
            left_agg_b = child_b[:, 0::2, :]

            left_prefix_a = prefix_a
            left_prefix_b = prefix_b

            right_prefix_a = (left_agg_a* prefix_a)
            right_prefix_b = torch.addcmul(left_agg_b, left_agg_a, prefix_b)

            prefix_a = torch.stack((left_prefix_a, right_prefix_a), dim=2).flatten(1,2)
            prefix_b = torch.stack((left_prefix_b, right_prefix_b), dim=2).flatten(1,2)


        inclusive_b = torch.addcmul(b_pad,a_pad,prefix_b)

        return inclusive_b[:, :C, :]

    # ==================================================================
    # Scan dispatcher
    # ==================================================================
    def _scan(self, a, b):
        if self.scan_mode == "sequential":
            return (self._sequential_affine_scan(a,b))

        if self.scan_mode in ["parallel","hillis_steele","hillis-steele"]:
            return (self._hillis_steele_affine_scan(a,b))

        raise ValueError(
            "scan_mode must be one of: "
            '"sequential", "parallel", or "work_efficient"'
        )


    def _bidirectional_scan(self,a_f,b_f,a_b,b_b):
        if self.scan_mode in ["work_efficient","work-efficient"]:
            if self.use_triton:
                return triton_bidirectional_affine_scan(a_f, b_f, a_b, b_b)
            else:
                B = a_f.size(0)

                a_b_rev = torch.flip(a_b,dims=(1,))
                b_b_rev = torch.flip(b_b,dims=(1,))

                a_all = torch.cat((a_f, a_b_rev),dim=0)
                b_all = torch.cat((b_f, b_b_rev),dim=0)

                h_all = self._work_efficient_affine_scan(a_all,b_all)

                h_f = h_all[:B]
                h_b = torch.flip(h_all[B:],dims=(1,))
                return h_f, h_b

        else:
            B = a_f.size(0)

            a_b_rev = torch.flip(a_b,dims=(1,))
            b_b_rev = torch.flip(b_b,dims=(1,))

            a_all = torch.cat((a_f, a_b_rev),dim=0)
            b_all = torch.cat((b_f, b_b_rev),dim=0)

            h_all = self._scan(a_all,b_all)

            h_f = h_all[:B]
            h_b = torch.flip(h_all[B:],dims=(1,))
            return h_f, h_b



    def forward(self, x):
        """
        Args:
            x: [B, D, C]
        Returns:
            out: [B, D, C]
        """

        e = x.transpose(1,2)
        e_norm = self.norm(e)

        (g_f,g_b,u_f,u_b) = self._compute_gate_and_candidate(e_norm)

        a_f = g_f
        a_b = g_b

        b_f = torch.addcmul(u_f, g_f, u_f, value=-1.0)
        b_b = torch.addcmul(u_b, g_b, u_b, value=-1.0)

        h_f, h_b = self._bidirectional_scan(a_f,b_f,a_b,b_b)

        h = torch.cat((h_f,h_b),dim=-1)
        z = self.out_proj(h)

        out = e + z
        return out.transpose(1,2)




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.use_norm = configs.use_norm
        self.freq = configs.freq
        self.d_model = configs.d_model


        self.compact_multiscale_mlp = CompactMultiscale_MLP(configs)
        mixer_name = str(getattr(configs, "channel_function", "relationscan")).lower()

        if mixer_name in ['plainscan']:
            self.model = nn.ModuleList([
                RelationAwareRecurrentMixer(configs, relation_aware=False)
                for _ in range(configs.e_layers)
            ])
        elif mixer_name in ['relationscan']:
            self.model = nn.ModuleList([
                RelationAwareRecurrentMixer(configs, relation_aware=True)
                for _ in range(configs.e_layers)
            ])

        self.projection = nn.Linear(configs.d_model, configs.pred_len)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = self.compact_multiscale_mlp(x_enc, x_mark_enc)
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection((x_enc).transpose(1, 2)).transpose(1, 2)
    
        if self.use_norm:
            enc_out = enc_out  * stdev + means

        return enc_out[:,:,:self.enc_in]



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out  = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]   # [B, L, D]