import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet_util import index_points, square_distance


# =========================================================
# nD-RoPE utils (3D point cloud)
# =========================================================

def generate_simplex_vectors_with_projection(dimension: int, normalize_rows: bool = True):
    """
    Return (M, d) simplex directions, where M = d+1 for d>=2, and M=1 for d=1.
    """
    if dimension == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)

    points = torch.eye(dimension + 1, dtype=torch.float32)  # (d+1, d+1)
    points -= points.mean(dim=0, keepdim=True)
    # project to d-dim subspace (hyperplane sum=0)
    U, _, _ = torch.linalg.svd(points.T, full_matrices=False)
    reduced = points @ U[:, :-1]  # (d+1, d)

    if normalize_rows:
        reduced = reduced / (reduced.norm(dim=1, keepdim=True) + 1e-12)
    return reduced  # (M=d+1, d)


def init_nd_freqs(reduced_vectors: torch.Tensor, num_heads: int, rotate: bool = True,
                  device=None, dtype=torch.float32):
    """
    Return freqs: (H, M, d)  per-head rotated simplex directions.
    """
    reduced_vectors = reduced_vectors.to(device=device, dtype=dtype)
    M, d = reduced_vectors.shape
    freqs_all = []
    for _ in range(num_heads):
        if rotate:
            Q, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype))
            if torch.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            freqs_all.append(reduced_vectors @ Q.T)  # (M,d)
        else:
            freqs_all.append(reduced_vectors)
    return torch.stack(freqs_all, dim=0)  # (H,M,d)


def compute_ndrope_cis_3d(
    freqs,
    position,
    head_dim,
    theta=100.0,
    phase_offset=None,
):
    """
    Return freqs_cis: complex, shape (B, H, N, D/2)
    We build M directions, and S scales so that D = 2 * M * S.
    """
    device, dtype = position.device, position.dtype
    B, N, d = position.shape
    H, M, d2 = freqs.shape
    assert d == d2 == 3

    dim_per_scale = 2 * M
    assert head_dim % dim_per_scale == 0, f"head_dim must be divisible by 2*M. got {head_dim} vs {2*M}"
    S = head_dim // dim_per_scale  # number of scales
    # scales (low->high)
    mag = 1.0 / (theta ** (torch.arange(S, device=device, dtype=dtype) / max(S, 1)))

    # proj: (B,H,N,M)
    proj = torch.einsum("bnd,hmd->bhnm", position.to(dtype), freqs.to(device=device, dtype=dtype))
    angles = proj.unsqueeze(-1) * mag.view(1, 1, 1, 1, S)   # (B,H,N,M,S)
    angles = angles.reshape(B, H, N, M * S)                 # (B,H,N,D/2)

    if phase_offset is not None:
        # phase_offset should broadcast to (B,H,N,D/2)
        angles = angles + phase_offset.to(device=device, dtype=dtype)

    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex
    return freqs_cis


def apply_rotary_emb_single(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    x: (B,H,N,D) real
    freqs_cis: (B,H,N,D/2) complex
    """
    B, H, N, D = x.shape
    assert D % 2 == 0
    D2 = D // 2
    x_c = torch.view_as_complex(x.float().reshape(B, H, N, D2, 2))
    x_rot = x_c * freqs_cis
    out = torch.view_as_real(x_rot).reshape(B, H, N, D).type_as(x)
    return out


# =========================================================
# Point Transformer × nD-RoPE × dot-product attention
# =========================================================

class TransformerBlock(nn.Module):
    """
    Drop-in replacement that keeps:
      - local KNN support
      - dot-product scalar attention (正宗)
      - nD-RoPE on Q/K
      - PointTransformer-style relative position injection into V (v + pos_enc)
      - (IMPORTANT) relative position bias added to logits (helps SOTA)
    """
    def __init__(self, d_points: int, d_model: int, k: int,
                 num_heads: int = 4, rope_theta: float = 100.0,
                 use_rel_pos_bias: bool = False, use_phase_offset: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.k = k
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_phase_offset = use_phase_offset

        # feature in/out
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        # qkv
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # relative position -> value injection (v + pos_enc)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        # relative position -> logits bias (scalar per head)
        if self.use_rel_pos_bias:
            self.fc_bias = nn.Sequential(
                nn.Linear(3, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, num_heads)   # output per-head scalar bias
            )

        # ndRoPE freqs buffer (H, M, 3)
        reduced = generate_simplex_vectors_with_projection(3, normalize_rows=True)
        freqs = init_nd_freqs(reduced, num_heads=num_heads, rotate=True)
        self.register_buffer("freqs", freqs, persistent=True)

        # optional learnable phase offset (small) (1,H,1,D/2)
        if self.use_phase_offset:
            self.phase_offset = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim // 2))
        else:
            self.phase_offset = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        xyz:      (B, N, 3)
        features: (B, N, d_points)
        return:   (B, N, d_points), attn (B,H,N,K)
        """
        B, N, _ = xyz.shape

        # ---- KNN ----
        # dists: (B,N,N) ; if you need more memory safety, replace with chunked knn.
        dists = square_distance(xyz, xyz)  # (B,N,N)
        K = min(self.k, N)
        _, knn_idx = torch.topk(dists, k=K, dim=-1, largest=False)  # (B,N,K)
        knn_xyz = index_points(xyz, knn_idx)                        # (B,N,K,3)
        rel = xyz[:, :, None, :] - knn_xyz                          # (B,N,K,3)

        # ---- project ----
        pre = features
        x = self.fc1(features)                                      # (B,N,d_model)

        q = self.w_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)
        k = self.w_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)
        v = self.w_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)

        # ---- nD-RoPE on Q/K (absolute -> dot-product gives relative phase) ----
        freqs_cis = compute_ndrope_cis_3d(
            self.freqs, xyz*50, head_dim=self.head_dim, theta=self.rope_theta,
            phase_offset=self.phase_offset if self.use_phase_offset else None
        )  # (B,H,N,D/2) complex

        q = apply_rotary_emb_single(q, freqs_cis)
        k = apply_rotary_emb_single(k, freqs_cis)

        # ---- gather neighbors for K/V ----
        # reshape for index_points: (B,N,H*D)
        k_flat = k.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        v_flat = v.permute(0, 2, 1, 3).reshape(B, N, self.d_model)

        k_nb = index_points(k_flat, knn_idx).reshape(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B,H,N,K,D)
        v_nb = index_points(v_flat, knn_idx).reshape(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B,H,N,K,D)

        # ---- relative position -> V injection ----
        pos_enc = self.fc_delta(rel)  # (B,N,K,d_model)
        pos_enc = pos_enc.reshape(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B,H,N,K,D)
        v_eff = v_nb + pos_enc

        # ---- dot-product logits ----
        # logits: (B,H,N,K)
        logits = torch.einsum("bhnd,bhnkd->bhnk", q, k_nb) * self.scale

        # ---- optional: relative position bias on logits ----
        if self.use_rel_pos_bias:
            bias = self.fc_bias(rel)                 # (B,N,K,H)
            bias = bias.permute(0, 3, 1, 2)         # (B,H,N,K)
            logits = logits + bias

        attn = F.softmax(logits, dim=-1)            # (B,H,N,K)

        # ---- aggregate ----
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, v_eff)          # (B,H,N,D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)    # (B,N,d_model)

        out = self.fc2(out) + pre                                   # (B,N,d_points)
        return out, attn