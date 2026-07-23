import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import index_points, square_distance
from typing import Optional

# =========================================================
# nD-RoPE utils (3D point cloud)
#   - simplex wave-vector directions (M=d+1)
#   - per-head random rotation
#   - build complex cis from RELATIVE positions (B,N,K,3)
# =========================================================

def generate_simplex_vectors_with_projection(dimension: int, normalize_rows: bool = True):
    """
    Return (M, d) simplex directions, where M = d+1 for d>=2, and M=1 for d=1.
    """
    if dimension == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)

    points = torch.eye(dimension + 1, dtype=torch.float32)  # (d+1, d+1)
    points -= points.mean(dim=0, keepdim=True)
    U, _, _ = torch.linalg.svd(points.T, full_matrices=False)
    reduced = points @ U[:, :-1]  # (d+1, d)

    if normalize_rows:
        reduced = reduced / (reduced.norm(dim=1, keepdim=True) + 1e-12)
    return reduced  # (M=d+1, d)


def init_nd_freqs(reduced_vectors: torch.Tensor, num_heads: int, rotate: bool = True,
                  device=None, dtype=torch.float32):
    """
    Return freqs: (H, M, d) per-head rotated simplex directions.
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


def compute_ndrope_cis_3d_rel(
    freqs: torch.Tensor,          # (H,M,3)
    rel_pos: torch.Tensor,        # (B,N,K,3)  relative position
    head_dim: int,
    theta: float = 100.0,
    phase_offset: Optional[torch.Tensor] = None,  # (1,H,1,1,D/2)
):
    """
    Return freqs_cis: complex (B,H,N,K,D/2)
    We build M directions, and S scales so that head_dim = 2 * M * S.
    """
    device, dtype = rel_pos.device, rel_pos.dtype
    B, N, K, d = rel_pos.shape
    H, M, d2 = freqs.shape
    assert d == d2 == 3

    dim_per_scale = 2 * M
    if head_dim % dim_per_scale != 0:
        raise ValueError(f"head_dim must be divisible by 2*M. got head_dim={head_dim}, 2*M={2*M}")
    S = head_dim // dim_per_scale  # number of scales

    # scales (low->high)
    mag = 1.0 / (theta ** (torch.arange(S, device=device, dtype=dtype) / max(S, 1)))

    # proj: (B,H,N,K,M)
    proj = torch.einsum("bnkd,hmd->bhnkm", rel_pos.to(dtype), freqs.to(device=device, dtype=dtype))
    # angles: (B,H,N,K,M,S) -> (B,H,N,K,M*S=D/2)
    angles = proj.unsqueeze(-1) * mag.view(1, 1, 1, 1, 1, S)
    angles = angles.reshape(B, H, N, K, M * S)  # (B,H,N,K,D/2)

    if phase_offset is not None:
        angles = angles + phase_offset.to(device=device, dtype=dtype)  # broadcast

    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex
    return freqs_cis


def apply_rotary_emb_relation(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    x:        (B,H,N,K,D) real
    freqs_cis:(B,H,N,K,D/2) complex
    """
    B, H, N, K, D = x.shape
    assert D % 2 == 0
    D2 = D // 2
    x_c = torch.view_as_complex(x.float().reshape(B, H, N, K, D2, 2))
    x_rot = x_c * freqs_cis
    out = torch.view_as_real(x_rot).reshape(B, H, N, K, D).type_as(x)
    return out


# =========================================================
# PointTransformer (v1-style) Vector Attention + nD-RoPE
#   - keep vector-attention form:
#       attn_ij = softmax( MLP( (q_i - k_j) + phi(delta_ij) ) )   (vector weights)
#       out_i   = sum_j attn_ij ⊙ (v_j + phi(delta_ij))
#   - add nD-RoPE to modulate the relation (q_i - k_j) using RELATIVE position delta_ij
# =========================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_points: int,
        d_model: int,
        k: int,
        num_heads: int = 4,
        rope_theta: float = 100.0,
        rope_rotate_freqs: bool = True,
        use_phase_offset: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_dim = d_model // num_heads
        assert head_dim % 2 == 0, "head_dim must be even for rotary"

        self.k = k
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        # same as PT-v1 block
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        # nD-RoPE freqs (H, M, 3), M = 4 for 3D
        reduced = generate_simplex_vectors_with_projection(3, normalize_rows=True)  # (4,3)
        freqs = init_nd_freqs(reduced, num_heads=num_heads, rotate=rope_rotate_freqs)
        self.register_buffer("freqs", freqs, persistent=True)

        # optional learnable phase offset (1,H,1,1,D/2)
        if use_phase_offset:
            self.phase_offset = nn.Parameter(torch.zeros(1, num_heads, 1, 1, head_dim // 2))
        else:
            self.phase_offset = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        xyz:      (B,N,3)
        features: (B,N,d_points)
        returns:
          out:  (B,N,d_points)
          attn: (B,N,K,d_model)  (vector weights like original PT-v1)
        """
        B, N, _ = xyz.shape

        # ---- KNN ----
        dists = square_distance(xyz, xyz)                     # (B,N,N)
        K = min(self.k, N)
        knn_idx = dists.argsort()[:, :, :K]                  # (B,N,K)
        knn_xyz = index_points(xyz, knn_idx)                 # (B,N,K,3)
        rel_pos = xyz[:, :, None, :] - knn_xyz               # (B,N,K,3)

        pre = features
        x = self.fc1(features)                               # (B,N,d_model)

        q = self.w_qs(x)                                     # (B,N,d_model)
        k = index_points(self.w_ks(x), knn_idx)              # (B,N,K,d_model)
        v = index_points(self.w_vs(x), knn_idx)              # (B,N,K,d_model)

        # ---- relative position encoding (same as PT-v1) ----
        pos_enc = self.fc_delta(rel_pos)                     # (B,N,K,d_model)

        # ---- nD-RoPE: modulate relation (q - k) using RELATIVE position ----
        # relation: (B,N,K,d_model) -> (B,H,N,K,D)
        relation = (q[:, :, None, :] - k)
        rel_h = relation.view(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()

        freqs_cis = compute_ndrope_cis_3d_rel(
            self.freqs, rel_pos, head_dim=self.head_dim, theta=self.rope_theta,
            phase_offset=self.phase_offset
        )  # (B,H,N,K,D/2) complex

        rel_h = apply_rotary_emb_relation(rel_h, freqs_cis)  # (B,H,N,K,D)
        relation_rope = rel_h.permute(0, 2, 3, 1, 4).reshape(B, N, K, self.d_model).contiguous()

        # ---- vector attention weights (same form as PT-v1) ----
        attn = self.fc_gamma(relation_rope + pos_enc)        # (B,N,K,d_model)
        attn = F.softmax(attn / math.sqrt(self.d_model), dim=-2)  # softmax over neighbors K

        # ---- aggregate (same as PT-v1) ----
        res = torch.einsum("bnkf,bnkf->bnf", attn, v + pos_enc)    # (B,N,d_model)
        out = self.fc2(res) + pre                                  # (B,N,d_points)
        return out, attn