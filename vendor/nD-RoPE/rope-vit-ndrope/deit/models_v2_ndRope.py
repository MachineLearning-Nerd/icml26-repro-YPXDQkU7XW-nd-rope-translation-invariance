import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from models_v2 import vit_models, Layer_scale_init_Block, Attention

def set_seed(seed: int = 42):
    import torch, random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def init_t_xy(end_x: int, end_y: int):
    """
    Generate 2D grid coordinates for patch positions.
    Return shape: (end_x * end_y, 2), each row = [x, y]
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    t_xy = torch.stack([t_x, t_y], dim=-1)  # (N, 2)
    return t_xy

def generate_simplex_vectors_with_projection(dimension, normalize_rows=True):
    """
    Generate n wave vectors (omega) in an n-dimensional space using a regular simplex projection.
    """
    if dimension == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)  # 1D case

    points = torch.eye(dimension + 1, dtype=torch.float32)
    points -= points.mean(dim=0)
    
    U, _, _ = torch.linalg.svd(points.T, full_matrices=False)
    
    reduced_vectors = points @ U[:, :-1]        # Ω^(n) = Ω^(n+1) U^(n)
    
    if normalize_rows:
        reduced_vectors = reduced_vectors / (
            reduced_vectors.norm(dim=1, keepdim=True) + 1e-12
        )

    return reduced_vectors

def init_nd_freqs(
    reduced_vectors: torch.Tensor,     # (M, d)
    num_heads: int,
    rotate: bool = True,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    reduced_vectors = reduced_vectors.to(device=device, dtype=dtype)
    M, d = reduced_vectors.shape
    freqs_all = []
    for _ in range(num_heads):
        if rotate:
            # Step 1: random orthogonal rotation in d-dim
            Q, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype))
            if torch.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            # Step 2: rotate simplex basis (M, d)
            rotated = reduced_vectors @ Q.T
        else:
            rotated = reduced_vectors  # (M, d)

        freqs_all.append(rotated)

    # (H, M, d)
    freqs = torch.stack(freqs_all, dim=0)

    return freqs

def compute_ndrope_cis(
    freqs: torch.Tensor,        # (H, L, M, d)
    position: torch.Tensor,     # (N, d)
    emb_dim: int,               # per-head embedding dim (not total across heads)
    num_heads: int,
    theta: float = 100.0,
    device=None,
    dtype=torch.float32,
):
    """
    Compute N-D RoPE complex phases (cis) using simplex directions with optional rotation,
    expanded over S scales per direction.

    Shapes
    ------
    freqs   : (H, L, M, d)   # head, layer(depth), num_vectors, space_dim
    position: (N, d)         # tokens/patches, space_dim
    emb_dim : int            # per-head embedding dim used by RoPE
    num_heads: int           # H

    Returns
    -------
    freqs_cis : complex tensor with shape (L, H, N, M*S)
        - L = num_layers (depth)
        - H = num_heads
        - N = tokens / patches
        - M = num_vectors_per_scale
        - S = emb_dim // (2*M)
      Each entry is exp(i * angle), ready to be broadcast and used in apply_rotary_emb.
    """
    # Move & type
    freqs = freqs.to(device=device, dtype=dtype)
    position = position.to(device=device, dtype=dtype)

    H, L, M, d = freqs.shape
    assert H == num_heads, f"num_heads mismatch: freqs has H={H}, but num_heads={num_heads}"
    assert position.shape[1] == d, f"Dim mismatch: position is (N,{position.shape[1]}), freqs d={d}"

    # Scales per direction so that per-head dims = 2 * M * S (cos/sin pairs)
    dim_per_scale = 2 * M
    assert emb_dim % dim_per_scale == 0, \
        f"emb_dim must be divisible by 2*M. Got emb_dim={emb_dim}, 2*M={2*M}"
    S = emb_dim // dim_per_scale
    # Build multi-scale magnitudes (low→high freq)
    mag = 1.0 / (theta ** (torch.arange(S, device=device, dtype=dtype) / max(S, 1)))

    # Projections: ⟨x, ω⟩ for every (layer, head, token, vector)
    # position: (N, d), freqs: (H, L, M, d)
    # Want: (L, H, N, M)
    proj = torch.einsum('nd,hlmd->lhnm', position, freqs)

    # Expand across S scales: (L,H,N,M,1) * (1,1,1,1,S) -> (L,H,N,M,S)
    angles = proj.unsqueeze(-1) * mag.view(1, 1, 1, 1, S)

    # Flatten directions×scales to match “frequency slots” per head: (L,H,N,M*S)
    angles_flat = angles.reshape(L, H, position.shape[0], M * S)

    # Convert to cis = exp(i * angle); torch.polar(r, theta) with r=1
    freqs_cis = torch.polar(torch.ones_like(angles_flat), angles_flat)  # complex dtype

    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor,
                     xk: torch.Tensor,
                     freqs_cis: torch.Tensor):

    B, H, N, D = xq.shape
    if D % 2 != 0:
        raise ValueError(f"Head dimension must be even, got D={D}.")
    D2 = D // 2

    xq_c = torch.view_as_complex(xq.float().reshape(B, H, N, D2, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(B, H, N, D2, 2))

    # reshape freqs_cis to (1, H, N, D2) for broadcasting
    if not torch.is_complex(freqs_cis):
        freqs_cis = torch.polar(torch.ones_like(freqs_cis),
                                freqs_cis.to(dtype=xq.float().dtype))
    if freqs_cis.ndim == 3:   # (H, N, D2)
        freqs_cis = freqs_cis.unsqueeze(0)
    elif freqs_cis.ndim == 2:  # (N, D2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    elif freqs_cis.ndim == 1:  # (D2,)
        freqs_cis = freqs_cis.view(1, 1, 1, D2)
    else:
        raise ValueError(f"Unsupported freqs_cis shape: {freqs_cis.shape}")

    xq_rot = xq_c * freqs_cis
    xk_rot = xk_c * freqs_cis

    xq_out = torch.view_as_real(xq_rot).reshape(B, H, N, D).type_as(xq)
    xk_out = torch.view_as_real(xk_rot).reshape(B, H, N, D).type_as(xk)

    return xq_out, xk_out
    
    
class ndRoPEAttention(Attention):
    def forward(self,x,freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, D)
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class ndRoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = ndRoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class ndrope_vit_models(vit_models):
    def __init__(self, dimension=2, rope_theta=10,**kwargs):
        super().__init__(**kwargs)

        if hasattr(self, "pos_embed") and isinstance(self.pos_embed, nn.Parameter):
            self.register_parameter("pos_embed", None)
            
        img_size = kwargs['img_size'] if 'img_size' in kwargs else 224
        patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 6
        embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 768
        mlp_ratio = kwargs['mlp_ratio'] if 'mlp_ratio' in kwargs else 4.

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.num_heads = num_heads
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        
        base_pos = init_t_xy(img_size // patch_size, img_size // patch_size)  # (N, 2)
        self.register_buffer("base_positions", base_pos, persistent=False)
        
        self.head_dim = embed_dim // num_heads 
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        set_seed(42)
        freqs_per_layer = []
        reduced_vectors = generate_simplex_vectors_with_projection(dimension=dimension, normalize_rows=True)
        for i, _ in enumerate(self.blocks):
            freqs_per_layer.append(init_nd_freqs(reduced_vectors, self.num_heads)) # [layer, Head, M, d]
        freqs = torch.stack(freqs_per_layer, dim=1) # [Head, layer, M, d]
        
        self.register_buffer("freqs", freqs, persistent=True)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def _get_positions(self, H_patches: int, W_patches: int):
        if (self.base_positions.shape[0] == H_patches * W_patches):
            return self.base_positions

        pos = init_t_xy(H_patches, W_patches).to(device=self.base_positions.device,
                                                dtype=self.base_positions.dtype)
        return pos
    
    def forward_features(self, x):
        B, C, H, W = x.shape # [batch, channel, height, width]
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        pos = self._get_positions(H_p, W_p)
        freqs_cis = compute_ndrope_cis(self.freqs, pos, self.head_dim, self.num_heads, self.rope_theta,device=x.device, dtype=x.dtype) # [layer, Head, N, M*S]

        for i, blk in enumerate(self.blocks):
            x = blk(x, freqs_cis=freqs_cis[i])
        
        x = self.norm(x)
        x = x[:, 0]
        return x
    
@register_model
def ndrope_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = ndrope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=396, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=ndRoPE_Layer_scale_init_Block, Attention_block=ndRoPEAttention,
        rope_theta=100.0,**kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ndrope_deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = ndrope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=ndRoPE_Layer_scale_init_Block, Attention_block=ndRoPEAttention,
        rope_theta=100.0, **kwargs)
    return model

@register_model
def ndrope_deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = ndrope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=ndRoPE_Layer_scale_init_Block, Attention_block=ndRoPEAttention,
        rope_theta=100.0, **kwargs)
    return model