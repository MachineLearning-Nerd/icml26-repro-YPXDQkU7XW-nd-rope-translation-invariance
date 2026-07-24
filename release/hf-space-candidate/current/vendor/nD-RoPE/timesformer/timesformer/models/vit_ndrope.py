# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

# ==== ndRoPE utils (ADD) ====
def set_seed(seed: int = 42):
    import torch, random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return torch.stack([t_x, t_y], dim=-1)  # (N,2)

def generate_simplex_vectors_with_projection(dimension, normalize_rows=True):
    if dimension == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    points = torch.eye(dimension + 1, dtype=torch.float32)
    points -= points.mean(dim=0)
    U, _, _ = torch.linalg.svd(points.T, full_matrices=False)
    reduced_vectors = points @ U[:, :-1]
    if normalize_rows:
        reduced_vectors = reduced_vectors / (reduced_vectors.norm(dim=1, keepdim=True) + 1e-12)
    return reduced_vectors  # (M,d)

def init_nd_freqs(reduced_vectors: torch.Tensor, num_heads: int, rotate: bool = True,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    reduced_vectors = reduced_vectors.cpu().float()
    M, d = reduced_vectors.shape
    freqs_all = []
    
    g = torch.Generator()
    g.manual_seed(42) 

    for _ in range(num_heads):
        if rotate and d > 0:
            rand_mat = torch.randn(d, d, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(rand_mat)
            if torch.linalg.det(Q) < 0: Q[:, 0] = -Q[:, 0]
            rotated = reduced_vectors @ Q.T
        else:
            rotated = reduced_vectors
        freqs_all.append(rotated)
    
    return torch.stack(freqs_all, dim=0).to(device=device, dtype=dtype)

def compute_ndrope_cis(freqs: torch.Tensor, position: torch.Tensor, emb_dim: int, num_heads: int,
                       theta: float = 100.0, device=None, dtype=torch.float32):
    # freqs: (H, L, M, d) or (H, 1, M, d); position: (N,d)
    freqs = freqs.to(device=device, dtype=dtype)
    position = position.to(device=device, dtype=dtype)
    H, L, M, d = freqs.shape
    assert position.shape[1] == d, f"Pos dim={position.shape[1]} != {d}"

    dim_per_scale = 2 * M
    
    S = emb_dim // dim_per_scale 

    mag = 1.0 / (theta ** (torch.arange(S, device=device, dtype=dtype) / max(S, 1)))

    # (L,H,N,M)
    proj = torch.einsum('nd,hlmd->lhnm', position, freqs)
    angles = proj.unsqueeze(-1) * mag.view(1,1,1,1,S)
    angles_flat = angles.reshape(L, H, position.shape[0], M * S)  # (L,H,N,MS)
    freqs_cis = torch.polar(torch.ones_like(angles_flat), angles_flat)  # complex
    return freqs_cis  # (L,H,N, D_rot/2)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    B, H, N, D = xq.shape
    assert D % 2 == 0, f"Head dim must be even, got {D}"
    D2 = D // 2
    
    xq_c = torch.view_as_complex(xq.float().reshape(B, H, N, D2, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(B, H, N, D2, 2))
    
    if not torch.is_complex(freqs_cis):
        freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis.to(dtype=xq.float().dtype))
    if freqs_cis.ndim == 3:   # (H,N,D_rot)
        freqs_cis = freqs_cis.unsqueeze(0)
    elif freqs_cis.ndim == 2: # (N,D_rot)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    elif freqs_cis.ndim == 1: # (D_rot,)
        freqs_cis = freqs_cis.view(1,1,1,-1)
        
    rope_dim = freqs_cis.shape[-1]
    
    if rope_dim == D2:
        xq_rot = xq_c * freqs_cis
        xk_rot = xk_c * freqs_cis
        xq_out = torch.view_as_real(xq_rot).reshape(B, H, N, D).type_as(xq)
        xk_out = torch.view_as_real(xk_rot).reshape(B, H, N, D).type_as(xk)
    else:
        xq_rot_part, xq_pass_part = xq_c[..., :rope_dim], xq_c[..., rope_dim:]
        xk_rot_part, xk_pass_part = xk_c[..., :rope_dim], xk_c[..., rope_dim:]
        
        xq_rot_part = xq_rot_part * freqs_cis
        xk_rot_part = xk_rot_part * freqs_cis
        
        xq_out_c = torch.cat([xq_rot_part, xq_pass_part], dim=-1)
        xk_out_c = torch.cat([xk_rot_part, xk_pass_part], dim=-1)
        
        xq_out = torch.view_as_real(xq_out_c).reshape(B, H, N, D).type_as(xq)
        xk_out = torch.view_as_real(xk_out_c).reshape(B, H, N, D).type_as(xk)
    
    return xq_out, xk_out
# ==== ndRoPE utils (END) ====

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        if freqs_cis is not None:
            L = q.size(2)                   
            F = freqs_cis.size(-2)       
            if F == L:
                q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            elif F == L - 1:
                q_cls, k_cls = q[:, :, :1], k[:, :, :1]
                q_tok, k_tok = q[:, :, 1:], k[:, :, 1:]
                q_tok, k_tok = apply_rotary_emb(q_tok, k_tok, freqs_cis=freqs_cis)
                q = torch.cat([q_cls, q_tok], dim=2)
                k = torch.cat([k_cls, k_tok], dim=2)
            else:
                raise RuntimeError(f"RoPE length mismatch: seq_len={L}, freqs_len={F}")
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W, freqs_spatial=None, freqs_temporal=None):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt),freqs_cis=freqs_temporal))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs),freqs_cis=freqs_spatial))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.,
                 # ==== ndRoPE args (ADD) ====
                 use_ndrope=True, theta_spatial=100, theta_temporal=10000, dim_spatial=2, dim_temporal=1, rotate=True):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        self.use_ndrope = use_ndrope
        if self.use_ndrope and hasattr(self, "pos_embed"):
            self.register_parameter("pos_embed", None)
        if self.use_ndrope and hasattr(self, "time_embed"):
            self.register_parameter("time_embed", None)
            
        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if getattr(self, "pos_embed", None) is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

        # ==== ndRoPE init (ADD) ====
        self.use_ndrope = use_ndrope
        self.theta_spatial = theta_spatial
        self.theta_temporal = theta_temporal
        self.dim_spatial = dim_spatial
        self.dim_temporal = dim_temporal
        self.rotate = rotate
        head_dim = embed_dim // num_heads
        self._head_dim = head_dim
        if self.use_ndrope:
            # set_seed(42)
            red_vec_spa = generate_simplex_vectors_with_projection(dim_spatial, True)   # M_s x 2
            red_vec_tmp = generate_simplex_vectors_with_projection(dim_temporal, True)  # M_t x 1
            freqs_spa, freqs_tmp = [], []
            for _ in range(self.depth):
                freqs_spa.append(init_nd_freqs(red_vec_spa, num_heads, rotate=rotate, dtype=torch.float32))
                freqs_tmp.append(init_nd_freqs(red_vec_tmp, num_heads, rotate=False, dtype=torch.float32))
            self.register_buffer("freqs_spatial",  torch.stack(freqs_spa, dim=1),  persistent=True)
            self.register_buffer("freqs_temporal", torch.stack(freqs_tmp, dim=1),  persistent=True)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)                      # x: ((B*T), N, C)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)              # ((B*T), 1+N, C)

        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]                                       # ((B*T), N, C)
        x = rearrange(x, '(b t) n m -> b (n t) m', b=B, t=T)  # (B, N*T, C)
        x = torch.cat((cls_tokens, x), dim=1)              # (B, 1+N*T, C)

        num_spatial_tokens = (x.size(1) - 1) // T
        H_p = num_spatial_tokens // W
        W_p = W

        if self.use_ndrope:
            pos_xy = init_t_xy(W_p, H_p).to(device=x.device, dtype=x.dtype)  # (H_p*W_p, 2)
            pos_t  = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(-1)  # (T,1)
            freqs_cis_spatial  = compute_ndrope_cis(self.freqs_spatial,  pos_xy, self._head_dim, self.blocks[0].attn.num_heads, self.theta_spatial, device=x.device, dtype=x.dtype)
            freqs_cis_temporal = compute_ndrope_cis(self.freqs_temporal, pos_t,  self._head_dim, self.blocks[0].attn.num_heads, self.theta_temporal, device=x.device, dtype=x.dtype)
        else:
            freqs_cis_spatial = freqs_cis_temporal = None

        # Blocks
        for i, blk in enumerate(self.blocks):
            fspa_i = freqs_cis_spatial[i]  if self.use_ndrope else None  # (H, H_p*W_p, D/2)
            ftmp_i = freqs_cis_temporal[i] if self.use_ndrope else None  # (H, T,       D/2)
            x = blk(x, B, T, W, freqs_spatial=fspa_i, freqs_temporal=ftmp_i)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224_ndrope(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224_ndrope, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        pretrained_model = "xxx/jx_vit_base_p16_224-80ecf9dd.pth"
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformerNDRope(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformerNDRope, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
