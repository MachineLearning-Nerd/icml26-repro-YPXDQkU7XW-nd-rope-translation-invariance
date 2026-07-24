# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified model creation / weight loading / state_dict helpers

import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timesformer.models.features import FeatureListNet, FeatureDictNet, FeatureHookNet
from timesformer.models.conv2d_same import Conv2dSame
from timesformer.models.linear import Linear


_logger = logging.getLogger(__name__)

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


import math
import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url as model_zoo

def load_pretrained(
    model,
    cfg=None,
    num_classes=1000,
    in_chans=3,
    filter_fn=None,
    img_size=224,
    num_frames=8,
    num_patches=196,
    attention_type='divided_space_time',
    pretrained_model="",
    strict=True,
):
    """
    Robust pretrained loader for ViT/TimeSformer(+NDRoPE).
    - Respects model.default_cfg if cfg is None
    - Supports local ckpt path or URL in cfg['url']
    - Handles in_chans != 3
    - Removes/ignores abs pos/time embeddings for NDRoPE
    - Resizes pos/time embeddings if present & needed
    - Duplicates spatial attn weights to temporal_attn for divided_space_time
    - Deletes classifier weights on num_classes mismatch
    """
    # ---- default cfg ----
    if cfg is None:
        cfg = getattr(model, 'default_cfg', None)
    if cfg is None:
        print("[load] No default_cfg on model; random init.")
        return

    # ---- load state dict (url or local) ----
    state_dict = None
    url = cfg.get('url', '')
    if pretrained_model:
        try:
            print("Try to load model weight from local")
            obj = torch.load(pretrained_model, map_location='cpu')
            # try common nesting
            if isinstance(obj, dict):
                if 'model' in obj and isinstance(obj['model'], dict):
                    state_dict = obj['model']
                elif 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                    state_dict = obj['state_dict']
                else:
                    # maybe already a flat state dict
                    state_dict = obj
            else:
                state_dict = obj
        except Exception as e:
            print(f"[load] Failed to load {pretrained_model}: {e}")
            return
    else:
        if url:
            state_dict = model_zoo(url, progress=False, map_location='cpu')
        else:
            print("[load] No pretrained path or url; random init.")
            return

    # ---- optional filter ----
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    # ---- handle NDRoPE: strip absolute pos/time embeddings ----
    use_ndrope = bool(getattr(model, "use_ndrope", False))
    if use_ndrope:
        to_del = [k for k in state_dict.keys()
                  if k.endswith('pos_embed') or k.endswith('time_embed') or
                     ('pos_embed' in k) or ('time_embed' in k)]
        for k in to_del:
            state_dict.pop(k, None)

    # ---- first conv channel adaptation (for timm-style stems) ----
    first_conv = cfg.get('first_conv', None)  # e.g., 'patch_embed.proj'
    if first_conv and (first_conv + '.weight') in state_dict:
        w = state_dict[first_conv + '.weight']  # [O, I, kH, kW]
        if in_chans == 1:
            print(f"[load] Convert first conv {first_conv} weights 3->1")
            wt = w.float()
            O, I, J, K = wt.shape
            if I > 3 and I % 3 == 0:
                wt = wt.reshape(O, I // 3, 3, J, K).sum(dim=2, keepdim=False)
            else:
                wt = wt.sum(dim=1, keepdim=True)
            state_dict[first_conv + '.weight'] = wt.to(w.dtype)
        elif in_chans != 3:
            wt = w.float()
            O, I, J, K = wt.shape
            if I != 3:
                print(f"[load] Drop first conv {first_conv} (I={I}) for in_chans={in_chans}")
                state_dict.pop(first_conv + '.weight', None)
                strict = False
            else:
                repeat = int(math.ceil(in_chans / 3))
                wt = wt.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
                wt *= (3.0 / float(in_chans))
                state_dict[first_conv + '.weight'] = wt.to(w.dtype)

    # ---- classifier head (num_classes mismatch) ----
    classifier = cfg.get('classifier', 'head')
    head_w_key = classifier + '.weight'
    head_b_key = classifier + '.bias'
    if head_w_key in state_dict:
        w = state_dict[head_w_key]
        need_1001_to_1000 = (num_classes == 1000 and cfg.get('num_classes', None) == 1001)
        if need_1001_to_1000 and w.size(0) == 1001:
            # strip background
            state_dict[head_w_key] = state_dict[head_w_key][1:]
            if head_b_key in state_dict:
                state_dict[head_b_key] = state_dict[head_b_key][1:]
        elif w.size(0) != num_classes:
            # remove head for re-init
            state_dict.pop(head_w_key, None)
            state_dict.pop(head_b_key, None)
            strict = False

    # ---- absolute pos/time embedding resize (if not NDRoPE and present) ----
    if not use_ndrope:
        # pos_embed: [1, 1+HW, C]
        pe_key = 'pos_embed'
        if pe_key in state_dict:
            pe = state_dict[pe_key]
            if pe is not None and pe.ndim == 3 and pe.size(0) == 1:
                old_tokens = pe.size(1)
                if num_patches is not None and (num_patches + 1) != old_tokens:
                    # split cls + grid
                    cls_pe = pe[:, :1, :]          # [1,1,C]
                    grid = pe[:, 1:, :]            # [1,HW,C]
                    C = grid.size(-1)
                    old_hw = int(grid.size(1) ** 0.5)
                    if old_hw * old_hw == grid.size(1):
                        grid = grid.reshape(1, old_hw, old_hw, C).permute(0, 3, 1, 2)  # [1,C,H,W]
                        new_hw = int(num_patches ** 0.5)
                        grid = F.interpolate(grid, size=(new_hw, new_hw), mode='bicubic', align_corners=False)
                        grid = grid.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, C)
                        state_dict[pe_key] = torch.cat([cls_pe, grid], dim=1)
                    else:
                        # fallback: 1D nearest
                        grid = grid.transpose(1, 2)  # [1,C,HW]
                        grid = F.interpolate(grid, size=(num_patches,), mode='nearest').transpose(1, 2)
                        state_dict[pe_key] = torch.cat([cls_pe, grid], dim=1)

        # time_embed: [1, T, C]
        te_key = 'time_embed'
        if te_key in state_dict:
            te = state_dict[te_key]
            if te is not None and te.ndim == 3 and te.size(0) == 1 and te.size(1) != num_frames:
                te2 = te.transpose(1, 2)  # [1,C,T]
                te2 = F.interpolate(te2, size=(num_frames,), mode='nearest')
                state_dict[te_key] = te2.transpose(1, 2)

    # ---- divided_space_time: copy spatial attn -> temporal_attn if temporal params absent ----
    if attention_type == 'divided_space_time':
        new_sd = state_dict.copy()
        for k, v in state_dict.items():
            if 'blocks' in k and '.attn.' in k:
                k2 = k.replace('.attn.', '.temporal_attn.')
                if k2 not in state_dict:
                    new_sd[k2] = v
            if 'blocks' in k and '.norm1.' in k:
                k2 = k.replace('.norm1.', '.temporal_norm1.')
                if k2 not in state_dict:
                    new_sd[k2] = v
        state_dict = new_sd

    # ---- final load ----
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} e.g., {missing[:5]}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} e.g., {unexpected[:5]}")
    if not strict:
        print("[load] used strict=False for head/conv adaptation.")


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: dict = None,
        feature_cfg: dict = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Callable = None,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = deepcopy(default_cfg)

    if pruned:
        model = adapt_model_from_file(model, variant)

    # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn, strict=pretrained_strict)

    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg

    return model
