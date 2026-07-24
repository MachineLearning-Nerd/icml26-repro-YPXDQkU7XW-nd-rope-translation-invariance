# Claim 6 method

The post-judge fixed campaign executes the exact released 224x224 DeiT-S
baseline (width 384) and nD-RoPE model (width 396) on CPU under
`torch.inference_mode`. PyTorch profiler records operations from real forward
passes. A separate closed-form MAC counter derives patch, attention QKV,
attention projection, QK, AV, MLP, and classifier costs from tensor dimensions.
An independent `torch.utils.flop_counter.FlopCounterMode` pass intercepts
supported operations at PyTorch dispatch level. It does not consume profiler
events or the symbolic formulas.

An independently instantiated width-396 baseline isolates the backbone-width
effect from rotary computation. A width-408 baseline is the monotonic negative
control. Parameter names and registered buffers are enumerated from the live
models, rather than inferred only from source text.

The verifier exits nonzero unless both dynamic counters and the symbolic delta
agree within 0.5 percentage points, the computed totals reproduce Table 8 within 0.03 GMAC,
the matched-width control explains at least 95% of the official operation
delta, attention computation strictly increases, frequency directions contain
zero trainable parameters, and every negative control behaves as expected.
