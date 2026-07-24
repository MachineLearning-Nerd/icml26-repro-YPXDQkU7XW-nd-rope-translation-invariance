# Claim 4 source audit

Table 5 reports 78.51 for nD-RoPE and 71.34 for RoPE-Mixed at 30 degrees.
Section 5.2 specifies resize to 256, rotation, then center crop to 224 with
fixed models and no fine-tuning. It does not specify interpolation, fill, or
antialias settings. The pinned official release contains neither the fixed
checkpoints nor a rotation evaluation entrypoint.
