# Anchored-claim audit

## C1 — Fourier/Hilbert derivation: verified

The finite-dimensional certificate instantiates the paper's Section 4.1 chain:
translation of a content function, Parseval transfer to Fourier space, a
frequency-dependent linear map preserving the content kernel, and multiplicative
phase modulation. Across 7,680 random query/key/position trials in dimensions
1, 2, 3, 4, 5, and 8, simultaneous global translation changes the rotary score
by at most `1.42e-14`; the independently evaluated displacement-only expression
matches to the same tolerance. A separate set of 192 one- and two-dimensional
finite Fourier experiments has maximum Parseval error `1.11e-15`, relative-shift
error `4.46e-14`, and kernel-preservation error `4.47e-15`.

The official float32 simplex Gram matrix agrees to `9.03e-8`, its phase bank to
`3.14e-16`, and its explicitly float32 rotary application to `1.28e-7`. An
absolute additive-position control is detected under global shifts, separating
the claimed rotary mechanism from a control that does not have the identity.

## C2 — simplex and economy optimum: verified

For every dimension from 2 through 32, the independent construction has rank
`n`, zero centroid, unit rows, off-diagonal Gram entries `-1/n`, and tight-frame
operator `(n+1)/n I`. Maximum numerical errors are `1.99e-15` (centroid),
`9.75e-16` (Equation 18 Gram), and `1.55e-15` (Equation 19). There are 7,936
random permutation-symmetry checks and 15 negative controls that drop a vertex,
perturb a vertex, or remove an axis; every control is rejected.

Equation 33 is minimized independently for 16 dimensions and four resolutions.
All 64 scalar optimizations recover `rho=e`, hence `r*=e^(1/n)`, with maximum
ratio error `4.44e-8`. Equation 37 gives `theta <= 207.127` for the paper's
3D `D_head=128`, `M=4` configuration, placing `theta=100` inside the bound and
`10^4`, `10^6` outside.

## C3 — ImageNet accuracy: inconclusive with a source confound

The exact 81.07%, 80.89%, and 80.90% values were not rerun. The release has no
checkpoint. Moreover, its nD-RoPE `deit_small` uses width 396, six heads, and
head dimension 66, whereas the released axial/mixed baselines use standard width
384 and head dimension 64. The official phase function rejects 64 because it is
not divisible by the six real channels per 2D simplex scale. The paper's own
Table 8 reports the resulting 23.36M versus 22.06M parameters. This is a material
comparison confound but does not, by itself, prove that the reported accuracies
are numerically false.

## C4 — 30-degree rotation: inconclusive

The released repository supplies the evaluation idea but no trained ImageNet
checkpoint. Without the exact fixed models, evaluating a fresh randomly
initialized model or a generic DeiT checkpoint would not test the 78.51% versus
71.34% claim. No proxy result is substituted.

## C5 — cross-modal benchmarks: falsified as written

The claim is conjunctive and identifies the 85.97-mIoU result as ModelNet40
segmentation. The released vector-attention training and test entrypoints instead:

- load `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`;
- instantiate `PartNormalDataset`;
- use 16 object categories and 50 point-level part labels; and
- calculate ShapeNetPart instance-average part IoU.

The repository's separate `ModelNetDataLoader` is used by `train_cls.py` for
40-way classification, not the 85.97-mIoU segmentation path. No SemanticKITTI
file exists in the 154-file release. Therefore the anchored benchmark identity
is false as written, irrespective of whether the Kinetics or SemanticKITTI
numbers might be correct in unreleased code.

## C6 — ablations and cost: partial, empirical claim inconclusive

Table arithmetic supports 6x10 as the best mean across the eight reported image
resolutions (`80.3813`) and theta 100 as the best mean across seven point counts
(`79.3643`). However, the release has no checkpoint or configuration for these
ablations. The Table 6 statement of 384 fixed positional channels is also not
directly reconciled by 6 scales x 10 heads x 6 channels = 360; the released image
model instead uses 11 scales x 6 heads at width 396.

Table 8 arithmetic gives +6.07% image FLOPs and +5.89% image parameters, 0% for
video, and roughly +2% FLOPs for point-cloud vector attention. This supports low
overhead in video/point-cloud settings, while "negligible" for the widened image
model remains interpretive. The empirical optimum is not claimed reproduced.
