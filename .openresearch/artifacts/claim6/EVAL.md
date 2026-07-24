# Claim 6 EVAL

Verdict: **FALSIFIED**

The decisive post-judge evidence is an independently executed Table 8 route.
The exact released 224x224 nD-RoPE image model widens DeiT-S from 384 to 396.
Dynamic CPU profiling and an independent symbolic counter show that this
strictly increases attention operations. A matched-width baseline attributes
the operation increase to the width change, while live parameter enumeration
finds zero trainable frequency parameters.

This contradicts Appendix D.4's exact statements that nD-RoPE only changes
frequency construction “without introducing additional attention cost” and
that the parameter increase is caused by additional frequency parameters.
The earlier Table 6/7 consistency findings are retained as corroboration, not
as the decisive evidence. No unavailable trained accuracy is presented as
regenerated.
