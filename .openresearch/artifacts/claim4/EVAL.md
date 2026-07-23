# Claim 4 EVAL

Final verdict after four routes: **BLOCKED**.

The exact zero-shot metric cannot yet be verified or falsified because the
fixed checkpoints, full ImageNet validation evidence, and complete transform
definition are absent, and the release has no rotation evaluator. No proxy
rotation test is promoted. Route 3 also confirms that resize interpolation,
rotation interpolation, fill, and antialias behavior are unspecified.

The mandatory falsification route rejected tests on random models, the
cross-table zero-degree rank reversal, and the released architecture mismatch:
none preserves the exact fixed trained models and complete 30-degree
ImageNet protocol. No valid counterexample was established.

Unblockers: the exact Table 5 checkpoints, ImageNet-1K validation access, and
the complete deterministic transform definition.
