# Publication gate

The local publication gate is `repro/src/publication_gate.py`. It is
stdlib-only and checks the checked-in evidence without launching the expensive
campaign:

```bash
python3 repro/src/publication_gate.py --skip-producers
```

The gate requires:

- the normalized project/repository name and current documentation;
- valid SHA-256 matches for the pinned arXiv source archive, PDF, and HTML
  source record;
- the official implementation commit and vendored manifest pin;
- C1/C2 `verified`, C3/C4 `BLOCKED`, and C5/C6 `FALSIFIED` evidence outputs;
- the 52-check independent verifier and six mutation controls to pass;
- the evaluator release gate to pass;
- no top-level `.trackio`, environment files, token-like values, private keys,
  or machine-specific absolute home-directory paths in publication files; and
- a clean, textually coherent reader-facing surface.

The command writes the machine-readable result to
`outputs/publication_gate.json`. The gate is a publication hygiene and
evidence-integrity check; it does not turn blocked empirical claims into
verified results and does not issue an ICML or evaluator score.
