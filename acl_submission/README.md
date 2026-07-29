# ACL anonymous submission

Anonymous review build of **Construct Validity Failures in Agentic AI Benchmarks: An Empirical Audit**.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Uses `\usepackage[review]{acl}` (double-blind, line numbers). Switch to `[final]` and uncomment the author block in `main.tex` for camera-ready.

## Notes

- Figure 1 includes bootstrap 95% CIs under each off-diagonal ρ.
- Table 5 uses percentile-normalized ranks (0 = best, 100 = worst).
- Reference to the EvalEval workshop paper cites the workshop proceedings, not the main ACL proceedings.
- Current length is ~11 pages with review line numbers; ACL long review limit is typically 8 content pages—trim before submission if required.
