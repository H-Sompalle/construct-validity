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
- REALM long papers: 8 pages of content + unlimited Limitations / references / appendix. Limitations is a dedicated `\section*` after the Conclusion.
- Current review build: Conclusion closes on page 8; Limitations, References, and Appendix follow (exempt).
