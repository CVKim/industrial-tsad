# Third-party notices

| Component in this repository | Origin | License | How it is used |
|---|---|---|---|
| `industrial_tsad/core/models/paano` | PaAno, Park & Kang, ICLR 2026 — https://github.com/jinnnju/PaAno | MIT | ported (patch encoder, training objective, k-means memory bank, kNN score); extensions documented in the module docstrings |
| `industrial_tsad/core/models/kanad` | KAN-AD, Zhou et al., ICML 2025 — https://github.com/issaccv/KAN-AD | MIT | the model class is imported from a local clone at run time (not vendored); training/scoring re-implemented |
| `industrial_tsad/core/models/axonad`, `industrial_tsad/evaluators/metrics.py`, `industrial_tsad/core/models/timesnet` | AxonAD, ECML-PKDD 2026 (https://github.com/iis-esslingen/AxonAD), which ships TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B) | Apache-2.0 | imported from a local clone at run time (not vendored) |
| `industrial_tsad/core/models/catch` | CATCH, Wu et al., ICLR 2025 — https://github.com/decisionintelligence/CATCH | no license file in the repository | imported from a local clone at run time; **no CATCH code is copied here** |
| `industrial_tsad/core/models/mscad` | MSCAD, Guo et al., arXiv 2609.38004 (no code released) | — | our re-implementation from the paper text |
| `docs/diagrams/*.html`, `docs/assets/architecture-*.svg`, `docs/assets/protocol-*.svg` | generated with Archify (https://github.com/tt-a1i/archify) from `docs/diagrams/*.archify.json` | MIT (notice below); the files embed JetBrains Mono subsets under the SIL Open Font License 1.1, whose text is included in each file | documentation only |

Copyright notice of PaAno (MIT):

```
MIT License

Copyright (c) 2026 jinnnju

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Copyright notice of Archify (MIT), whose viewer is embedded in `docs/diagrams/*.html`:

```
MIT License

Copyright (c) 2026 tt-a1i (Archify)
Copyright (c) 2025 Cocoon AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
