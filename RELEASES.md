# Releases

Each release of this repository is archived on Zenodo with its own DOI. The manuscript is a preprint and has not been
peer reviewed.

## 1.0.5 (2026-10-02)

Editorial update. The manuscript now ends with Funding and a labelled statement on the use of AI (**Use of AI.**) at the body's own size; it had neither. The interval notation of Appendix A (for example `[x_k](Q)`), which the Markdown-to-PDF build had read as links, now prints as written. Hastings (1976), Foote and Chen (1981) and Carpenter (1976) are cited for what is known of them, with their reading basis in Appendix C, without apology. The hypothesis labels in the table of runs match the text, (H2). Names and titles carry their diacritics (Ważewski, Zgliczyński, and the title of Lohner's dissertation), spelling follows American usage, and Hodgkin and Huxley (1952) gains its DOI. The data availability paragraph cites the companion's Zenodo concept DOI. Numerical inputs, proof programs, certificates and results are unchanged. No new proof or scientific validation is claimed; previous archives remain unchanged.

## 1.0.4 (2026-09-29)

**DOI:** [10.5281/zenodo.23050604](https://doi.org/10.5281/zenodo.23050604) (2026-09-30).

Figure layout update. Replaces the gating-variable keys inside the right-hand pulse panels with one shared key above all four panels, with reserved figure margins. The m, n and h styles, temperature labels, independent coordinate scales and resting-voltage guide are preserved. The vector figure and manuscript PDF were rebuilt and inspected at manuscript scale. Shooting-node input hashes and the display-check results are unchanged. Scientific captions, numerical results, proof programs and certificates are unchanged. No new proof or scientific validation is claimed; previous archives remain unchanged.

## 1.0.3 (2026-09-29)

**DOI:** [10.5281/zenodo.23048254](https://doi.org/10.5281/zenodo.23048254). Publication / Preprint; both the actual GitHub source ZIP and the downloaded Zenodo ZIP contain the reviewed manuscript PDF byte for byte.

Publication figures, contact and rights update. Adds a vector voltage/gating figure reconstructed from the stored printed-leak shooting nodes, with segment and refinement diagnostics. The manuscript uses the updated public research contact. Manuscript rights are stated outside the scientific abstract, preserving the existing policy and earlier license grants. Archive metadata identifies mixed component rights rather than applying the code license to the whole preprint ZIP. Reference-list reading-status annotations have been removed where present. No theorem, proof program or certificate changes. The release includes its rebuilt manuscript PDF; previous archives remain unchanged.

## 1.0.2 (2026-09-29)

**DOI:** [10.5281/zenodo.23047089](https://doi.org/10.5281/zenodo.23047089). Publication / Preprint; the downloaded archive ZIP contains the registered manuscript PDF.

Publication metadata and packaging update. The manuscript now identifies the public companion and its immutable checking-release archive. The source ZIP includes the rebuilt manuscript PDF. Citation metadata includes a usable publication locator and explains the component license terms. No theorem, proof program, certificate or scientific claim changes. Earlier archives remain available unchanged.

Excludes seven unfinished stability and temperature-strip scripts that were inadvertently shipped in
previous archives; they remain in the development repository. No existence-proof file is excluded.

Adds `paper/hh-pulse.pdf`, a printable rendering of the existing Markdown manuscript, to the source ZIP. The title is capitalized consistently in the PDF, Markdown, README and citation metadata. The Markdown remains the canonical text. No theorem, proof program, certificate or scientific claim changes. Releases 1.0.0 and 1.0.1 stay available with their original DOIs.

The manuscript PDF can be rebuilt with Pandoc and a TeX PDF engine; the source and the proof programs remain included. The release publisher now checks that the registered manuscript PDF is present and unchanged in the companion's Git source ZIP before pushing a new release's tree.

## 1.0.1 (2026-09-28)

**DOI:** [10.5281/zenodo.23028512](https://doi.org/10.5281/zenodo.23028512) (2026-09-29). The previous archive is unchanged.

A checking release of the same preprint. The manuscript is unchanged. This archive adds the programs that read the printed numbers against the stored certificates: `code/check_abstract.py`, `code/check_speed.py`, `code/check_speed_decimal.py`, `code/check_fail.py` and `code/check_hypotheses.py`, and the exact-solution, Hopf, Nagumo and pendulum checks `code/suite_exact.py`, `code/suite_hopf.py`, `code/suite_nagumo.py` and `code/suite_pendulum.py`. `code/hypotheses.json` is the ledger those checks vouch for. The earlier-work sentence still says "within them". The suites are not part of the existence proof.

## 1.0.0 (2026-09-28)

The first public release of the preprint *The propagated action potential of Hodgkin and Huxley at their 1952
constants: a computer-assisted existence proof*, with the programs that check its results and their output.

### What the paper shows

Hodgkin and Huxley computed their propagated action potential in 1952 by shooting in the conduction speed by hand. The
existence proofs that followed, by Hastings (1976) and Carpenter (1977), treat systems with artificial small
parameters.

- **The pulse at 18.5 C** (Theorem 1, computer-assisted). With the 1952 rate functions and constants as printed,
  including the leak potential 10.613 mV, the travelling-wave equation has an orbit homoclinic to rest, with the speed
  parameter in an interval of width 3e-45; for the fibre of their p. 528 the speed begins
  18.73188824788048354046831343329624387695575077 m/s (they computed 18.8 m/s).
- **The pulse at 6.3 C** (Theorem 2, computer-assisted), with the speed parameter in an interval of width 2.8e-61 and
  the speed beginning 12.31375672016229859330140853674732362368394751453213155988245 m/s.
- **The zero-current leak potential** (Remark 1): the same at 18.5 C with the leak potential that makes the resting
  current exactly zero.
- **The method:** the unstable manifold of rest leaves a small box through a transversal exit face (a cone
  condition, Lemma 1); a validated Lohner-type Taylor integrator carries the whole speed interval through the spike
  into an isolating block around rest with a cone condition (Lemma 2); a Wazewski-type shooting argument in the speed
  closes the proof. Every lemma is proved in the text, with standard facts on ordinary differential equations cited
  from Teschl's textbook.
- **Earlier work:** the searches are in Appendix C. They are not a claim of priority. Hastings (1976)
  beyond pp. 229-230 and Foote and Chen (1981) could not be obtained, and the paper says so.

### Checked by computer

- `code/run.sh all` reruns everything from scratch: the tests (`test_temperature.py`, phi from the decimal
  temperature; `test_field.py`, the vector field against an independent transcription of the 1952 equations;
  `test_lohner6.py`, the jets, and the integrator's enclosures against a high-precision solution computed with the
  same jets; each with a control that must fail), and for each of the three proofs, after deleting everything an
  earlier run of it wrote, the numerical centre (`hp_pulse.py`), the closing block (`block0.py`), the stages of
  `hh_prove_pulse.py` (the setup with its negative controls, the interval run, the two endpoint runs, a shifted speed
  interval and a perturbed rate function that must fail), the independent re-check of the block in mpmath interval
  arithmetic (`hh_block_check_iv.py`), the summary, which checks the hashes and the consistency of every certificate,
  and a control that plants stale certificates the summary must refuse. Its exit status is 0 only if every check
  passed and every control failed for its stated reason.

### Files

- `paper/paper.md`: the manuscript.
- `code/`: the programs, `run.sh` and `requirements.txt`.
- `data/`: the certificates (`pulse_proof_*.json`) and summaries of the three proofs, the numerical centres, the
  closing blocks, the reports of the independent block check and of the tests, and the starting profiles.

### Reproduce

```
python3 -m pip install -r code/requirements.txt
sh code/run.sh all
```

### License

The manuscript in `paper/` is Copyright (c) 2026 Chase Hendrick, all rights reserved. The programs in `code/` and the
data in `data/` are licensed under the Apache License 2.0.
