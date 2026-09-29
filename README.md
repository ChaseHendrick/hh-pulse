# The propagated action potential of Hodgkin and Huxley at their 1952 constants: a computer-assisted existence proof

**Chase Hendrick**, Independent Researcher · [ORCID 0009-0002-9754-6087](https://orcid.org/0009-0002-9754-6087)

**Preprint**, release 1.0.1 (2026-09-29), [doi:10.5281/zenodo.23028512](https://doi.org/10.5281/zenodo.23028512). Release 1.0.0 remains at [doi:10.5281/zenodo.23013935](https://doi.org/10.5281/zenodo.23013935). Not peer reviewed. The GitHub release is [1.0.1](https://github.com/ChaseHendrick/hh-pulse/releases/tag/1.0.1).

**[Read the manuscript](paper/paper.md)**

## Abstract

In 1952 Hodgkin and Huxley computed their propagated action potential by hand. They shot in the conduction speed on their travelling-wave equation (J. Physiol. 117, eq. (31)), and noted that the solution goes off towards plus or minus infinity on the two sides of the speed.

The existence proofs that followed do not treat that equation. Hastings (1976) and Carpenter (1977) slow or speed the gating variables by a small parameter.

We prove that the unmodified equation has a pulse, an orbit homoclinic to rest, at 18.5 C and at 6.3 C. The rate functions and constants are those Hodgkin and Huxley printed, including the leak potential 10.613 mV. The proof is computer-assisted, in ball arithmetic. The speed parameter K lies in an interval of width 3e-45 at 18.5 C and of width 2.8e-61 at 6.3 C.

For the fibre constants of their p. 528, the conduction speed begins

    18.73188824788048354046831343329624387695575077 m/s

at 18.5 C, against the 18.8 m/s they computed, and

    12.31375672016229859330140853674732362368394751453213155988245 m/s

at 6.3 C. Every digit shown is proved.

The orbit leaves rest along its one-dimensional unstable manifold. A validated Taylor integrator carries a whole interval of speeds through the spike. An isolating block with a cone condition around rest, and a Wazewski-type shooting argument, closes it.

## Status of the results

- **Proved, computer-assisted** (ball arithmetic, FLINT/Arb through python-flint, 256 bits): Theorem 1 (18.5 C) and
  Theorem 2 (6.3 C), with the printed leak potential; Remark 1, the same at 18.5 C with the leak potential that makes
  the resting current zero; the enclosure of the rest state. Lemmas 0, 1 and 2 and the lemmas of Appendices A and B
  are proved in the text; the only outside results used are standard facts about ordinary differential equations,
  cited from Teschl's textbook with theorem and page.
- **Numerical, not proved:** the speed parameter to about 58 digits by high-precision shooting (Remark 2); the profile.
- **Not claimed:** uniqueness of the pulse, stability, other temperatures, the slow pulse.
- **Earlier work, as far as the search reached:** Appendix C records the searches. They are not a claim of priority.
  Two papers could not be obtained in full: Hastings (1976), read on pp. 229-230 only, and Foote
  and Chen (1981), not read at all; zbMATH Open has no review of either. Carpenter (1977) was read in full and treats
  modified systems with small parameters. No step of the proof depends on these papers.
- **Checked by the programs:** negative controls (a shifted speed interval, a perturbed rate function, a bracket
  above the unstable eigenvalue, thinner exit faces, an enlarged block) fail as they must, each for its stated
  reason; the closing block is re-checked by an independent program in mpmath interval arithmetic; the vector field
  is compared with an independent transcription of the 1952 equations; the integrator's enclosures are tested against
  a high-precision solution computed with the same Taylor jets and a different step sequence, with a control that must
  miss; the temperature factor is checked to be computed from the decimal temperature. Every certificate records the
  hashes of the configuration, the closing block and the programs it was computed from, and the summary refuses any
  certificate that does not match or contradicts itself; a control plants such certificates and must see them
  refused.

## Contents

| Folder | What is in it |
|---|---|
| [`paper/`](paper/) | The manuscript, [`paper.md`](paper/paper.md) |
| [`code/`](code/) | The programs, [`run.sh`](code/run.sh) to rerun everything, and [`requirements.txt`](code/requirements.txt) |
| [`data/`](data/) | The certificates of the three proofs (`pulse_proof_*.json`) and their summaries, the numerical centres (`hp_pulse_*.json`), the closing blocks, the reports of the independent block check and of the tests, and the starting profiles |

| Program | What it does | Time |
|---|---|---|
| [`hh_prove_pulse.py`](code/hh_prove_pulse.py) | The proof, stage by stage: configuration, setup (Lemma B, the block, the check of (H1)), the interval run, the endpoint runs, the two negative controls, the summary and its control | 1 h (18.5 C), 2 h (6.3 C) |
| [`hp_pulse.py`](code/hp_pulse.py) | Numerical: the speed parameter by multiple shooting at 256 bits, to centre the interval | 10 min (18.5 C), 20 min (6.3 C) |
| [`block0.py`](code/block0.py) | The closing block and its cone and entrance conditions | seconds |
| [`hh_block_check_iv.py`](code/hh_block_check_iv.py) | Independent re-check of the block in mpmath interval arithmetic | seconds |
| [`test_lohner6.py`](code/test_lohner6.py) | Tests of the jets and of the integrator's enclosures, with a negative control | 3 min |
| [`test_field.py`](code/test_field.py) | The vector field against an independent transcription of the 1952 equations, with a negative control | seconds |
| [`test_temperature.py`](code/test_temperature.py) | phi = 3^((T - 6.3)/10) from the decimal temperature, with the binary 6.3 as negative control | seconds |
| [`tables.py`](code/tables.py) | Not part of the proof: prints the values the manuscript quotes from the certificates, and with `--check` requires each in the manuscript | seconds |
| [`certify_rest_wave.py`](code/certify_rest_wave.py), [`lohner6.py`](code/lohner6.py), [`hhjet6.py`](code/hhjet6.py), [`hhseries.py`](code/hhseries.py), [`hhjet.py`](code/hhjet.py) | The rest state and Lemmas A and B; the integrator; the Taylor jets of the field | |
| [`hhwave.py`](code/hhwave.py), [`pulse_bvp.py`](code/pulse_bvp.py) | Double-precision model and the boundary-value solver that made the starting profiles | |

## Reproduce

From this folder:

```
python3 -m pip install -r code/requirements.txt
sh code/run.sh all          # or: tests, 18.5, 6.3, zero
```

`run.sh` reruns every computation from scratch, each process under `nice` and a time limit. For each proof it first
deletes every file an earlier run of that proof wrote, so no old certificate can stand in for a stage that did not
run; a stage that must pass must exit with status 0, and a negative control must end with a written FAIL verdict.
It exits with status 0 only if every check passed, every negative control failed for its stated reason, and the
summary, which checks the hashes of every certificate, accepted them all.

## Cite

```bibtex
@misc{hendrick2026hhpulse,
  author = {Hendrick, Chase},
  title  = {The propagated action potential of {Hodgkin} and {Huxley} at their 1952 constants: a computer-assisted existence proof},
  year   = {2026},
  doi    = {10.5281/zenodo.23028512},
  url    = {https://doi.org/10.5281/zenodo.23028512}
}
```

## License

The manuscript in `paper/` is Copyright (c) 2026 Chase Hendrick, all rights reserved. The programs in `code/` and the
data in `data/` are under the Apache License 2.0 (see `NOTICE`). The paper's own repository, written when it is
published, and each of its releases carry a `LICENSE` file with both texts.
