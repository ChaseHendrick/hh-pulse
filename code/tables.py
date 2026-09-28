#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""The numbers the manuscript quotes from the certificates, generated from data/ (not part of the proof).

Prints, for each proof, the row of the tables of Section 5 and the values the theorems quote (K1, K2 - K1, the speed
interval, the lower bound on max u), every one read from the certificates and rounded outward where it is a bound.
With --check, it also requires every generated table row and quoted value to appear verbatim in paper/paper.md, and
exits with status 1 if one does not (so the manuscript cannot quote a number that is not in a certificate).
Usage: python3 tables.py [--check]
"""
import json
import math
import os
import re
import sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
PAPER = os.path.join(HERE, '..', 'paper', 'paper.md')
PROOFS = [('18.5_El10.613', 'Theorem 1 (18.5 C, printed E_l)'), ('6.3_El10.613', 'Theorem 2 (6.3 C, printed E_l)'),
          ('18.5', 'Remark 1 (18.5 C, zero-current E_l)')]


def load(tag, stage):
    return json.load(open(os.path.join(DATA, 'pulse_proof_%s_%s.json' % (tag, stage))))


def ball(s):
    """(mid, rad) of an arb string '[m +/- r]' or '[+/- r]'."""
    m = re.match(r'^\[(?:([-+0-9.e]+) )?\+/- ([0-9.e+-]+)\]$', s.strip())
    if not m:
        raise ValueError('not a ball: %r' % s)
    return Decimal(m.group(1) or '0'), Decimal(m.group(2))


def up(x, digits):
    """x rounded up to `digits` significant digits."""
    e = math.floor(math.log10(abs(x))) if x != 0 else 0
    q = Decimal(1).scaleb(e - digits + 1)
    return x.quantize(q, rounding=ROUND_CEILING)


def down(x, places):
    return Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_FLOOR)


def rows(tag):
    setup, iv, k1, k2 = load(tag, 'setup'), load(tag, 'interval'), load(tag, 'K1'), load(tag, 'K2')
    ns, nm = load(tag, 'neg-shift'), load(tag, 'neg-model')
    cfg = load(tag, 'config')
    lam = setup['lambda_u'].strip('[')[:8]
    cells = re.search(r'cone True on (\d+) cells, entrance True on (\d+) cells', setup['lines'][-1])
    rho = '0.8' if tag.startswith('18.5') else '0.6'
    z1 = ball(iv['zeta_at_T_enter'][0])
    a = up(abs(z1[0]) + z1[1], 3)
    zs = ball(iv['|zeta_s|_upper_at_T_enter'])
    zs_up = up(zs[0] + zs[1], 5)
    esc = re.search(r't = \[([0-9.]+) \+/-', nm['reason']).group(1)
    shift = ball(ns['zeta_at_T_enter'][0])[0]
    out = [
        '| setup (H2, H3, the check of H1) | lambda_u = %s...; Lemma B at r_B = %s; z1\' > 0 and u > u* on the exit '
        'set; B0 certified on %s + %s cells | seconds |' % (lam, cfg['r_B'], cells.group(1), cells.group(2)),
        '| interval (H4) | at T_enter = %s ms: zeta_1 in [-%s, %s], abs(zeta_s) <= %s < %s | %d s |' % (
            iv['T_enter'], a, a, zs_up, rho, iv['secs']),
        '| K1 (H5) | enters K- at %s ms, path in int B0 | %d s |' % (k1['t_cone'], k1['secs']),
        '| K2 (H5) | enters K+ at %s ms, path in int B0 | %d s |' % (k2['t_cone'], k2['secs']),
        '| negative control: K interval shifted by 40 half-widths | zeta_1 about %d at T_enter, outside B0: fails, '
        'as it must | %d s |' % (round(float(shift)), ns['secs']),
        '| negative control: alpha_m times (1 + 1e-12 (u - u*)^2) | the whole set escapes below u = -60 mV at %s ms: '
        'fails, as it must | %d s |' % (Decimal(esc).quantize(Decimal('0.01'), rounding=ROUND_FLOOR), nm['secs']),
    ]
    summ = open(os.path.join(DATA, 'pulse_proof_%s_summary.txt' % tag)).read()
    speed = re.search(r'speed theta in \(([0-9.]+), ([0-9.]+)\)', summ)
    width = re.search(r'K2 - K1 = \[([0-9.e+-]+) \+/-', summ).group(1)
    umax = min(iv['u_max_lower_bound'], k1['u_max_lower_bound'], k2['u_max_lower_bound'])
    quotes = ['K1 = %s...' % cfg['K1_dec'][:52], '(%s, %s) m/s' % (speed.group(1), speed.group(2)),
              'max u > %s mV' % down(umax, 2)]
    return out, quotes, width


def classical(paper):
    """Hodgkin and Huxley's printed K, their reported speed and the measured speed, each as a rounding interval
    of half a unit in the last printed digit. Theorem 1's interval must be recorded as containing or excluding each."""
    from flint import arb, ctx
    import lohner6
    ctx.prec = 256
    cfg = load('18.5_El10.613', 'config')
    K1, K2 = lohner6.deser(cfg['K1']), lohner6.deser(cfg['K2'])
    summ = open(os.path.join(DATA, 'pulse_proof_18.5_El10.613_summary.txt')).read()
    speed = re.search(r'speed theta in \(([0-9.]+), ([0-9.]+)\)', summ)
    t1, t2 = arb(speed.group(1)), arb(speed.group(2))
    published = [('K in [10.465, 10.475] /ms', K1, K2, arb('10.465'), arb('10.475')),
                 ('speed in [18.75, 18.85] m/s', t1, t2, arb('18.75'), arb('18.85')),
                 ('speed in [21.15, 21.25] m/s', t1, t2, arb('21.15'), arb('21.25'))]
    missing = []
    for name, lo, hi, a, b in published:
        # the proved interval is (lo, hi) for the speed and [K1, K2] for K; either way, disjointness is hi < a or lo > b
        excludes = bool(hi < a) or bool(lo > b)
        word = 'excludes' if excludes else 'contains'
        line = 'Theorem 1 %s %s' % (word, name)
        print(line)
        bracket = name[name.index('['):]
        if not excludes or bracket not in paper or 'disjoint from the interval of Theorem 1' not in paper:
            missing.append(line)
    return missing


def main():
    check = '--check' in sys.argv
    paper = re.sub(r'\s+', ' ', open(PAPER).read()) if check else ''
    missing = []
    for tag, name in PROOFS:
        out, quotes, width = rows(tag)
        print('%s: K2 - K1 = %s' % (name, width))
        for q in quotes:
            print('  ' + q)
        print('| stage | result | CPU time |\n|---|---|---|')
        print('\n'.join(out))
        print()
        if check and tag != '18.5':                  # Remark 1 quotes only the speed; Section 5 tabulates 1 and 2
            missing += [r for r in out + quotes if re.sub(r'\s+', ' ', r) not in paper]
        elif check:
            missing += [q for q in quotes[1:2] if q not in paper]
    if check:
        missing += classical(paper)
    if check:
        for m in missing:
            print('NOT IN THE MANUSCRIPT: ' + m)
        print('every generated value is in paper/paper.md' if not missing else 'SOME VALUE IS NOT IN THE MANUSCRIPT')
        return 1 if missing else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
