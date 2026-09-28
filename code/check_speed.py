#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Independent algebraic check of the conduction speeds in the pulse certificates.

The manuscript (paper/paper.md, Section 2) states

    theta = sqrt(K a / (2 R_2 C_M)),

with a = 238 um, R_2 = 35.4 ohm cm and C_M = 1 uF/cm^2. K is in 1/ms. In the
units next to that formula in the proof (a = 0.0238 cm, R_2 = 35.4 ohm cm,
C_M = 1e-6 F/cm^2) the speed in m/s is

    theta = sqrt(K * 1000 * a / (2 * R_2 * C_M)) / 100.

This program types those constants itself. It does not import the proof.

The K1 and K2 stage certificates store a K ball and no speed. The exact
endpoints are the dyadic fractions in the configuration certificate. The speed
interval is the one printed in the summary and in paper/paper.md. For each of
18.5 C and 6.3 C (printed leak, tag El10.613) the recomputed enclosure of
theta([K1, K2]) must match that printed interval through every leading digit
the two printed endpoints share, and the printed interval must contain the
enclosure. Shifting K by a relative 1e-6 must make the recomputed speed miss
the printed interval; if it still overlaps, the comparison is tightened.

Exit status 0 only if both temperatures match and both shifted speeds miss.
"""
import json
import os
import re
import sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, getcontext

from flint import arb, ctx

ctx.prec = 1024
getcontext().prec = 800

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
PAPER = os.path.join(HERE, '..', 'paper', 'paper.md')

# a = 238 um = 238 * 10^{-4} cm, R_2 = 35.4 ohm cm, C_M = 10^{-6} F/cm^2.
A_UM = 238
R2_TENTHS = 354          # 35.4 = 354/10
CM_EXP = 6               # C_M = 10^{-6}
REL_SHIFT = arb(1) / arb(10) ** 6   # exact 10^{-6}

CASES = (
    ('18.5_El10.613', '18.5 C, printed leak'),
    ('6.3_El10.613', '6.3 C, printed leak'),
)


def deser(s):
    """Exact dyadic ball [mantissa * 2^exp +/- radius], as stored in the config JSON."""
    m, e, rm, re_ = s
    c = arb(int(m)) * arb(2) ** int(e)
    if int(rm) == 0:
        return c
    return c + arb(0, arb(int(rm)) * arb(2) ** int(re_))


def speed(K):
    """theta in m/s from K in 1/ms.

    sqrt(K * 1000 * (238/10000) / (2 * (354/10) * 10^{-6})) / 100
    simplifies, with no rounding, to sqrt(K * 238 * 10^6 / (2 * 354)) / 100.
    """
    return (K * arb(A_UM) * arb(10) ** CM_EXP / (arb(2) * arb(R2_TENTHS))).sqrt() / arb(100)


def speed_dimensional(K):
    """The same formula written with the three fibre constants kept separate."""
    a_cm = arb(A_UM) / arb(10) ** 4
    r2 = arb(R2_TENTHS) / arb(10)
    cm = arb(10) ** (-CM_EXP)
    return (K * arb(1000) * a_cm / (arb(2) * r2 * cm)).sqrt() / arb(100)


def to_decimal(exact):
    """Decimal equal to an exact arb (a dyadic rational)."""
    if not exact.is_exact():
        raise RuntimeError('expected an exact endpoint')
    m, e = exact.man_exp()
    return Decimal(int(m)) * (Decimal(2) ** int(e))


def places_str(exact, places, rounding):
    q = Decimal(1).scaleb(-places)
    return format(to_decimal(exact).quantize(q, rounding=rounding), 'f')


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    if n and a[n - 1] == '.':
        n -= 1
    return a[:n]


def sig_digits(s):
    return len(s.replace('.', '').lstrip('+-'))


def window(a, b):
    """Short view of the first position where two decimal strings differ."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    lo = max(0, n - 12)
    return n, a[lo:n + 8], b[lo:n + 8]


def load_case(tag):
    cfg = json.load(open(os.path.join(DATA, 'pulse_proof_%s_config.json' % tag)))
    k1 = json.load(open(os.path.join(DATA, 'pulse_proof_%s_K1.json' % tag)))
    k2 = json.load(open(os.path.join(DATA, 'pulse_proof_%s_K2.json' % tag)))
    summ = open(os.path.join(DATA, 'pulse_proof_%s_summary.txt' % tag)).read()
    found = re.search(r'speed theta in \(([0-9.]+), ([0-9.]+)\) m/s', summ)
    if not found:
        raise RuntimeError('%s summary has no speed interval' % tag)
    return cfg, k1, k2, found.group(1), found.group(2)


def enclosure(K1, K2):
    """Rigorous (lower, upper) of theta on [K1, K2], as exact arbs, for K > 0."""
    t1, t2 = speed(K1), speed(K2)
    return t1.lower(), t2.upper(), t1, t2


def inside(inner_lo, inner_hi, outer_lo, outer_hi):
    return bool(outer_lo < inner_lo) and bool(inner_hi < outer_hi)


def disjoint(a_lo, a_hi, b_lo, b_hi):
    """True only when the two intervals are proved not to meet."""
    return bool(a_hi < b_lo) or bool(a_lo > b_hi)


def main():
    lines = []
    ok = True

    def say(s=''):
        lines.append(s)

    probe = arb(1)
    if not speed(probe).overlaps(speed_dimensional(probe)):
        say('the simplified speed formula does not meet the dimensional one')
        print('\n'.join(lines))
        return 1

    paper = open(PAPER).read()
    say('theta = sqrt(K * a / (2 * R_2 * C_M))')
    say('a = 238 um = 0.0238 cm, R_2 = 35.4 ohm cm, C_M = 1 uF/cm^2 = 1e-6 F/cm^2')
    say('K in 1/ms; theta in m/s = sqrt(K * 1000 * a_cm / (2 * R_2 * C_M)) / 100')
    say('K1/K2 stage JSON stores no speed; the speed checked below is the summary, also printed in paper.md')

    for tag, title in CASES:
        say()
        say(title + ' (' + tag + ')')
        cfg, k1, k2, lo_s, hi_s = load_case(tag)
        K1, K2 = deser(cfg['K1']), deser(cfg['K2'])
        if not (K1.is_exact() and K2.is_exact() and bool(K1 < K2)):
            say('  K endpoints are not exact dyadics with K1 < K2')
            ok = False
            continue
        balls_ok = arb(k1['K']).contains(K1) and arb(k2['K']).contains(K2)
        if not balls_ok:
            say('  K1.json / K2.json balls do not contain the config endpoints')
        no_speed = 'speed' not in k1 and 'speed' not in k2 and 'theta' not in k1 and 'theta' not in k2
        if not no_speed:
            say('  a stage certificate stores a speed; this checker expected none')
        in_paper = lo_s in paper and hi_s in paper
        if not in_paper:
            say('  stored speed is not the interval printed in paper.md')

        lo_b, hi_b, t1, t2 = enclosure(K1, K2)
        stored_lo, stored_hi = arb(lo_s), arb(hi_s)
        prefix = common_prefix(lo_s, hi_s)
        n_shared = sig_digits(prefix)
        places = max(len(lo_s.split('.')[1]), len(hi_s.split('.')[1])) + 8
        rec_lo = places_str(lo_b, places, ROUND_FLOOR)
        rec_hi = places_str(hi_b, places, ROUND_CEILING)
        n_lo, w_lo_a, w_lo_b = window(lo_s, rec_lo)
        n_hi, w_hi_a, w_hi_b = window(hi_s, rec_hi)
        agree_lo = sig_digits(common_prefix(lo_s, rec_lo))
        agree_hi = sig_digits(common_prefix(hi_s, rec_hi))
        contained = inside(t1, t2, stored_lo, stored_hi)
        digits_ok = agree_lo >= n_shared and agree_hi >= n_shared and common_prefix(rec_lo, rec_hi).startswith(prefix)

        say('  K1 = %s /ms' % cfg['K1_dec'])
        say('  K2 = %s /ms' % cfg['K2_dec'])
        say('  stored speed (%s, %s) m/s' % (lo_s, hi_s))
        say('  recomputed enclosure (%s, %s) m/s' % (rec_lo, rec_hi))
        say('  leading digits the stored endpoints share: %d' % n_shared)
        say('    %s' % prefix)
        say('  recomputed lower agrees with the stored lower for %d leading digits' % agree_lo)
        say('  recomputed upper agrees with the stored upper for %d leading digits' % agree_hi)
        if not digits_ok:
            say('  DIVERGE lower at digit %d: stored ...%s  recomputed ...%s' % (n_lo, w_lo_a, w_lo_b))
            say('  DIVERGE upper at digit %d: stored ...%s  recomputed ...%s' % (n_hi, w_hi_a, w_hi_b))
        say('  stored interval contains the recomputed enclosure: %s' % ('yes' if contained else 'NO'))

        K1s, K2s = K1 * (1 + REL_SHIFT), K2 * (1 + REL_SHIFT)
        s_lo, s_hi, _, _ = enclosure(K1s, K2s)
        misses = disjoint(s_lo, s_hi, stored_lo, stored_hi)
        say('  negative control K * (1 + 1e-6): speed about %s m/s' % s_lo.str(20, radius=False))
        say('  shifted speed misses the stored interval: %s' % ('yes' if misses else 'NO, still overlaps'))
        if not misses:
            say('  the stored interval is too wide to see a relative 1e-6 shift in K; tightening')
            # A relative 1e-6 change in K moves theta by about 5e-7. Require the
            # recomputed speed to lie within a relative 1e-8 of the stored midpoint,
            # a band the same shift cannot reach.
            mid = (stored_lo + stored_hi) / 2
            tol = abs(mid) * arb('1e-8')
            band_lo, band_hi = mid - tol, mid + tol
            contained = inside(t1, t2, band_lo, band_hi) and digits_ok and contained
            misses = disjoint(s_lo, s_hi, band_lo, band_hi)
            say('  tightened band is the stored midpoint +/- 1e-8 relative')
            say('  real speed inside the tightened band: %s' % ('yes' if contained else 'NO'))
            say('  shifted speed misses the tightened band: %s' % ('yes' if misses else 'NO'))

        case_ok = digits_ok and contained and misses and in_paper and balls_ok and no_speed
        say('  %s' % ('ok' if case_ok else 'FAILED'))
        ok = ok and case_ok

    say()
    say('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    print('\n'.join(lines))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
