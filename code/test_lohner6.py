#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Tests of hhjet6.py and lohner6.py (exit status 0 iff all pass, including the negative control).

1. hhjet6.jet agrees with hhjet.jet (a different Picard scheme) on values and gradients, at five points including
   u near 10 and 25 (the Psi-near-0 branch), and its K-derivatives agree with central differences in K.
2. Lohner enclosures over 1.5 ms through the upstroke (from a small box at u = 1e-3 mV, K a small ball), at order 30
   with tolerance 1e-40 and at order 8 with tolerance 1e-10 (where the remainder term does real work), contain a
   high-precision reference solution: hp_pulse.flow (plain Taylor steps of order 40, a different step sequence,
   tolerance 1e-60, exact time) from the box centre at the ball centre. The reference uses the same Taylor jets
   (hhjet6), so this tests the enclosure machinery (a priori box, remainder, set update), not the field or the jets;
   test_field.py tests the field against an independent transcription, and test 1 the jets.
3. Negative control: the order-8 run with the Lagrange remainder replaced by 0 must NOT contain the reference solution
   (so test 2 is not vacuous).
"""
import sys
import time
from flint import arb, arb_mat, ctx
import certify_rest_wave as C
import hhjet
import hhjet6
import lohner6 as L
import hp_pulse as HP

ctx.prec = 256


def test_jets(phi, EL, y):
    K = arb('10.4383548')
    ok = True
    for u0 in ['0.001', '24.9', '10.2', '60', '-8']:
        x = [arb(u0), arb('3.5'), y[2] + arb('0.01'), y[3], y[4]]
        p = 25
        v5, g5 = hhjet.jet(x, K, phi, EL, p)
        v6, g6 = hhjet6.jet(x + [K], phi, EL, p)
        agree = all(v5[i][k].overlaps(v6[i][k]) and all(g5[i][k][j].overlaps(g6[i][k][j]) for j in range(5))
                    for i in range(5) for k in range(p + 1))
        dK = arb('1e-30')
        vp = hhjet6.values(x + [K + dK], phi, EL, p)
        vm = hhjet6.values(x + [K - dK], phi, EL, p)
        err = max(abs(float(((vp[i][k] - vm[i][k]) / (2 * dK) - g6[i][k][5]).mid())) /
                  (abs(float(g6[i][k][5].mid())) + 1e-30) for i in range(5) for k in range(p + 1))
        print('  jet at u = %5s: hhjet6 and hhjet overlap: %s; d/dK vs central difference: max rel err %.1e'
              % (u0, agree, err))
        ok = ok and agree and err < 1e-20
    return ok


def run_enclosure(phi, EL, y, drop_remainder, tol, p):
    x0 = [arb('1e-3'), arb('1.0892e-2'), y[2], y[3], y[4]]
    Kc = arb(arb('10.4383548291385707688931284503716').mid())
    rad = arb('1e-30')
    box = [x + arb(0, rad) for x in x0]
    xbar = [arb(v.mid()) for v in box] + [Kc]
    Cm = arb_mat(6, 6)
    for i in range(5):
        Cm[i, i] = rad
    Cm[5, 5] = arb('1e-35')
    R0 = [L.ball(-1, 1)] * 6
    X = L.LSet(xbar, Cm, R0, arb_mat([[1 if i == j else 0 for j in range(6)] for i in range(6)]), [arb(0)] * 6)
    F = L.Field(phi, EL)
    saved = L.remainder
    if drop_remainder:
        L.remainder = lambda F_, vxh, W, h, p_, nsub: [arb(0)] * 5
    try:
        X, t, ns = L.integrate(F, X, 1.5, p, tol, hmax=0.25)
    finally:
        L.remainder = saved
    # reference: plain Taylor steps of order 40, a different step sequence, exact time, from the box centre
    ref, _ = HP.flow([arb(v.mid()) for v in box[:5]], Kc, 1.5, phi, EL, 1e-60, want_jac=False, hmax=0.02)
    hx = X.hull()
    inside = all(hx[i].contains(ref[i]) for i in range(5))
    return inside, ns, hx, ref


def main():
    phi = C.phi_of('18.5')
    y, EL = C.rest_state()
    t0 = time.time()
    print('1. jets')
    ok1 = test_jets(phi, EL, y)
    print('2. enclosures through the upstroke contain a high-precision reference solution (same jets)')
    ok2 = True
    for p, tol in ((30, 1e-40), (8, 1e-10)):
        inside, ns, hx, ref = run_enclosure(phi, EL, y, False, tol, p)
        print('   order %d, tol %g: %d steps; u(1.5) in %s, reference %s; contained: %s' % (
            p, tol, ns, hx[0].str(20), ref[0].str(25), inside))
        ok2 = ok2 and inside
    print('3. negative control: order 8, tol 1e-10, Lagrange remainder dropped')
    inside_bad, ns_b, hxb, refb = run_enclosure(phi, EL, y, True, 1e-10, 8)
    print('   %d steps; u(1.5) in %s, reference %s; contained: %s (must be False)' % (
        ns_b, hxb[0].str(20), refb[0].str(25), inside_bad))
    ok = ok1 and ok2 and not inside_bad
    print('ALL TESTS PASSED' if ok else 'SOME TEST FAILED', '(%.0f s)' % (time.time() - t0))
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
