#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Test of the vector field against an independent transcription of Hodgkin and Huxley (1952).

The transcription below is written in their own notation and sign convention (J. Physiol. 117, pp. 518-519 and
eq. (31) on p. 524): V is the displacement of the membrane potential from rest, depolarization negative, t in ms,
    alpha_n = 0.01 (V + 10) / (exp((V + 10)/10) - 1),   beta_n = 0.125 exp(V/80),
    alpha_m = 0.1 (V + 25) / (exp((V + 25)/10) - 1),    beta_m = 4 exp(V/18),
    alpha_h = 0.07 exp(V/20),                          beta_h = 1 / (exp((V + 30)/10) + 1),
    I_i = g_K n^4 (V - V_K) + g_Na m^3 h (V - V_Na) + g_l (V - V_l),
    g_Na = 120, g_K = 36, g_l = 0.3, V_Na = -115, V_K = +12 (mV, mmho/cm^2), C_M = 1 uF/cm^2,
    d^2V/dt^2 = K (dV/dt + I_i / C_M),   dx/dt = phi (alpha_x (1 - x) - beta_x x),   phi = 3^((T - 6.3)/10),
evaluated with mpmath at 60 digits (the two removable singularities by their limits 0.1 and 1). The programs use
u = -V, w = u' (hhjet6.vfield, the field that the integrator and the jets are built on), so at a state (u, w, m, n, h)
the two must agree with dV/dt = -w, d^2V/dt^2 = -w' and V_l = -E_l. Compared at random states (a seeded generator)
at 18.5 C and 6.3 C, for E_l = 10.613 and for the zero-current E_l of certify_rest_wave.rest_state, and at the two
removable singularities and next to them. Negative control: the same transcription with V_K = -12 must be caught
(a relative difference above 1e-6 at some state).
Output: data/test_field.txt; exit status 0 iff all agree to 1e-40 (relative) and the control disagrees.
"""
import os
import random
import sys
from mpmath import mp, mpf
from flint import arb, ctx
import certify_rest_wave as C
import hhjet6

mp.dps = 60
ctx.prec = 256
HERE = os.path.dirname(os.path.abspath(__file__))


def efrac(x, c):
    """c x / (exp(x) - 1), with its limit c at x = 0."""
    return c if x == 0 else c * x / mp.expm1(x)


def hh1952(V, dV, m, n, h, K, phi, Vl, VK=mpf(12)):
    """The 1952 equations in their sign convention: returns (dV/dt, d^2V/dt^2, dm/dt, dn/dt, dh/dt)."""
    an = efrac((V + 10) / 10, mpf('0.1'))           # 0.01 (V + 10) / (exp((V + 10)/10) - 1)
    bn = mpf('0.125') * mp.exp(V / 80)
    am = efrac((V + 25) / 10, mpf(1))               # 0.1 (V + 25) / (exp((V + 25)/10) - 1)
    bm = 4 * mp.exp(V / 18)
    ah = mpf('0.07') * mp.exp(V / 20)
    bh = 1 / (mp.exp((V + 30) / 10) + 1)
    Ii = 36 * n ** 4 * (V - VK) + 120 * m ** 3 * h * (V - mpf(-115)) + mpf('0.3') * (V - Vl)
    return [dV, K * (dV + Ii), phi * (am * (1 - m) - bm * m), phi * (an * (1 - n) - bn * n),
            phi * (ah * (1 - h) - bh * h)]


def programs(u, w, m, n, h, K, phi, EL):
    f = hhjet6.vfield([arb(u), arb(w), arb(m), arb(n), arb(h), arb(K)], phi, EL)
    return [mpf(v.mid().str(70, radius=False)) for v in f[:5]]


def rel(a, b):
    return max(abs(x - y) / max(abs(x), abs(y), mpf(1)) for x, y in zip(a, b))


def main():
    rng = random.Random(1952)
    _, EL0 = C.rest_state() if not C.HH_EL else (None, None)
    if EL0 is None:
        raise SystemExit('run without HH_EL (the zero-current E_l is one of the cases)')
    cases = []
    for T in ('18.5', '6.3'):
        for ELs in ('10.613', EL0.mid().str(70, radius=False)):
            for _ in range(12):
                st = ['%.25f' % rng.uniform(-20, 110), '%.25f' % rng.uniform(-60, 60), '%.25f' % rng.random(),
                      '%.25f' % rng.random(), '%.25f' % rng.random(), '%.25f' % rng.uniform(4, 11)]
                cases.append((T, ELs, st))
            for u in ('10', '25', '10.000000000000000000000000001', '24.999999999999999999999999999'):
                cases.append((T, ELs, [u, '3.5', '0.2', '0.4', '0.5', '10.4']))
    worst = mpf(0)
    worst_ctrl = mpf(0)
    for T, ELs, st in cases:
        phi_a = C.phi_of(T)
        phi_m = mpf(3) ** ((mpf(T) - mpf('6.3')) / 10)
        u, w, m, n, h, K = [mpf(x) for x in st]
        got = programs(*st, phi_a, arb(ELs))
        ref = hh1952(-u, -w, m, n, h, K, phi_m, -mpf(ELs))
        want = [-ref[0], -ref[1], ref[2], ref[3], ref[4]]
        worst = max(worst, rel(got, want))
        ctrl = hh1952(-u, -w, m, n, h, K, phi_m, -mpf(ELs), VK=mpf(-12))
        worst_ctrl = max(worst_ctrl, rel(got, [-ctrl[0], -ctrl[1], ctrl[2], ctrl[3], ctrl[4]]))
    ok = worst < mpf('1e-40')
    ctrl_ok = worst_ctrl > mpf('1e-6')
    jworst = handwritten_jacobian()
    jok = jworst < 1e-8
    lines = ['vector field of the programs (hhjet6.vfield, u = -V) against an independent transcription of the 1952',
             'equations in their convention (mpmath, 60 digits): %d states at 18.5 and 6.3 C,' % len(cases),
             'E_l = 10.613 and the zero-current value, including u = 10 and u = 25 and states next to them',
             'largest relative difference %s (needs < 1e-40): %s' % (mp.nstr(worst, 3), ok),
             'negative control (V_K = -12 in the transcription): largest relative difference %s, caught: %s' % (
                 mp.nstr(worst_ctrl, 3), ctrl_ok),
             'handwritten float Jacobian in stab_num.Model.Df against hhjet6.jacobian: largest absolute',
             'difference %s (needs < 1e-8): %s' % (('%.3e' % jworst), jok),
             'ALL TESTS PASSED' if ok and ctrl_ok and jok else 'SOME TEST FAILED']
    print('\n'.join(lines))
    open(os.path.join(HERE, '..', 'data', 'test_field.txt'), 'w').write('\n'.join(lines) + '\n')
    return ok and ctrl_ok and jok


def handwritten_jacobian():
    """stab_num.Model.Df is written by hand, in float. The proof's Jacobian is the degree-1 jet."""
    import stab_num
    worst = 0.0
    points = [(25.0, 1.0, 0.05, 0.3, 0.6, 10.4), (10.0, -2.0, 0.2, 0.4, 0.5, 4.5),
              (0.0, 0.0, 0.1, 0.4, 0.6, 8.0), (-8.0, 3.0, 0.9, 0.2, 0.7, 6.0)]
    for u, w, m, n, h, K in points:
        model = stab_num.Model.__new__(stab_num.Model)
        model.K, model.phi = K, 1.0
        J = model.Df([u, w, m, n, h])
        G = hhjet6.jacobian([arb(u), arb(w), arb(m), arb(n), arb(h), arb(K)], arb(1), arb('10.613'))
        for i in range(5):
            for j in range(5):
                worst = max(worst, abs(J[i, j] - float(G[i][j].mid())))
    return worst


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
