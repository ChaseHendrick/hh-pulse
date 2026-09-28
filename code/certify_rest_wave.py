#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): the rest state of the Hodgkin-Huxley wave ODE, its eigenvalues, and an enclosure of
the exit point of the u-increasing branch of its unstable manifold from a small block, for every K in a ball.

Model and conventions: hhwave.py (u = -V, phi = 3^((T - 6.3)/10) with T the decimal temperature as typed, enclosed
exactly). By default E_l is the value that makes the resting current zero, 10.5989..., and the rest state is
y* = (0, 0, m_inf(0), n_inf(0), h_inf(0)); with HH_EL set (HH_EL=10.613 for the printed leak potential) E_l is that
decimal number and rest is the nearby equilibrium y* = (u*, 0, m_inf(u*), n_inf(u*), h_inf(u*)), enclosed by an
interval Newton step (rest_state).

Lemma A (eigenvalues). For every K in the ball, the characteristic polynomial P of Df(y*) has exactly one root with
positive real part, a simple real root lam_u, enclosed; the quotient Q = P / (x - lam_u) is Hurwitz (all four roots
have negative real part). Checked: P(a) < 0 < P(b) and P' > 0 on [a, b] (so lam_u is the only real root in [a, b]),
then the Hurwitz inequalities for Q, whose coefficients are enclosed by synthetic division with the ball lam_u.

Lemma B (local unstable manifold). Let T be a fixed matrix (exact dyadic entries) and z = T (y - y*). Let
B = { |z1| <= r, |z2| <= s2, z3^2 + z4^2 <= s3^2, |z5| <= s5 } and A = T Df T^-1 over the box hull of y* + T^-1 B
and over the K ball, written A = Lam + E with Lam = diag(l1, l2, [[a, -b], [b, a]], l5) fixed. Checked:
 (F) every stable face is strictly inflowing: on z_j = +-s_j (j = 2, 5), sign(z_j) z_j' <= (l_j + |E_jj|) s_j +
     sum_{k != j} |E_jk| |z_k|max < 0; on z3^2 + z4^2 = s3^2, (z3^2 + z4^2)'/2 <= a s3^2 + s3 |E_34 z| < 0;
 (C) the matrix D A + A^T D, D = diag(1, -1, -1, -1, -1), is positive definite for every A in the interval matrix
     (Gershgorin on its lower bounds).
Since f(y*) = 0 and B is convex, z' = A(z) z with A(z) = int_0^1 T Df(y* + T^-1 tau z) T^-1 dtau in the interval
matrix, so (C) gives d/dt L(z) = z^T (D A(z) + A(z)^T D) z > 0 for z != 0, L = z1^2 - |z'|^2. The branch of W^u(y*)
with z1 > 0 lies in the interior of B for all early times; there L increases strictly and tends to 0 as t -> -infinity
(the orbit tends to y*), so L > 0 on it; while in B, L keeps increasing, so z1 never vanishes and the orbit cannot converge to y* (L(y*) = 0) nor stay in B forever (an
omega-limit set in B would carry an orbit with constant L, which (C) allows only at y*). It cannot leave through a
stable face (F). Hence it leaves B through the face z1 = +r at a point with |z2| <= s2, |(z3, z4)| <= s3, |z5| <= s5.
This exit set is what hh_prove_pulse.py integrates.
"""
import json
import os
import re
import sys
import numpy as np
from flint import arb, arb_mat, arb_poly, ctx
import hhjet

ctx.prec = 256


def ball(lo, hi):
    return arb(lo).union(arb(hi))


DECIMAL = re.compile(r'^[0-9]+(\.[0-9]+)?$')


def temperature(s):
    """The temperature as the decimal string it was given (argv), checked to be a plain decimal number. Every program
    passes this string, never a float, to phi_of and tag: the float 6.3 is 6.29999999999999982236... C."""
    if not isinstance(s, str) or not DECIMAL.match(s):
        raise ValueError('the temperature must be a decimal number such as 18.5 or 6.3, not %r' % (s,))
    return s


def phi_of(T):
    """phi = 3^((T - 6.3)/10) as a ball that contains the exact value for the decimal temperature T. T is a decimal
    string (temperature()); a float is converted by repr, the shortest decimal that reads back as that float, so that
    6.3 means the decimal 6.3 (arb(6.3) would be the binary number 6.29999999999999982236..., whose phi excludes 1)."""
    d = temperature(T if isinstance(T, str) else repr(float(T)))
    return arb(3) ** ((arb(d) - arb('6.3')) / 10)


# The leak potential. Default: the value that makes the resting current zero, so that rest is at u = 0. With the
# environment variable HH_EL set (e.g. HH_EL=10.613, Hodgkin and Huxley's printed V_l = -10.613 mV in their sign
# convention), E_l is that decimal number and rest is the nearby equilibrium, enclosed by an interval Newton method.
HH_EL = os.environ.get('HH_EL')


def tag(T):
    """File-name tag of a run: the temperature, and the leak potential when it is not the zero-current value."""
    return '%s' % T + ('_El%s' % HH_EL if HH_EL else '')


def _gates_inf(U):
    """m_inf, n_inf, h_inf at U (an arb or an arb_series)."""
    from hhseries import psi_series
    if isinstance(U, arb):
        x1, x2 = (25 - U) / 10, (10 - U) / 10
        am, an = x1 / (x1.exp() - 1), x2 / (x2.exp() - 1) / 10
    else:
        am, an = psi_series((25 - U) / 10, ctx.cap), psi_series((10 - U) / 10, ctx.cap) / 10
    bm, bn = (U * (-arb(1) / 18)).exp() * 4, (U * (-arb(1) / 80)).exp() / 8
    ah, bh = (U * (-arb(1) / 20)).exp() * (arb(7) / 100), 1 / (((30 - U) / 10).exp() + 1)
    return am / (am + bm), an / (an + bn), ah / (ah + bh)


def _iss(U, EL):
    m, n, h = _gates_inf(U)
    return m ** 3 * h * (U - 115) * 120 + n ** 4 * (U + 12) * 36 + (U - EL) * (arb(3) / 10)


def rest_state():
    if not HH_EL:
        am, bm = arb('2.5') / (arb('2.5').exp() - 1), arb(4)
        an, bn = arb(1) / 10 / (arb(1).exp() - 1), arb(1) / 8
        ah, bh = arb(7) / 100, 1 / (arb(3).exp() + 1)
        m, n, h = am / (am + bm), an / (an + bn), ah / (ah + bh)
        EL = (36 * n ** 4 * 12 - 120 * m ** 3 * h * 115) / (arb(3) / 10)
        return [arb(0), arb(0), m, n, h], EL
    from flint import arb_series
    EL = arb(HH_EL)
    # interval Newton for g(u) = I(u, m_inf(u), n_inf(u), h_inf(u)) = 0 on U = [-1, 1]: if
    # N(U) = mid(U) - g(mid(U)) / g'(U) lies in the interior of U, g has exactly one zero in U and it lies in N(U)
    import hhwave
    U = arb(hhwave.Wave(18.5, EL=float(HH_EL)).urest) + arb(0, '1e-3')   # float start (rest does not depend on T)
    old = ctx.cap
    for it in range(60):
        ctx.cap = 2
        dg = _iss(arb_series([U, 1]), EL).coeffs()[1]
        ctx.cap = old
        mU = arb(U.mid())
        N = mU - _iss(mU, EL) / dg
        if it == 0 and not (N.lower() > U.lower() and N.upper() < U.upper()):
            raise ArithmeticError('interval Newton for the rest state did not contract')
        N = N.intersection(U)          # later steps: the zero stays in N and in U
        if N.rad() >= U.rad() * 0.5:
            U = N
            break
        U = N
    m, n, h = _gates_inf(U)
    return [U, arb(0), m, n, h], EL


def jac(y, K, phi, EL):
    J = hhjet.jacobian(y, K, phi, EL)
    return arb_mat(J)


def lemma_A(K, phi, EL, a, b):
    y, _ = rest_state()
    A = jac(y, K, phi, EL)
    P = A.charpoly()
    c = [P[i] for i in range(6)]
    dP = P.derivative()
    npc = 256                                  # P' > 0 on [a, b], checked on 256 subintervals
    ok = P(arb(a)) < 0 and P(arb(b)) > 0 and all(
        dP(ball(a + (b - a) * i / npc, a + (b - a) * (i + 1) / npc)) > 0 for i in range(npc))
    lo, hi = arb(a), arb(b)
    for _ in range(200):                   # bisection, keeping P(lo) < 0 < P(hi)
        mid = (lo + hi) / 2
        mid = arb(mid.mid())
        v = P(mid)
        if v < 0:
            lo = mid
        elif v > 0:
            hi = mid
        else:
            break
    lam = lo.union(hi)
    # synthetic division: P = (x - lam) Q + rem
    q = [arb(0)] * 5
    q[4] = c[5]
    for k in range(4, 0, -1):
        q[k - 1] = c[k] + lam * q[k]
    rem = c[0] + lam * q[0]
    q0, q1, q2, q3, q4 = q
    hur = [q3 / q4, q2 / q4, q1 / q4, q0 / q4]
    a3, a2, a1, a0 = hur
    D2 = a3 * a2 - a1
    D3 = a3 * a2 * a1 - a1 * a1 - a3 * a3 * a0
    ok_h = all(x > 0 for x in (a3, a2, a1, a0, D2, D3)) and q4 > 0
    return dict(ok=bool(ok and ok_h and rem.contains(0)), lam_u=lam, P=P, Q=[a0, a1, a2, a3],
                D2=D2, D3=D3, rem=rem, A=A)


def real_basis(Am):
    """Real eigenbasis (columns: unstable, fast real, complex pair Re/Im, slow real) of a float matrix."""
    w, V = np.linalg.eig(Am)
    iu = int(np.argmax(w.real))
    real = [i for i in range(5) if abs(w[i].imag) < 1e-12 and i != iu]
    cplx = [i for i in range(5) if w[i].imag > 1e-12]
    real.sort(key=lambda i: w[i].real)                # fast (most negative) first
    vu = V[:, iu].real
    vu = vu / vu[0]
    vc = V[:, cplx[0]]
    vc = vc / vc[0]                                    # normalise so that the u-component is 1
    cols = [vu, V[:, real[0]].real / np.max(np.abs(V[:, real[0]].real)), vc.real, vc.imag,
            V[:, real[1]].real / np.max(np.abs(V[:, real[1]].real))]
    return np.array(cols).T, w


def exact(Mf):
    return arb_mat([[arb(float(v)) for v in row] for row in Mf])


def lemma_B(K, phi, EL, r, s, Tf=None):
    """s = (s2, s3, s5). Returns the check results and the exit set data."""
    y, _ = rest_state()
    A0 = jac(y, arb(K.mid()), phi, EL)
    Am = np.array([[float(A0[i, j].mid()) for j in range(5)] for i in range(5)])
    if Tf is None:
        V, _ = real_basis(Am)
        Tf = np.linalg.inv(V)
    T = Tf if isinstance(Tf, arb_mat) else exact(Tf)       # an exact arb_mat is used as given (hh_prove_pulse.py)
    Ti = T.inv()
    s2, s3, s5 = [arb(v) for v in s]
    r = arb(r)
    zbox = [ball(-r, r), ball(-s2, s2), ball(-s3, s3), ball(-s3, s3), ball(-s5, s5)]
    ybox = [y[i] + sum((Ti[i, k] * zbox[k] for k in range(5)), arb(0)) for i in range(5)]
    Df = jac(ybox, K, phi, EL)
    Az = T * Df * Ti
    Lm = T * A0 * Ti
    l1, l2, l5 = arb(Lm[0, 0].mid()), arb(Lm[1, 1].mid()), arb(Lm[4, 4].mid())
    pa = arb(((Lm[2, 2] + Lm[3, 3]) / 2).mid())
    pb = arb(((Lm[3, 2] - Lm[2, 3]) / 2).mid())
    Lam = arb_mat(5, 5)
    Lam[0, 0], Lam[1, 1], Lam[4, 4] = l1, l2, l5
    Lam[2, 2], Lam[3, 3], Lam[2, 3], Lam[3, 2] = pa, pa, -pb, pb
    E = Az - Lam
    zmax = [r, s2, s3, s3, s5]
    absE = [[arb(E[i, j].abs_upper()) for j in range(5)] for i in range(5)]
    res = {}
    for j, sj, lj in ((1, s2, l2), (4, s5, l5)):
        bound = (lj + absE[j][j]) * sj + sum((absE[j][k] * zmax[k] for k in range(5) if k != j), arb(0))
        res['inflow z%d' % (j + 1)] = bound
    e3 = sum((absE[2][k] * zmax[k] for k in range(5)), arb(0))
    e4 = sum((absE[3][k] * zmax[k] for k in range(5)), arb(0))
    res['inflow pair'] = pa * s3 * s3 + s3 * (e3 * e3 + e4 * e4).sqrt()
    D = [1, -1, -1, -1, -1]
    H = [[D[i] * Az[i, j] + D[j] * Az[j, i] for j in range(5)] for i in range(5)]
    gersh = [H[i][i] - sum((arb(H[i][j].abs_upper()) for j in range(5) if j != i), arb(0)) for i in range(5)]
    ok = all(v < 0 for v in res.values()) and all(g > 0 for g in gersh)
    # exit set: y* + T^-1 (r e1 + z'), z' in the stable box (the disc z3^2 + z4^2 <= s3^2 is inside the square)
    return dict(ok=bool(ok), inflow=res, gersh=gersh, T=Tf, Ti=Ti, Lam=(l1, l2, pa, pb, l5), Emax=max(
        float(absE[i][j].mid()) for i in range(5) for j in range(5)), r=r, s=(s2, s3, s5), y=y)


def main():
    T = temperature(sys.argv[1]) if len(sys.argv) > 1 else '18.5'
    Klo, Khi = (arb('10.4383548'), arb('10.4383549')) if float(T) == 18.5 else (arb('4.5107697'), arb('4.5107698'))
    K = Klo.union(Khi)
    phi = phi_of(T)
    y, EL = rest_state()
    print('T = %s C, phi = %s' % (T, phi.str(20)))
    print('E_l (zero resting current) = %s mV' % EL.str(25))
    print('rest: m = %s, n = %s, h = %s' % (y[2].str(20), y[3].str(20), y[4].str(20)))
    from hhseries import taylor
    f = [c[1] for c in taylor(y, K, phi, EL, 1)]
    print('f(rest) encloses 0:', all(v.contains(0) for v in f), ' radius', max(float(v.rad()) for v in f))
    a, b = float(K.mid()) * 0.9, float(K.mid()) * 1.2
    A = lemma_A(K, phi, EL, a, b)
    print('\nLemma A for every K in %s:' % K.str(12))
    print('  P(a) < 0 < P(b), P\' > 0 on [a, b] = [%.4f, %.4f] and the Hurwitz inequalities of Q: %s' % (a, b, A['ok']))
    print('  lambda_u in %s' % A['lam_u'].str(15))
    print('  Q = x^4 + a3 x^3 + a2 x^2 + a1 x + a0: a0..a3 = %s' % ', '.join(q.str(8) for q in A['Q']))
    print('  Hurwitz determinants D2 = %s, D3 = %s' % (A['D2'].str(8), A['D3'].str(8)))
    print('  so dim W^u(rest) = 1 and dim W^s(rest) = 4 for every K in the ball')
    # negative control: a bracket that does not contain lambda_u must fail
    bad = lemma_A(K, phi, EL, b, 2 * b)
    print('  negative control (bracket above lambda_u) rejected:', not bad['ok'])
    r = 1e-4
    B = lemma_B(K, phi, EL, r, (2e-9, 1e-7, 6e-9))
    print('\nLemma B, r = %g, (s2, s3, s5) = %s:' % (r, tuple(x.str(3) for x in B['s'])))
    for k, v in B['inflow'].items():
        print('  %s: bound %s (needs < 0)' % (k, v.str(5)))
    print('  cone: Gershgorin lower bounds %s (need > 0)' % ', '.join(g.str(5) for g in B['gersh']))
    print('  max |E_ij| = %.3e;  Lemma B holds: %s' % (B['Emax'], B['ok']))
    badB = lemma_B(K, phi, EL, r, (2e-11, 1e-9, 6e-11))
    print('  negative control (stable faces too thin) rejected:', not badB['ok'])
    ok = A['ok'] and B['ok'] and (not bad['ok']) and (not badB['ok'])
    print('\nALL CHECKS PASSED' if ok else '\nSOME CHECK FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
