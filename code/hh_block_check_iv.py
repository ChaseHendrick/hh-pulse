#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous re-check of the closing block (block0.py) by a separate implementation: mpmath interval arithmetic
(mpmath.iv) instead of Arb, the Jacobian from hand-derived formulas instead of the Taylor jets of hhjet6.py, M^-1
computed exactly in rational arithmetic, its own cover of the block and its own interval Cholesky test.

It reads only data/closing_block_<T>.json (T, the weights, rho, r, all exact binary floats) and
data/pulse_proof_<T>_config.json (K1, K2) and checks, for every x in B0 = {|zeta_1| <= r, |zeta_s|_2 <= rho},
zeta = M (y - y*), M = diag(weights) T, and every K in [K1, K2]:
 (C) D A + A^T D positive definite, A = M Df(x) M^-1, D = diag(1, -1, -1, -1, -1);
 (E) for |zeta_1| <= rho: -sym(A_ss) - mu I positive definite, mu = the Frobenius norm of A_s1 (upper bound).
Exit status 0 iff both hold and the negative control (radius x 1.5) fails.

Jacobian of f = (w, K (w + I), phi (a_x (1 - x) - b_x x) for x = m, n, h), I = 120 m^3 h (u - 115) +
36 n^4 (u + 12) + 0.3 (u - E_l):
  dI/du = 120 m^3 h + 36 n^4 + 0.3, dI/dm = 360 m^2 h (u - 115), dI/dn = 144 n^3 (u + 12), dI/dh = 120 m^3 (u - 115);
  a_m = psi((25 - u)/10), a_m' = -psi'((25 - u)/10)/10; b_m = 4 e^(-u/18), b_m' = -b_m/18;
  a_n = psi((10 - u)/10)/10, a_n' = -psi'((10 - u)/10)/100; b_n = e^(-u/80)/8, b_n' = -b_n/80;
  a_h = 0.07 e^(-u/20), a_h' = -a_h/20; b_h = 1/(e^((30 - u)/10) + 1), b_h' = e^((30 - u)/10)/(10 (e^((30 - u)/10) + 1)^2);
  psi(x) = x/(e^x - 1), psi'(x) = (e^x - 1 - x e^x)/(e^x - 1)^2 (x stays in [0.8, 2.7] on the block: no singularity).
"""
import json
import os
import re
import sys
import time
from fractions import Fraction
from mpmath import iv, mp

iv.prec = 113
mp.prec = 113


def I(lo, hi=None):
    return iv.mpf([lo, lo if hi is None else hi])


def frac_iv(q):
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)


def dec_iv(s):
    """An interval containing the decimal number s (outward rounding by mpmath's interval parser)."""
    return iv.mpf(s)


def psi(x):
    e = iv.exp(x)
    return x / (e - 1)


def dpsi(x):
    e = iv.exp(x)
    return (e - 1 - x * e) / (e - 1) ** 2


def rates(u):
    am = psi((25 - u) / 10)
    dam = -dpsi((25 - u) / 10) / 10
    bm = 4 * iv.exp(-u / 18)
    dbm = -bm / 18
    an = psi((10 - u) / 10) / 10
    dan = -dpsi((10 - u) / 10) / 100
    bn = iv.exp(-u / 80) / 8
    dbn = -bn / 80
    ah = iv.mpf('0.07') * iv.exp(-u / 20)
    dah = -ah / 20
    E = iv.exp((30 - u) / 10)
    bh = 1 / (E + 1)
    dbh = E / (10 * (E + 1) ** 2)
    return am, dam, bm, dbm, an, dan, bn, dbn, ah, dah, bh, dbh


def jac(y, K, phi):
    u, w, m, n, h = y
    am, dam, bm, dbm, an, dan, bn, dbn, ah, dah, bh, dbh = rates(u)
    Iu = 120 * m ** 3 * h + 36 * n ** 4 + iv.mpf('0.3')
    Im = 360 * m ** 2 * h * (u - 115)
    In = 144 * n ** 3 * (u + 12)
    Ih = 120 * m ** 3 * (u - 115)
    z = iv.mpf(0)
    return [[z, iv.mpf(1), z, z, z],
            [K * Iu, K, K * Im, K * In, K * Ih],
            [phi * (dam * (1 - m) - dbm * m), z, -phi * (am + bm), z, z],
            [phi * (dan * (1 - n) - dbn * n), z, z, -phi * (an + bn), z],
            [phi * (dah * (1 - h) - dbh * h), z, z, z, -phi * (ah + bh)]]


HH_EL = os.environ.get('HH_EL')
DECIMAL = re.compile(r'^[0-9]+(\.[0-9]+)?$')
# The rest value u* of certify_rest_wave.rest_state for HH_EL = 10.613 (manuscript, Section 2): its interval Newton step
# proves that the resting current has exactly one zero within 1e-3 mV of the float start 0.0036206688079..., so a zero
# found here within that window is the same rest state. For other HH_EL values uniqueness is not checked here.
U_STAR = {'10.613': '0.0036206688079425688'}


def gates(u):
    am, _, bm, _, an, _, bn, _, ah, _, bh, _ = rates(u)
    return am / (am + bm), an / (an + bn), ah / (ah + bh)


def rest():
    """Rest: u = 0 for the zero-current leak potential; with HH_EL set, the zero of the steady-state current
    g(u) = I(u, m_inf, n_inf, h_inf) enclosed by bisection with interval evaluations of g (own code, no derivatives):
    g(a) < 0 < g(b) at the ends of the final interval, so it contains a zero. Bisection alone does not show that the
    zero is unique; for HH_EL = 10.613 the final interval is checked to lie within 1e-3 mV (less a margin of 1e-9) of
    u* (U_STAR), where certify_rest_wave.rest_state proves by an interval Newton step that the zero is unique, so this
    is the rest state of the proof."""
    if not HH_EL:
        m, n, h = gates(iv.mpf(0))
        return [iv.mpf(0), iv.mpf(0), m, n, h]
    EL = iv.mpf(HH_EL)

    def g(u):
        m, n, h = gates(u)
        return 120 * m ** 3 * h * (u - 115) + 36 * n ** 4 * (u + 12) + iv.mpf('0.3') * (u - EL)
    a, b = mp.mpf('-0.5'), mp.mpf('0.5')
    if not (g(iv.mpf(a)).b < 0 < g(iv.mpf(b)).a):
        raise ArithmeticError('no sign change for the rest state')
    for _ in range(300):
        c = (a + b) / 2
        gc = g(iv.mpf(c))
        if gc.b < 0:
            a = c
        elif gc.a > 0:
            b = c
        else:
            break
    u = iv.mpf([a, b])
    if HH_EL in U_STAR:
        c = mp.mpf(U_STAR[HH_EL])
        if not (a > c - mp.mpf('0.000999999999') and b < c + mp.mpf('0.000999999999')):
            raise ArithmeticError('the zero found is not in the window where the proof shows it is unique')
    m, n, h = gates(u)
    return [u, iv.mpf(0), m, n, h]


def matmul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(len(B))), iv.mpf(0)) for j in range(len(B[0]))]
            for i in range(len(A))]


def chol_pd(H):
    n = len(H)
    L = [[iv.mpf(0)] * n for _ in range(n)]
    for j in range(n):
        s = H[j][j] - sum((L[j][k] ** 2 for k in range(j)), iv.mpf(0))
        if not s.a > 0:
            return False
        L[j][j] = iv.sqrt(s)
        for i in range(j + 1, n):
            L[i][j] = (H[i][j] - sum((L[i][k] * L[j][k] for k in range(j)), iv.mpf(0))) / L[j][j]
    return True


def rational_inverse(M):
    n = len(M)
    A = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [row[n:] for row in A]


def check(Mi, Mii, ystar, K, phi, rho, r, which, maxdepth=24):
    """Cover the block by boxes (bisect the widest coordinate relative to its block extent) and test each."""
    R1 = r if which == 'C' else rho
    ext = [R1, rho, rho, rho, rho]
    stack = [([(-R1, R1)] + [(-rho, rho)] * 4, 0)]
    n = 0
    while stack:
        cell, d = stack.pop()
        # skip cells disjoint from the 4-ball |zeta_s| <= rho (rational arithmetic, exact)
        d2 = sum((min(abs(a), abs(b)) ** 2 if not (a <= 0 <= b) else Fraction(0)) for a, b in cell[1:])
        if d2 > rho * rho:
            continue
        n += 1
        zb = [I(float(a), float(b)) if Fraction(float(a)) == a and Fraction(float(b)) == b else
              iv.mpf([frac_iv(a).a, frac_iv(b).b]) for a, b in cell]
        yb = [ystar[i] + sum((Mii[i][k] * zb[k] for k in range(5)), iv.mpf(0)) for i in range(5)]
        A = matmul(matmul(Mi, jac(yb, K, phi)), Mii)
        if which == 'C':
            D = [1, -1, -1, -1, -1]
            H = [[D[i] * A[i][j] + D[j] * A[j][i] for j in range(5)] for i in range(5)]
        else:
            mu = iv.sqrt(sum((A[i][0] ** 2 for i in range(1, 5)), iv.mpf(0))).b
            H = [[-(A[i][j] + A[j][i]) / 2 - (mu if i == j else 0) for j in range(1, 5)] for i in range(1, 5)]
        if chol_pd(H):
            continue
        if d >= maxdepth:
            return False, n
        j = max(range(5), key=lambda k: (cell[k][1] - cell[k][0]) / ext[k] * (3 if k == 2 else 1))
        a, b = cell[j]
        m = (a + b) / 2
        stack.append((cell[:j] + [(a, m)] + cell[j + 1:], d + 1))
        stack.append((cell[:j] + [(m, b)] + cell[j + 1:], d + 1))
    return True, n


def phi_iv(T):
    """3^((T - 6.3)/10) for the decimal string T (iv.mpf of a decimal string encloses the decimal number)."""
    if not isinstance(T, str) or not DECIMAL.match(T):
        raise ValueError('the temperature must be a decimal string such as 18.5 or 6.3')
    return iv.exp((iv.mpf(T) - iv.mpf('6.3')) / 10 * iv.log(3))


def main():
    T = sys.argv[1] if len(sys.argv) > 1 else '18.5'       # the decimal string, as typed (6.3 is not a float here)
    if not DECIMAL.match(T):
        raise SystemExit('the temperature must be a decimal number such as 18.5 or 6.3')
    tg = T + ('_El%s' % HH_EL if HH_EL else '')
    blk = json.load(open('../data/closing_block_%s.json' % tg))
    cfg = json.load(open('../data/pulse_proof_%s_config.json' % tg))
    Tm = [[Fraction(float.fromhex(v)) for v in row] for row in blk['T_hex']]
    Mq = [[blk['weights'][i] * Tm[i][j] for j in range(5)] for i in range(5)]
    Mqi = rational_inverse(Mq)
    Mi = [[frac_iv(v) for v in row] for row in Mq]
    Mii = [[frac_iv(v) for v in row] for row in Mqi]
    rho = Fraction(float.fromhex(blk['rho_hex']))
    r = Fraction(float.fromhex(blk['r_hex']))
    K = iv.mpf([dec_iv(cfg['K1_dec'][:45]).a, dec_iv(cfg['K2_dec'][:45]).b])
    # the 45-digit truncations bracket K1 and K2 up to 1e-43; widen by 1e-40 to contain [K1, K2] with margin
    K = iv.mpf([K.a - iv.mpf('1e-40').b, K.b + iv.mpf('1e-40').b])
    phi = phi_iv(T)
    ystar = rest()
    lines = ['independent re-check of the closing block at T = %s C (mpmath.iv, %d bits)' % (T, iv.prec),
             'rho = %s, r = %s, weights %s, K in %s' % (float(rho), float(r), blk['weights'], K)]
    t0 = time.time()
    okC, nC = check(Mi, Mii, ystar, K, phi, rho, r, 'C')
    okE, nE = check(Mi, Mii, ystar, K, phi, rho, r, 'E')
    lines.append('(C) cone condition: %s (%d cells); (E) entrance condition: %s (%d cells); %.0f s' % (
        okC, nC, okE, nE, time.time() - t0))
    # the negative control uses the same depth limit as the check itself, so its failure is not a depth artefact
    b1, _ = check(Mi, Mii, ystar, K, phi, rho * Fraction(3, 2), r * Fraction(3, 2), 'C')
    b2, _ = check(Mi, Mii, ystar, K, phi, rho * Fraction(3, 2), r * Fraction(3, 2), 'E')
    lines.append('negative control, radius x 1.5: %s' % ('rejected as expected' if not (b1 and b2) else 'CERTIFIED (BAD)'))
    ok = okC and okE and not (b1 and b2)
    lines.append('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    print('\n'.join(lines))
    open('../data/block_check_iv_%s.txt' % tg, 'w').write('\n'.join(lines) + '\n')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
