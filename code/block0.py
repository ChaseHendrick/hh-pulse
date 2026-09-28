#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): the closing block B0 at rest, with a cone (Lyapunov-form) condition, for every K in a
ball.

Coordinates zeta = M (y - y*), M = S T, T a fixed matrix with exact dyadic entries (an approximate inverse of the
real eigenbasis of Df(y*) at the numerical pulse speed: unstable, fast real, complex pair (Re, Im), slow real, as in
certify_rest_wave.real_basis) and S = diag(WEIGHTS), exact integers. Block
    B0 = { |zeta_1| <= r, |zeta_s|_2 <= rho },   zeta_s = (zeta_2, ..., zeta_5),   r > rho,
    L(zeta) = zeta_1^2 - |zeta_s|^2,  D = diag(1, -1, -1, -1, -1).
Checked, for every x in B0 and every K in the ball, with A(x) = M Df(x) M^-1 (an interval matrix over a cell):
 (C) cone:      H = D A + A^T D is positive definite;
 (E) entrance:  lambda_max(sym A_ss) + |A_s1|_2 < 0 on the part { |zeta_1| <= rho } of B0.
(C) is checked by an interval Cholesky factorization of H with positive pivots (then every symmetric matrix in the
interval matrix is positive definite); (E) by bounding |A_s1|_2 by the Frobenius norm, mu, and checking that
-sym(A_ss) - mu I is positive definite the same way. B0 is covered by cells (boxes in zeta, intersected with the
4-ball only when they meet it, tested rigorously); a cell that fails is bisected, up to a depth limit.

Why these suffice. Since f(y*) = 0 and B0 is convex and
contains 0, zeta' = A-bar zeta with A-bar = int_0^1 M Df(y* + s M^-1 zeta) M^-1 ds, an average of matrices A(x) over x
on the segment [y*, y], which lies in B0 (and in { |zeta_1| <= rho } when |zeta_1| <= rho). The smallest eigenvalue
of D A + A^T D is a concave function of A, and lambda_max(sym A_ss) + |A_s1|_2 a convex one, so the bounds of (C) and
(E), which hold for every A(x), hold for their averages. Hence, while an orbit is in B0:
 * dL/dt = zeta^T (D A-bar + A-bar^T D) zeta > 0 for zeta != 0: L increases strictly;
 * at a point of the stable face |zeta_s| = rho with |zeta_1| <= rho (these are all the boundary points with
   L <= 0, since r > rho), d|zeta_s|^2/dt / 2 = zeta_s^T A-bar_ss zeta_s + zeta_s^T A-bar_s1 zeta_1
   <= (lambda_max(sym A-bar_ss) + |A-bar_s1|) rho^2 < 0: the orbit enters B0 strictly there;
 * K+ = {L > 0, zeta_1 > 0} and K- = {L > 0, zeta_1 < 0} cannot be left while the orbit stays in B0 (L stays
   positive, so zeta_1 cannot vanish);
 * an orbit that stays in B0 for all t >= t0 tends to y*: L is increasing and bounded on B0, so it has a limit, and
   the omega-limit set, which is invariant and in B0, lies in a level set of L; (C) allows that only at zeta = 0.
The flow on the face |zeta_1| = r is not checked and not needed.

Usage: python3 block0.py [T]   (writes data/block0_<T>.json and prints a summary; exit status 0 iff all checks,
including the negative control (a block too large must fail), behave as expected.)
"""
import itertools
import math
import json
import sys
import time
import numpy as np
from flint import arb, arb_mat, ctx
import certify_rest_wave as C
import hhjet6

ctx.prec = 128
WEIGHTS = (10, 7, 1, 1, 40)
RHO = {18.5: '0.8', 6.3: '0.6'}
R_OVER_RHO = '1.05'
CYCLE = (2, 0, 2, 3, 1, 2, 4)
K_REF = {18.5: 10.438354829417266, 6.3: 4.5107697268}


def ball(lo, hi):
    return arb(lo).union(arb(hi))


def exact(Mf):
    return arb_mat([[arb(float(v)) for v in row] for row in Mf])


def make_T(T, phi, EL, Kf):
    """T = the float inverse of the real eigenbasis of Df(y*) at K = Kf (numpy); only used once, to create the stored
    matrix: the proofs read T back exactly from data/closing_block_<T>.json, so they do not depend on LAPACK."""
    y, _ = C.rest_state()
    A0 = C.jac(y, arb(Kf), phi, EL)
    Am = np.array([[float(A0[i, j].mid()) for j in range(5)] for i in range(5)])
    V, _ = C.real_basis(Am)
    return np.linalg.inv(V)


def block_file(T):
    return '../data/closing_block_%s.json' % C.tag(T)


def load_block(T):
    """The stored block: T (exact dyadic entries, from hex floats), M = S T (exact), M^-1 (an enclosure), rho and r
    (exact binary floats)."""
    d = json.load(open(block_file(T)))
    Tm = arb_mat([[arb(float.fromhex(v)) for v in row] for row in d['T_hex']])
    S = arb_mat([[d['weights'][i] if i == j else 0 for j in range(5)] for i in range(5)])
    M = S * Tm
    return M, M.inv(), float.fromhex(d['rho_hex']), float.fromhex(d['r_hex']), d


def store_block(T, Kf, rho=None):
    phi = C.phi_of(T)
    _, EL = C.rest_state()
    Tf = make_T(T, phi, EL, Kf)
    rho = float(rho if rho is not None else RHO[float(T)])
    r = rho * float(R_OVER_RHO)
    d = {'T': float(T), 'K_ref': repr(Kf), 'weights': list(WEIGHTS), 'T_hex': [[float(v).hex() for v in row] for row in Tf],
         'T_matrix': [[float(v) for v in row] for row in Tf], 'rho_hex': rho.hex(), 'r_hex': r.hex(), 'rho': rho,
         'r': r, 'note': 'zeta = M (y - y*), M = diag(weights) T; B0 = {|zeta_1| <= r, |zeta_2..5|_2 <= rho}; '
                         'the entries of T, rho and r are exact binary floats (hex)'}
    json.dump(d, open(block_file(T), 'w'), indent=1)
    return d


def chol_margin(H):
    """Smallest lower bound of the pivots of an interval Cholesky factorization of the symmetric interval matrix H
    (a float; > 0 proves that every symmetric matrix in H is positive definite). A pivot whose lower bound is <= 0 is
    replaced by its upper bound so that the factorization can go on; the margin returned is then <= 0."""
    n = H.nrows()
    Lm = [[arb(0)] * n for _ in range(n)]
    margin = float('inf')
    for j in range(n):
        s = H[j, j] - sum((Lm[j][k] ** 2 for k in range(j)), arb(0))
        lo = float(s.lower())
        margin = min(margin, lo)
        if not s > 0:
            up_ = arb(s.upper())
            if not up_ > 0:
                return -float('inf')
            s = up_
        Lm[j][j] = s.sqrt()
        for i in range(j + 1, n):
            Lm[i][j] = (H[i, j] - sum((Lm[i][k] * Lm[j][k] for k in range(j)), arb(0))) / Lm[j][j]
    return margin


def chol_pd(H):
    """True if an interval Cholesky factorization of the symmetric interval matrix H has positive pivots (then every
    symmetric matrix in H is positive definite)."""
    n = H.nrows()
    Lm = [[arb(0)] * n for _ in range(n)]
    for j in range(n):
        s = H[j, j] - sum((Lm[j][k] ** 2 for k in range(j)), arb(0))
        if not s > 0:
            return False
        Lm[j][j] = s.sqrt()
        for i in range(j + 1, n):
            Lm[i][j] = (H[i, j] - sum((Lm[i][k] * Lm[j][k] for k in range(j)), arb(0))) / Lm[j][j]
    return True


def cell_matrix(zbox, M, Minv, ystar, K, phi, EL):
    """A = M Df(x) M^-1 over the y-box hull of y* + M^-1 zbox (all K in the ball)."""
    yb = [ystar[i] + sum((Minv[i, k] * zbox[k] for k in range(5)), arb(0)) for i in range(5)]
    J = hhjet6.jacobian(yb + [K], phi, EL)
    Df = arb_mat([[J[i][j] for j in range(5)] for i in range(5)])
    return M * Df * Minv


def cone_matrix(A):
    D = [1, -1, -1, -1, -1]
    return arb_mat([[D[i] * A[i, j] + D[j] * A[j, i] for j in range(5)] for i in range(5)])


def entrance_matrix(A):
    mu = sum((arb(A[i, 0].abs_upper()) ** 2 for i in range(1, 5)), arb(0)).sqrt()     # >= |A_s1|_F >= |A_s1|_2
    return arb_mat([[-(A[i, j] + A[j, i]) / 2 - (mu if i == j else 0) for j in range(1, 5)] for i in range(1, 5)])


def check_cone(A):
    return chol_pd(cone_matrix(A))


def check_entrance(A):
    return chol_pd(entrance_matrix(A))


def meets_ball(cell_s, rho):
    """Rigorous: False only if the 4-box cell_s (float endpoints, exact) is disjoint from the ball |zeta_s| <= rho."""
    d2 = arb(0)
    for a, b in cell_s:
        if a <= 0 <= b:
            continue
        d2 += arb(min(abs(a), abs(b))) ** 2
    return not (d2 > rho * rho)


def up(x):
    """A float (an exact dyadic) >= the ball x."""
    ub = arb(arb(x).upper())
    v = float(ub)
    while not arb(v) >= ub:
        v = math.nextafter(v, math.inf)
    return v


def cover_check(M, Minv, ystar, K, phi, EL, rho, r, which, n0=(2, 2, 4, 4, 2), maxdepth=8, log=None):
    """Check (C) (which='C', zeta_1 in [-r, r]) or (E) (which='E', zeta_1 in [-rho, rho]) on a cover of the block.
    Cells are boxes with float (exact dyadic) endpoints; the outer endpoints are rounded outward, so the cells cover
    the block. Returns (ok, number of cells checked, number of failed leaves)."""
    R1 = up(r) if which == 'C' else up(rho)
    Rs = up(rho)
    bound = [R1] + [Rs] * 4
    stack = []
    for idx in itertools.product(*[range(n) for n in n0]):
        cell = [(-bound[j] + 2 * bound[j] * idx[j] / n0[j], -bound[j] + 2 * bound[j] * (idx[j] + 1) / n0[j])
                for j in range(5)]
        cell = [(-bound[j] if idx[j] == 0 else a, bound[j] if idx[j] == n0[j] - 1 else b)
                for j, (a, b) in enumerate(cell)]
        stack.append((cell, 0))
    ncheck = nfail = 0
    test = check_cone if which == 'C' else check_entrance
    mat = cone_matrix if which == 'C' else entrance_matrix
    while stack:
        cell, depth = stack.pop()
        if not meets_ball(cell[1:], rho):
            continue
        zbox = [ball(a, b) for a, b in cell]
        ncheck += 1
        try:
            A = cell_matrix(zbox, M, Minv, ystar, K, phi, EL)
            ok = test(A)
        except (ArithmeticError, ZeroDivisionError, ValueError):
            ok = False
        if ok:
            continue
        if depth >= maxdepth:
            nfail += 1
            if log:
                log('   %s fails on cell %s at depth %d' % (which, cell, depth))
            if nfail > 20:
                return False, ncheck, nfail
            continue
        # bisect along a fixed cycle of coordinates (the Re coordinate of the complex pair, which drives u, most
        # often); the choice only affects efficiency, never the validity of the cover
        j = CYCLE[depth % len(CYCLE)]
        a, b = cell[j]
        m = (a + b) / 2                      # exact in binary floating point
        best = (None, (cell[:j] + [(a, m)] + cell[j + 1:], cell[:j] + [(m, b)] + cell[j + 1:]))
        for hc in best[1]:
            stack.append((hc, depth + 1))
    return nfail == 0, ncheck, nfail


def run(T, Kball, M, Minv, rho, r, log=print, n0=(1, 1, 2, 2, 1), maxdepth=20, phi=None):
    """Check (C) and (E) on B0 = {|zeta_1| <= r, |zeta_s| <= rho} for every K in Kball (and every phi in the ball phi,
    if given: a temperature interval, tstrip.py)."""
    phi = C.phi_of(T) if phi is None else phi
    ystar, EL = C.rest_state()
    rho, r = arb(rho), arb(r)
    t0 = time.time()
    okC, nC, fC = cover_check(M, Minv, ystar, Kball, phi, EL, rho, r, 'C', n0, maxdepth, log)
    t1 = time.time()
    okE, nE, fE = cover_check(M, Minv, ystar, Kball, phi, EL, rho, r, 'E', n0, maxdepth, log)
    t2 = time.time()
    return dict(ok=bool(okC and okE), cone=dict(ok=okC, cells=nC, failed=fC, secs=round(t1 - t0)),
                entrance=dict(ok=okE, cells=nE, failed=fE, secs=round(t2 - t1)))


def main():
    """Create the stored block if needed, then check it for the K ball of the proof (from data/hp_pulse_<T>.json,
    +- 1e-40, which contains the proof's [K1, K2]) and run the negative control (the same weights with radius 1.5
    times larger must fail)."""
    T = C.temperature(sys.argv[1]) if len(sys.argv) > 1 else '18.5'    # a decimal string: phi is exact for it
    import os
    ctx.prec = 128
    hp = '../data/hp_pulse_%s.json' % C.tag(T)
    if os.path.exists(hp):
        Ks = arb(json.load(open(hp))['K'])
        Kf = float(Ks.mid())
    else:
        Ks, Kf = arb(K_REF[float(T)]), K_REF[float(T)]
    if not os.path.exists(block_file(T)):
        store_block(T, Kf)
    M, Minv, rho, r, d = load_block(T)
    Kball = ball(Ks - arb('1e-40'), Ks + arb('1e-40'))
    out = ['T = %s C; zeta = M (y - y*), weights %s, rho = %r, r = %r; K ball %s' % (T, d['weights'], rho, r,
                                                                                      Kball.str(20))]
    res = run(T, Kball, M, Minv, rho, r, log=lambda s: None)
    out.append('block B0: cone condition (C) %s on %d cells (%d s); entrance condition (E) %s on %d cells (%d s)' % (
        'CERTIFIED' if res['cone']['ok'] else 'FAILED', res['cone']['cells'], res['cone']['secs'],
        'CERTIFIED' if res['entrance']['ok'] else 'FAILED', res['entrance']['cells'], res['entrance']['secs']))
    neg = run(T, Kball, M, Minv, 1.5 * rho, 1.5 * r, log=lambda s: None)
    out.append('negative control, radius x 1.5 (rho = %r): %s' % (1.5 * rho, 'rejected as expected' if not neg['ok']
                                                                   else 'CERTIFIED (BAD)'))
    ok = res['ok'] and not neg['ok']
    out.append('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    print('\n'.join(out))
    open('../data/block0_%s.txt' % C.tag(T), 'w').write('\n'.join(out) + '\n')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
