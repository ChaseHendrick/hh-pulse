#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic) with written arguments: (L1), (L2a), (L2b) of the stability proof, no eigenvalue of the
linearization about the pulse outside the box [-eta, Lambda] x [-Omega, Omega] in Re lambda >= -eta
(the design note, Section 8.3, not part of this release). Every check runs over the region of stab_region.py,
which contains the whole orbit of every pulse of Theorem 1 and the rest point.

Eigenvalue problem (PDE form, moving frame, eigenfunction bounded, hence for Re lambda > -0.448 (E) exponentially
decaying with its derivatives):
    (1/K) p'' - p' - a p - b.q = lambda p,     -q' + c p - diag(kappa) q = lambda q.
First-order form: Y = (p, p', q), Y' = A(xi, lambda) Y, A = Df(x(xi), K) + lambda E.

(L1) Let s = (s_1, s_2, s_3) > 0 and qt_i = s_i q_i, Y = (p, qt). Then lambda Y = (1/K)(p'', 0) - Y' + Z(xi) Y with
     Z = [[-a, -b_i/s_i], [s_i c_i, -diag(kappa)]] (real 4 x 4). Taking the L^2 inner product with Y, and using
     Re <Y', Y> = 0 and <p'', p> = -||p'||^2, Re lambda ||Y||^2 = -(1/K) ||p'||^2 + <sym(Z) Y, Y> <= Lambda ||Y||^2 if
     Lambda I - sym(Z(x)) is positive definite for every state x of the region. Checked by interval Cholesky.
(L2a) For a lambda-cell (a complex rectangle) and an invertible complex matrix M (exact entries: weights times a
     floating-point inverse eigenbasis of A_inf at the centre), if H = D At + At^* D is positive definite for
     At = M (Df(x) + lambda E) M^-1, every x in the region and every lambda in the cell (D = diag(1, -1, -1, -1, -1)),
     then no lambda in the cell is an eigenvalue. Proof: Z = M Y, Q = Z^* D Z has Q' = Z^* H Z > 0 along any nonzero
     solution. At the rest point (in the region) the complex form of Lemma 0 of the paper gives A_inf(lambda) exactly
     one eigenvalue nu with Re nu > 0, simple, with eigenvector v, Q(v) > 0. The solution phi^- that decays at
     -infinity satisfies e^{-nu xi} phi^-(xi) -> v, so Q(phi^-(xi)) > 0 for xi near -infinity, hence for all xi, and
     Q increases: |Z_1(xi)|^2 >= Q(phi^-(xi_0)) > 0 for xi >= xi_0, so phi^- does not decay at +infinity. An
     eigenfunction decays at both ends, and those decaying at -infinity are multiples of phi^- (A_inf has one unstable
     eigenvalue), so lambda is not an eigenvalue. Hermitian positive definiteness is checked by interval Cholesky of
     the real symmetric 10 x 10 matrix [[Re H, -Im H], [Im H, Re H]].
(L2b) For |Im lambda| >= Omega_big and Re lambda >= -eta: coordinates z+ = (w - nu_- p)/(2R), z- = (nu_+ p - w)/(2R),
     qt_i = w_i q_i with R^2 = K^2/4 + K(lambda + a0), a0 = a(y*), Re R > 0, nu_+- = K/2 +- R. In them (derivation in
     the REPORT) A = diag(nu_+ + g, nu_- - g, -(lambda + kappa_i)) plus off-diagonal entries g (1,2), -g (2,1),
     beta_i/w_i (1,i), -beta_i/w_i (2,i), w_i c_i (i,1), (i,2), with g = K (a - a0)/(2R), beta_i = K b_i/(2R); so
     H11 = K + 2 Re R + 2 Re g, H22 = -K + 2 Re R + 2 Re g, H12 = 2 Re g, H1i = H2i = e_i = beta_i/w_i - w_i c_i,
     Hii = 2(Re lambda + kappa_i), other entries 0. By the Schur complement, H is positive definite if Hii > 0 and
     H22 > |H12| + 2 sum |e_i|^2/Hii (then also H11 = H22 + 2K works). With w_i^2 = K B_i/(2|R| C_i) this holds if
         Re R > K/2 + (K/|R|) (sup|a - a0| + sum_i B_i C_i/(kappa_lo_i - eta)),
     B_i = sup|b_i|, C_i = sup|c_i|, kappa_lo_i = inf kappa_i over the region. Since Re R^2 = K^2/4 + K(a0 + Re lambda)
     > 0, Re R >= |R|/sqrt(2), and |R|^2 >= K |Im lambda|; f(r) = r/sqrt(2) - K/2 - K C*/r increases, so
     f(sqrt(K Omega_big)) > 0 proves the cone condition, hence (as in (L2a)) no eigenvalue, for all such lambda.
Negative controls: (L1) with Lambda = 12 must fail; (L2a) on a cell at 40 i (where the condition fails numerically) must
fail; (L2b) must fail at Omega_big / 4.

Usage: HH_EL=10.613 python3 stab_large.py 18.5 L1 | L2b | L2a    -> ../data/stab_large_<tag>_<part>.json
"""
import json
import math
import sys
import time
import numpy as np
from flint import arb, acb, arb_mat, acb_mat, ctx
import certify_rest_wave as C
import block0
import stab_region as SR

CFG = {18.5: {'s': ['84.46898832', '194.99204336', '193.32096089'], 'Lambda': '14', 'Lambda_neg': '12',
              'eta': '0.1', 'Omega': 100, 'Omega_big': 2600, 'L2a_dy': 25}}


def ball(lo, hi):
    return arb(lo).union(arb(hi))


def time_pieces_default(R):
    def f(j):
        h = float(R.recs[j][1].mid())
        return max(1, min(64, int(math.ceil(h / 0.02))))
    return f


# ---------------------------------------------------------------------------------------------------------- (L1)
def L1_test(R, s, Lam):
    def test(yb):
        a, b, c, kap = SR.zeroth_order(yb, R.phi, R.EL)
        Z = [[None] * 4 for _ in range(4)]
        Z[0][0] = -a
        for i in range(3):
            Z[0][i + 1] = -b[i] / s[i]
            Z[i + 1][0] = s[i] * c[i]
            for k in range(3):
                Z[i + 1][k + 1] = -kap[i] if i == k else arb(0)
        H = arb_mat([[(Lam if i == j else 0) - (Z[i][j] + Z[j][i]) / 2 for j in range(4)] for i in range(4)])
        return block0.chol_pd(H)
    return test


def run_L1(T):
    cfg = CFG[T]
    R = SR.Region(T)
    s = [arb(v) for v in cfg['s']]
    out = {'part': 'L1', 'T': T, 's': cfg['s'], 'Lambda': cfg['Lambda']}
    res = SR.run_checks(R, L1_test(R, s, arb(cfg['Lambda'])), time_pieces=time_pieces_default(R),
                        log=lambda m: print(m, flush=True))
    neg = SR.run_checks(R, L1_test(R, s, arb(cfg['Lambda_neg'])), time_pieces=time_pieces_default(R))
    out.update(ok=res['ok'], stats=res['stats'], secs=res['secs'], fails=str(res['fails'])[:300],
               negative_control={'Lambda': cfg['Lambda_neg'], 'rejected': not neg['ok'], 'fails': str(neg['fails'])[:300]})
    out['verdict'] = 'PASS' if res['ok'] and not neg['ok'] else 'FAIL'
    print('(L1) Re lambda <= %s for every eigenvalue: %s (%s boxes, %d s); negative control Lambda = %s rejected: %s' % (
        cfg['Lambda'], res['ok'], res['stats'], res['secs'], cfg['Lambda_neg'], not neg['ok']))
    json.dump(out, open('../data/stab_large_%s_L1.json' % C.tag(T), 'w'), indent=1)
    return out['verdict'] == 'PASS'


# ---------------------------------------------------------------------------------------------------------- (L2b)
def sup_bounds(R):
    """sup|a - a0|, B_i = sup|b_i|, C_i = sup|c_i|, inf kappa_i over the region (upper/lower bounds)."""
    a0, _, _, _ = SR.zeroth_order(R.ystar, R.phi, R.EL)
    acc = {'da': arb(0), 'B': [arb(0)] * 3, 'C': [arb(0)] * 3, 'klo': [arb(1e9)] * 3}

    def test(yb):
        a, b, c, kap = SR.zeroth_order(yb, R.phi, R.EL)
        acc['da'] = acc['da'].max(arb((a - a0).abs_upper()))
        for i in range(3):
            acc['B'][i] = acc['B'][i].max(arb(b[i].abs_upper()))
            acc['C'][i] = acc['C'][i].max(arb(c[i].abs_upper()))
            acc['klo'][i] = acc['klo'][i].min(arb(kap[i].lower()))
        return True
    SR.run_checks(R, test, time_pieces=lambda j: 1)
    return a0, acc


def run_L2b(T):
    cfg = CFG[T]
    R = SR.Region(T)
    a0, acc = sup_bounds(R)
    eta = arb(cfg['eta'])
    K = R.Kball
    Cstar = acc['da'] + sum((acc['B'][i] * acc['C'][i] / (acc['klo'][i] - eta) for i in range(3)), arb(0))
    ok_k = all(bool(acc['klo'][i] - eta > 0) for i in range(3)) and bool(a0 - eta > 0)

    def f(Om):
        r = (arb(K.lower()) * Om).sqrt()
        return r / arb(2).sqrt() - K / 2 - K * Cstar / r
    val = f(arb(cfg['Omega_big']))
    neg = f(arb(cfg['Omega_big']) / 4)
    ok = ok_k and bool(val > 0) and not bool(neg > 0)
    out = {'part': 'L2b', 'T': T, 'Omega_big': cfg['Omega_big'], 'a0': a0.str(10), 'sup|a-a0|': acc['da'].str(8),
           'B': [x.str(8) for x in acc['B']], 'C': [x.str(8) for x in acc['C']], 'kappa_lo': [x.str(8) for x in acc['klo']],
           'Cstar': Cstar.str(8), 'f(sqrt(K Omega_big))': val.str(8), 'negative control f at Omega_big/4': neg.str(8),
           'verdict': 'PASS' if ok else 'FAIL'}
    print(json.dumps(out, indent=1))
    json.dump(out, open('../data/stab_large_%s_L2b.json' % C.tag(T), 'w'), indent=1)
    return ok


# ---------------------------------------------------------------------------------------------------------- (L2a)
def exact_c(z):
    return acb(float(z.real), float(z.imag))


def coords_matrix(Mf):
    """exact complex matrix from floats, and an enclosure of its inverse."""
    M = acb_mat([[exact_c(Mf[i, j]) for j in range(5)] for i in range(5)])
    return M, M.inv()


def hermitian_pd(H):
    """H: acb_mat enclosing a Hermitian matrix (entries computed for all i, j). Interval Cholesky of the real 10 x 10
    embedding [[Re H, -Im H], [Im H, Re H]]; the diagonal of H is taken real."""
    n = H.nrows()
    Rm = arb_mat(2 * n, 2 * n)
    for i in range(n):
        for j in range(n):
            re, im = H[i, j].real, H[i, j].imag
            if i == j:
                im = arb(0)
            Rm[i, j] = re
            Rm[i + n, j + n] = re
            Rm[i + n, j] = im
            Rm[i, j + n] = -im
    return block0.chol_pd(Rm)


DVEC = [1, -1, -1, -1, -1]


def cone_H(At):
    n = 5
    return acb_mat([[DVEC[i] * At[i, j] + At[j, i].conjugate() * DVEC[j] for j in range(n)] for i in range(n)])


def E_matrix(K):
    E = arb_mat(5, 5)
    E[1, 0] = K
    E[2, 2] = E[3, 3] = E[4, 4] = -1
    return acb_mat(E)


def cone_test(R, M, Minv, lam_cell):
    ME = M * E_matrix(R.Kball) * Minv * lam_cell

    def test(yb):
        J = SR.jac5(yb, R.Kball, R.phi, R.EL)
        At = M * acb_mat(arb_mat(J)) * Minv + ME
        return hermitian_pd(cone_H(At))
    return test


def rect(x0, x1, y0, y1):
    xc, yc = (arb(x0) + arb(x1)) / 2, (arb(y0) + arb(y1)) / 2
    return acb(xc + arb(0, (arb(x1) - arb(x0)) / 2), yc + arb(0, (arb(y1) - arb(y0)) / 2))


def float_model(T):
    import stab_num as N
    M = N.Model(T)
    xs = np.concatenate([np.linspace(M.XL, 1.5, 3000), np.linspace(1.5, M.XR, 1500)])
    Js = np.array([M.Df(y) for y in M.S(xs).T] + list(N.sample_B0(M, T, N=600)))
    return N, M, Js


def cell_check(T, R, cell, fm, w0=None, log=print):
    """(L2a) on the rectangle cell = (x0, x1, y0, y1): optimise weights numerically at the centre (float profile and
    B0 samples), then check rigorously over the region. Returns (ok, info)."""
    N, M, Js = fm
    x0, x1, y0, y1 = cell
    lc = complex(x0 + 0.25 * (x1 - x0), (y0 + y1) / 2)      # nearer the left edge, where the margin is smallest
    starts = N.STARTS if w0 is None else [np.log(w0[1:])] + N.STARTS[:2]
    m, w = N.best_weights(M, lc, Js, starts)
    Mf = N.eig_coords(M, lc, w)
    Mx, Mi = coords_matrix(Mf)
    lam = rect(x0, x1, y0, y1)
    res = SR.run_checks(R, cone_test(R, Mx, Mi, lam), time_pieces=time_pieces_default(R), max_time_depth=6,
                        max_z_depth=10)
    return res['ok'], {'cell': cell, 'float_margin': m, 'weights': [float(v) for v in w], 'stats': res['stats'],
                       'secs': res['secs'], 'fails': str(res['fails'])[:200]}, w


def run_L2a(T):
    cfg = CFG[T]
    R = SR.Region(T)
    fm = float_model(T)
    eta, Lam = -float(cfg['eta']), float(cfg['Lambda'])
    y, Y = float(cfg['Omega']), float(cfg['Omega_big'])
    todo = []
    dy = float(cfg['L2a_dy'])
    while y < Y:
        y1 = min(Y, y + dy)
        for xa, xb in ((eta, 2.0), (2.0, 6.0), (6.0, Lam)):
            todo.append((xa, xb, y, y1))
        y, dy = y1, dy * 1.25
    done, w = [], None
    t0 = time.time()
    while todo:
        cell = todo.pop(0)
        ok, info, w = cell_check(T, R, cell, fm, w)
        print('(L2a) cell %s: %s, float margin %.3g, %s, %d s' % (cell, ok, info['float_margin'], info['stats'],
                                                                info['secs']), flush=True)
        if ok:
            done.append(info)
            continue
        x0, x1, y0, y1 = cell
        if (y1 - y0) < 0.5 and (x1 - x0) < 0.5:
            print('(L2a) FAILED on a small cell %s' % (cell,))
            return False
        if (x1 - x0) >= 0.2 * (y1 - y0):
            xm = (x0 + x1) / 2
            todo[:0] = [(x0, xm, y0, y1), (xm, x1, y0, y1)]
        else:
            ym = (y0 + y1) / 2
            todo[:0] = [(x0, x1, y0, ym), (x0, x1, ym, y1)]
    # negative control: a cell around 40 i, where the condition fails numerically
    okn, infon, _ = cell_check(T, R, (-0.1, 0.1, 39.9, 40.1), fm)
    out = {'part': 'L2a', 'T': T, 'cells': done, 'n_cells': len(done), 'secs': round(time.time() - t0),
           'negative_control_40i': {'rejected': not okn, 'info': infon}}
    out['verdict'] = 'PASS' if not okn else 'FAIL'
    print('(L2a) %d cells cover [%s, %s] x [%s, %s]; negative control at 40 i rejected: %s' % (
        len(done), eta, Lam, cfg['Omega'], cfg['Omega_big'], not okn))
    json.dump(out, open('../data/stab_large_%s_L2a.json' % C.tag(T), 'w'), indent=1)
    return out['verdict'] == 'PASS'


if __name__ == '__main__':
    T = float(sys.argv[1])
    part = sys.argv[2]
    fn = {'L1': run_L1, 'L2a': run_L2a, 'L2b': run_L2b}[part]
    sys.exit(0 if fn(T) else 1)
