#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): (C) of the stability proof (the design note, Section 8.3, not part of this release),
the complex cone condition on the part of the region that every pulse of Theorem 1 visits after T_c, for every lambda
in the box [-eta, Lambda] x [0, Omega] (the lower half follows by complex conjugation: if M works for lambda, conj(M)
works for conj(lambda), since Df is real).

For a lambda-cell (a rectangle) and an exact complex matrix M (weights times a floating-point inverse eigenbasis of
A_inf at a point of the cell), checked: H = D At + At^* D is positive definite, At = M (Df(x) + lambda E) M^-1, for
every lambda in the cell and every x in B0 (a cover by cells) and in the path boxes of the record from T_c to T_enter
(stab_record.py; every pulse of Theorem 1 lies in B0 for xi >= T_c). Consequences (written in the REPORT, Section
8.3, (d) and (e)): for such lambda, (i) the rest point is in B0, so by the complex form of Lemma 0 of the paper
A_inf(lambda) has exactly one eigenvalue with positive real part, simple, and its eigenvector v_u satisfies
|(M v_u)_1| > |(M v_u)_s|; (ii) Q = |Z_1|^2 - |Z_s|^2 (Z = M Y) increases strictly along every nonzero solution of
Y' = A(xi, lambda) Y for xi >= T_c, so the solutions that decay at +infinity have Q < 0 at T_c: their values form a
4-dimensional subspace S of the open cone Q < 0, and the adjoint vector psi (psi^T Y = 0 on S) is, in the dual
coordinates M^-T psi, a multiple of (1, -g) with |g|_2 < 1; (iii) hence c(lambda) = v_u^T psi is not 0.

Output: ../data/stab_cone_<tag>.json with every cell, its weights and its matrix M (exact: hex floats), and the
negative control (B0 enlarged 1.5 times at lambda = 0 must fail).
Usage: HH_EL=10.613 python3 stab_cone.py 18.5
"""
import json
import sys
import time
import numpy as np
from flint import arb, acb, acb_mat, arb_mat, ctx
import certify_rest_wave as C
import stab_region as SR
import stab_large as SL

CFG = {18.5: {'eta': 0.1, 'Lambda': 14.0, 'Omega': 100.0,
              # initial cells: (x0, x1, y0, y1); split on failure
              'grid': [(-0.1, 0.5, 0, 0.5), (-0.1, 0.5, 0.5, 1), (-0.1, 0.5, 1, 2), (-0.1, 0.5, 2, 4),
                       (-0.1, 0.5, 4, 8), (-0.1, 0.5, 8, 16), (-0.1, 0.5, 16, 32), (-0.1, 0.5, 32, 64),
                       (-0.1, 0.5, 64, 100), (0.5, 2, 0, 2), (0.5, 2, 2, 8), (0.5, 2, 8, 32), (0.5, 2, 32, 100),
                       (2, 6, 0, 4), (2, 6, 4, 16), (2, 6, 16, 50), (2, 6, 50, 100), (6, 14, 0, 8), (6, 14, 8, 30),
                       (6, 14, 30, 100)]}}


def hexc(z):
    return [float(z.real).hex(), float(z.imag).hex()]


def cell_M(fm, cell, w0=None):
    N, M, Js = fm
    x0, x1, y0, y1 = cell
    lopt = complex(x0 + 0.25 * (x1 - x0), (y0 + y1) / 2)      # nearer the left edge, where the margin is smallest
    starts = N.STARTS if w0 is None else [np.log(w0[1:])] + N.STARTS
    m, w = N.best_weights(M, lopt, Js, starts)
    Mf = N.eig_coords(M, lopt, w)
    return m, w, Mf


def check_cell(R, cell, Mf, j_c):
    Mx, Mi = SL.coords_matrix(Mf)
    lam = SL.rect(*cell)
    res = SR.run_checks(R, SL.cone_test(R, Mx, Mi, lam), parts=('B0', 'record'), j_from=j_c,
                        time_pieces=lambda j: 4, max_time_depth=6, max_z_depth=12)
    return res


def float_model_B0(T):
    import stab_num as N
    import numpy as np
    M = N.Model(T)
    rec = json.load(open('../data/stab_record_%s.json' % C.tag(T)))
    # float profile near and after T_c: the collocation profile has t = 0 at the upstroke (u = 50 mV); the record has
    # t = 0 at the exit from the Lemma B box. The float samples only steer the weights; the rigorous check is separate.
    xs = np.linspace(6.0, 12.0, 400)
    Js = np.array([M.Df(y) for y in M.S(xs).T] + list(N.sample_B0(M, T, N=500)))
    return N, M, Js


def main():
    T = float(sys.argv[1])
    cfg = CFG[T]
    t0 = time.time()
    R = SR.Region(T)
    rec = json.load(open('../data/stab_record_%s.json' % C.tag(T)))
    j_c = rec['j_c']
    fm = float_model_B0(T)
    todo = list(cfg['grid'])
    done = []
    w = None
    while todo:
        cell = todo.pop(0)
        m, w, Mf = cell_M(fm, cell, w)
        res = check_cell(R, cell, Mf, j_c)
        print('(C) cell %s: %s (float margin %.3g; %s; %d s)' % (cell, res['ok'], m, res['stats'], res['secs']),
              flush=True)
        if res['ok']:
            done.append({'cell': cell, 'weights': [float(v) for v in w], 'float_margin': m, 'stats': res['stats'],
                         'M_hex': [[hexc(Mf[i, j]) for j in range(5)] for i in range(5)]})
            continue
        x0, x1, y0, y1 = cell
        if (x1 - x0) < 0.05 and (y1 - y0) < 0.05:
            print('(C) FAILED on a small cell %s: %s' % (cell, res['fails']))
            return False
        if (x1 - x0) >= (y1 - y0):
            xm = (x0 + x1) / 2
            todo[:0] = [(x0, xm, y0, y1), (xm, x1, y0, y1)]
        else:
            ym = (y0 + y1) / 2
            todo[:0] = [(x0, x1, y0, ym), (x0, x1, ym, y1)]
    # coverage of [-eta, Lambda] x [0, Omega]: the initial grid is a partition and cells are only split in two
    # negative control: B0 enlarged 1.5 times, at lambda = 0 with the cell matrix of the cell containing 0
    c0 = [d for d in done if d['cell'][0] <= 0 <= d['cell'][1] and d['cell'][2] <= 0 <= d['cell'][3]][0]
    Mf0 = np.array([[complex(float.fromhex(a), float.fromhex(b)) for a, b in row] for row in c0['M_hex']])
    R.rho, R.r = R.rho * arb('1.5'), R.r * arb('1.5')
    neg = check_cell(R, (0.0, 0.0, 0.0, 0.0), Mf0, j_c)
    R.rho, R.r = R.rho / arb('1.5'), R.r / arb('1.5')
    out = {'T': T, 'tag': C.tag(T), 'eta': cfg['eta'], 'Lambda': cfg['Lambda'], 'Omega': cfg['Omega'], 'j_c': j_c,
           'cells': done, 'n_cells': len(done), 'negative_control_B0_x1.5_at_0': {'rejected': not neg['ok'],
                                                                                  'fails': str(neg['fails'])[:300]},
           'secs': round(time.time() - t0)}
    out['verdict'] = 'PASS' if not neg['ok'] else 'FAIL'
    json.dump(out, open('../data/stab_cone_%s.json' % C.tag(T), 'w'), indent=1)
    print('(C) %d cells cover [%s, %s] x [0, %s]; negative control (B0 x 1.5 at 0) rejected: %s; %d s' % (
        len(done), -cfg['eta'], cfg['Lambda'], cfg['Omega'], not neg['ok'], out['secs']))
    return out['verdict'] == 'PASS'


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
