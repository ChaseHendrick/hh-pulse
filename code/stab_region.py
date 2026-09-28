#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): the region visited by every pulse of Theorem 1 (K* in [K1, K2], leaving rest through the
exit set E at xi = 0, in B0 from T_enter on), as a finite union of boxes, and a driver that applies a test to every box,
subdividing a box whose test fails (the stability proof, REPORT.md Section 8.3).

The three parts, for xi <= 0, 0 <= xi <= T_enter and xi >= T_enter:
 * the Lemma B box: y* + T_B^-1 [r_B-box] (its interval hull); every pulse lies in it for xi <= 0 (Lemma 1 of the paper:
   the orbit is in B until it leaves through E at xi = 0), and it contains y*;
 * the record of stab_record.py: for the step [t_j, t_j + h_j] and a piece [tau_a, tau_b] of [0, h_j], the box
   sum_{k <= p} x_k(X_j) [tau_a, tau_b]^k + [0, tau_b^(p+1)] x_{p+1}(W_j), intersected with W_j (Taylor's theorem with
   the Lagrange remainder, Lemma A.2 of the paper: x_k(X_j) encloses the Taylor coefficients of every solution from the
   hull X_j, and the solutions stay in W_j on the step);
 * B0 = {|zeta_1| <= r, |zeta_s| <= rho}, zeta = M (y - y*): covered by cells (boxes in zeta, those that meet the
   4-ball, as in block0.cover_check), each mapped to the interval hull of y* + M^-1 cell.
Every test is a function of a y-box (5 balls) that returns True only if a property is proved for every state in it; a
failing piece is bisected (in time for the record, in zeta for B0) up to a depth limit, and a failure at the limit
fails the whole check. So a passing check proves the property along the whole orbit of every pulse of Theorem 1.
"""
import os
import pickle
import time
from flint import arb, ctx
import certify_rest_wave as C
import hhjet6
import lohner6 as L
import block0

P_PATH = 20            # Taylor order of the path boxes


def ball(lo, hi):
    return arb(lo).union(arb(hi))


class Region:
    def __init__(self, T, prec=128):
        import prove_pulse as PP
        ctx.prec = PP.PREC
        S = PP.Setup(T)
        self.S = S
        B = S.lemma_B()
        assert B['ok']
        Ti = B['Ti']
        s2, s3, s5 = B['s']
        zb = [ball(-S.rB, S.rB), ball(-s2, s2), ball(-s3, s3), ball(-s3, s3), ball(-s5, s5)]
        self.boxB = [S.ystar[i] + sum((Ti[i, k] * zb[k] for k in range(5)), arb(0)) for i in range(5)]
        d = pickle.load(open('../data/logs/stab_record_%s.pkl' % C.tag(T), 'rb'))
        self.recs = [(L.deser(a), L.deser(b), [L.deser(v) for v in X], [L.deser(v) for v in W])
                     for a, b, X, W, _ in d['recs']]
        self.j_c = d['j_c']
        self.T = T
        self.phi, self.EL, self.ystar, self.Kball = S.phi, S.EL, S.ystar, S.Kball
        self.M, self.Minv, self.rho, self.r = S.M, S.Minv, S.rho, S.r
        ctx.prec = prec
        self.prec = prec
        self._jets = {}

    def jets(self, j):
        """Taylor coefficients (order P_PATH) of the solutions from the hull X_j, and the order P_PATH + 1 coefficient
        over W_j (values only)."""
        if j not in self._jets:
            t0, h, X, W = self.recs[j]
            xk = hhjet6.values(X, self.phi, self.EL, P_PATH)
            xW = hhjet6.values(W, self.phi, self.EL, P_PATH + 1)
            self._jets[j] = (xk, [xW[c][P_PATH + 1] for c in range(5)])
        return self._jets[j]

    def path_box(self, j, a, b):
        """The box containing x(t_j + tau) for tau in [a, b] (fractions a, b of the step, 0 <= a < b <= 1)."""
        t0, h, X, W = self.recs[j]
        xk, rem = self.jets(j)
        ta, tb = h * a, h * b
        tau = ball(ta.lower() if hasattr(ta, 'lower') else ta, arb(tb.upper()))
        out = []
        for c in range(5):
            v = L.horner(xk[c][:P_PATH + 1], tau) + ball(0, arb(tb.upper()) ** (P_PATH + 1)) * rem[c]
            vi = v.intersection(W[c])
            out.append(vi if vi.is_finite() else W[c])
        return out

    def b0_cells(self, which='full', n0=(1, 1, 2, 2, 1)):
        """Initial zeta cells covering B0 (which='full': |zeta_1| <= r) or B0 with |zeta_1| <= |zeta_s| bound rho
        ('entrance': |zeta_1| <= rho)."""
        import itertools
        R1 = block0.up(self.r) if which == 'full' else block0.up(self.rho)
        Rs = block0.up(self.rho)
        bound = [R1] + [Rs] * 4
        cells = []
        for idx in itertools.product(*[range(n) for n in n0]):
            cell = [(-bound[k] + 2 * bound[k] * idx[k] / n0[k], -bound[k] + 2 * bound[k] * (idx[k] + 1) / n0[k])
                    for k in range(5)]
            cells.append(cell)
        return cells

    def zcell_box(self, cell):
        zbox = [ball(a, b) for a, b in cell]
        return [self.ystar[i] + sum((self.Minv[i, k] * zbox[k] for k in range(5)), arb(0)) for i in range(5)]


def run_checks(R, test, parts=('B', 'record', 'B0'), j_from=0, max_time_depth=10, max_z_depth=16, log=None,
               time_pieces=None):
    """Apply test(ybox) -> bool over the region; returns dict(ok, counts, failures). time_pieces(j) gives the initial
    number of pieces of step j (default 1)."""
    t0 = time.time()
    stats = {'B': 0, 'record': 0, 'B0': 0}
    fails = []
    ctx.prec = R.prec
    if 'B' in parts:
        stats['B'] += 1
        if not test(R.boxB):
            fails.append(('B', None))
            return dict(ok=False, stats=stats, fails=fails, secs=round(time.time() - t0))
    if 'record' in parts:
        for j in range(j_from, len(R.recs)):
            m0 = time_pieces(j) if time_pieces else 1
            stack = [(k / m0, (k + 1) / m0, 0) for k in range(m0)]
            while stack:
                a, b, dep = stack.pop()
                stats['record'] += 1
                ok = False
                try:
                    ok = test(R.path_box(j, arb(a), arb(b)))
                except (ArithmeticError, ZeroDivisionError, ValueError):
                    ok = False
                if ok:
                    continue
                if dep >= max_time_depth:
                    fails.append(('record', (j, a, b)))
                    return dict(ok=False, stats=stats, fails=fails, secs=round(time.time() - t0))
                m = (a + b) / 2
                stack += [(a, m, dep + 1), (m, b, dep + 1)]
            if log and j % 100 == 0:
                log('   record step %d of %d, %d boxes so far (%.0f s)' % (j, len(R.recs), stats['record'],
                                                                         time.time() - t0))
    if 'B0' in parts:
        cyc = block0.CYCLE
        stack = [(c, 0) for c in R.b0_cells()]
        while stack:
            cell, dep = stack.pop()
            if not block0.meets_ball(cell[1:], R.rho):
                continue
            stats['B0'] += 1
            ok = False
            try:
                ok = test(R.zcell_box(cell))
            except (ArithmeticError, ZeroDivisionError, ValueError):
                ok = False
            if ok:
                continue
            if dep >= max_z_depth:
                fails.append(('B0', cell))
                return dict(ok=False, stats=stats, fails=fails, secs=round(time.time() - t0))
            k = cyc[dep % len(cyc)]
            a, b = cell[k]
            m = (a + b) / 2
            stack += [(cell[:k] + [(a, m)] + cell[k + 1:], dep + 1), (cell[:k] + [(m, b)] + cell[k + 1:], dep + 1)]
    return dict(ok=not fails, stats=stats, fails=fails, secs=round(time.time() - t0))


def jac5(ybox, K, phi, EL):
    """Df over a y-box (5 x 5 list of balls) with K as given (a ball)."""
    J = hhjet6.jacobian(list(ybox) + [K], phi, EL)
    return [[J[i][j] for j in range(5)] for i in range(5)]


def zeroth_order(ybox, phi, EL):
    """(a, b, c, kappa) over the box: a = dI/du, b = dI/d(m, n, h), c_i = phi dG_i/du, kappa_i = phi (alpha + beta)."""
    J = hhjet6.jacobian(list(ybox) + [arb(1)], phi, EL)
    a = J[1][0]
    b = [J[1][2], J[1][3], J[1][4]]
    c = [J[2][0], J[3][0], J[4][0]]
    kap = [-J[2][2], -J[3][3], -J[4][4]]
    return a, b, c, kap
