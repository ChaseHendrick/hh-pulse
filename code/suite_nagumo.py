#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Exact Nagumo front for the Lohner integrator, not the Hodgkin-Huxley field.

tools/rdx-science.js records the bistable front of
    u_t = u_xx + f(u),  f(u) = -(u - u1)(u - u2)(u - u3),
as c = (u1 + u3 - 2 u2) / sqrt(2) and
    phi(xi) = u1 + (u3 - u1) / (1 + exp(kappa xi)),  kappa = (u3 - u1) / sqrt(2).
The cubic u(1 - u)(u - a) is that polynomial at the roots (0, a, 1). With a = 1/4,
    phi(xi) = 1 / (1 + exp(xi / sqrt(2))),  c = sqrt(2) / 4,
and phi'' + c phi' + phi(1 - phi)(phi - a) = 0. As a first-order system the components
are (phi, phi'): phi' = v, v' = -c v - phi(1 - phi)(phi - a). Components 0 and 1 move,
components 2, 3 and 4 stay at their initial values, and component 5 is the constant
parameter, as in suite_exact.py.

Exit status 0 means every check passed, including the negative control.
"""
import sys

from flint import arb, arb_mat, ctx

import lohner6 as L

ctx.prec = 128


def _nonlinearity(u, du, k, a, n):
    """Coefficient k of f(u) = -u^3 + (1 + a) u^2 - a u, and its gradient in the initial data."""
    u2 = []
    du2 = [] if du is not None else None
    for m in range(k + 1):
        u2.append(sum((u[j] * u[m - j] for j in range(m + 1)), arb(0)))
        if du is not None:
            d = []
            for jdir in range(n):
                s = arb(0)
                for j in range(m + 1):
                    s += du[j][jdir] * u[m - j] + u[j] * du[m - j][jdir]
                d.append(s)
            du2.append(d)
    u3 = sum((u2[j] * u[k - j] for j in range(k + 1)), arb(0))
    one_a = arb(1) + a
    f = -u3 + one_a * u2[k] - a * u[k]
    if du is None:
        return f, None
    df = []
    for jdir in range(n):
        du3 = arb(0)
        for j in range(k + 1):
            du3 += du2[j][jdir] * u[k - j] + u2[j] * du[k - j][jdir]
        df.append(-du3 + one_a * du2[k][jdir] - a * du[k][jdir])
    return f, df


class NagumoODE:
    """Taylor jet of the Nagumo profile ODE embedded in the integrator's six components."""

    def __init__(self):
        self.prec_aux = L.PREC_AUX

    def _series(self, x, p, with_grad):
        n = len(x)
        a = arb(1) / 4
        c = arb(2).sqrt() / 4
        u = [x[0]]
        w = [x[1]]
        if with_grad:
            du = [[arb(1) if j == 0 else arb(0) for j in range(n)]]
            dw = [[arb(1) if j == 1 else arb(0) for j in range(n)]]
        else:
            du = dw = None
        for k in range(p):
            f, df = _nonlinearity(u, du, k, a, n)
            u.append(w[k] / (k + 1))
            w.append((-c * w[k] - f) / (k + 1))
            if with_grad:
                du.append([dw[k][j] / (k + 1) for j in range(n)])
                dw.append([(-c * dw[k][j] - df[j]) / (k + 1) for j in range(n)])
        return u, w, du, dw

    def _pack(self, x, p, u, w):
        out = [u, w]
        for i in range(2, 5):
            out.append([x[i]] + [arb(0)] * p)
        for q in range(5, len(x)):
            out.append([x[q]] + [arb(0)] * p)
        return out

    def vals(self, x, p):
        u, w, _, _ = self._series(x, p, False)
        return self._pack(x, p, u, w)

    def f(self, x):
        v = self.vals(x, 1)
        return [v[i][1] for i in range(5)] + [arb(0)] * (len(x) - 5)

    def jet(self, x, p):
        n = len(x)
        u, w, du, dw = self._series(x, p, True)
        grads = [du, dw]
        for i in range(2, 5):
            row = []
            for k in range(p + 1):
                g = [arb(0)] * n
                if k == 0:
                    g[i] = arb(1)
                row.append(g)
            grads.append(row)
        for q in range(5, n):
            grads.append([[arb(1) if j == q else arb(0) for j in range(n)]]
                         + [[arb(0)] * n for _ in range(p)])
        return self._pack(x, p, u, w), grads


def initial_set():
    """Point (1/2, -1/(4 sqrt(2)), 0, 0, 0, 0), with the radius of the v0-ball carried in C."""
    n = 6
    v_exact = -arb(1) / (arb(4) * arb(2).sqrt())
    xbar = [arb(1) / 2, arb(v_exact.mid()), arb(0), arb(0), arb(0), arb(0)]
    C = arb_mat(n, n)
    C[1, 1] = v_exact.rad()
    B = arb_mat([[arb(1) if i == j else arb(0) for j in range(n)] for i in range(n)])
    return L.LSet(xbar, C, [L.ball(-1, 1)] * n, B, [arb(0)] * n), v_exact


def closed_form(xi):
    """phi and phi' of 1 / (1 + exp(xi / sqrt(2))), xi an arb."""
    alpha = arb(1) / arb(2).sqrt()
    e = (alpha * xi).exp()
    phi = arb(1) / (arb(1) + e)
    dphi = -alpha * e / (arb(1) + e) ** 2
    return phi, dphi


def main():
    F = NagumoODE()
    X, v_exact = initial_set()
    hull0 = X.hull()
    if not (hull0[0].contains(arb(1) / 2) and hull0[1].contains(v_exact)):
        print('initial set misses (phi(0), phi\'(0))')
        return 1
    X, t, ns = L.integrate(F, X, 1, 12, 1e-16, hmax=0.25)
    hull = X.hull()
    phi1, dphi1 = closed_form(arb(1))
    phi_other, _ = closed_form(-arb(1))
    hit = hull[0].contains(phi1)
    hit_d = hull[1].contains(dphi1)
    excluded = not hull[0].contains(phi_other)
    on_time = abs(float(t.mid()) - 1.0) < 1e-12
    print('phi(1) contains 1/(1+exp(1/sqrt(2))): %s, t=%.12g, steps %d, radius %.3e' % (
        hit and on_time, float(t.mid()), ns, float(hull[0].rad())))
    print('phi\'(1) contains the closed form: %s, radius %.3e' % (hit_d, float(hull[1].rad())))
    print('negative control: phi(1) excludes phi(-1): %s' % excluded)
    return 0 if (hit and hit_d and excluded and on_time) else 1


if __name__ == '__main__':
    sys.exit(main())
