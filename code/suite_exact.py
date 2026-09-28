#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Exact solutions for the Lohner integrator, not the Hodgkin-Huxley field.

The moving component is the first one. The other four stay at their initial values, and the
sixth component is the constant parameter the integrator already carries.

1. y' = -y, y(0) = 1. At t = 1 the solution is e^{-1}.
2. y' = y(1 - y), y(0) = 1/2. At t = 1 the solution is 1 / (1 + e^{-1}).
3. A ball of radius 1e-6 around y(0) = 1 for y' = -y. The image of that ball at t = 1 is
   e^{-1} times the ball, and the enclosure must contain both ends.
4. Negative control: the decay enclosure must not contain e^{+1}.

Exit status 0 means every check passed, including the negative control.
"""
import sys

from flint import arb, arb_mat, ctx

import lohner6 as L

ctx.prec = 128


class ScalarODE:
    """Taylor jet of a scalar ODE embedded in the integrator's six components."""

    def __init__(self, kind):
        self.kind = kind
        self.prec_aux = L.PREC_AUX

    def _series(self, x0, p):
        c = [arb(x0)]
        for k in range(p):
            if self.kind == 'decay':
                c.append(-c[k] / (k + 1))
            else:
                square = sum((c[j] * c[k - j] for j in range(k + 1)), arb(0))
                c.append((c[k] - square) / (k + 1))
        return c

    def vals(self, x, p):
        out = [self._series(x[0], p)]
        for i in range(1, 5):
            out.append([x[i]] + [arb(0)] * p)
        for q in range(5, len(x)):
            out.append([x[q]] + [arb(0)] * p)
        return out

    def f(self, x):
        v = self.vals(x, 1)
        return [v[i][1] for i in range(5)] + [arb(0)] * (len(x) - 5)

    def jet(self, x, p):
        n = len(x)
        vals = self.vals(x, p)
        y = vals[0]
        z = [arb(1)]
        for k in range(p):
            if self.kind == 'decay':
                z.append(-z[k] / (k + 1))
            else:
                prod = sum((y[j] * z[k - j] for j in range(k + 1)), arb(0))
                z.append((z[k] - 2 * prod) / (k + 1))
        grads = []
        for i in range(5):
            row = []
            for k in range(p + 1):
                g = [arb(0)] * n
                if i == 0:
                    g[0] = z[k]
                if k == 0:
                    g[i] = arb(1)
                row.append(g)
            grads.append(row)
        for q in range(5, n):
            grads.append([[arb(1) if j == q else arb(0) for j in range(n)]]
                         + [[arb(0)] * n for _ in range(p)])
        return vals, grads


def point(y0):
    n = 6
    xbar = [arb(y0), arb(0), arb(0), arb(0), arb(0), arb(0)]
    C = arb_mat(n, n)
    B = arb_mat([[arb(1) if i == j else arb(0) for j in range(n)] for i in range(n)])
    return L.LSet(xbar, C, [L.ball(-1, 1)] * n, B, [arb(0)] * n)


def ball_set(y0, rad):
    n = 6
    xbar = [arb(y0), arb(0), arb(0), arb(0), arb(0), arb(0)]
    C = arb_mat(n, n)
    C[0, 0] = arb(rad)
    B = arb_mat([[arb(1) if i == j else arb(0) for j in range(n)] for i in range(n)])
    return L.LSet(xbar, C, [L.ball(-1, 1)] * n, B, [arb(0)] * n)


def go(kind, y0, T, rad=0):
    F = ScalarODE(kind)
    X = ball_set(y0, rad) if rad else point(y0)
    X, t, ns = L.integrate(F, X, T, 12, 1e-16, hmax=0.25)
    return X.hull(), float(t.mid()), ns


def contains(box, value):
    return box.contains(value)


def main():
    ok = True
    e_minus = (-arb(1)).exp()
    e_plus = arb(1).exp()

    hull, t, ns = go('decay', '1', 1)
    hit = contains(hull[0], e_minus)
    other = max(float(hull[i].rad()) for i in range(1, 5))
    not_growth = not contains(hull[0], e_plus)
    print('decay y(1) = e^{-1}: contained %s in %d steps, radius %.3e; e^{+1} excluded %s; other radii %.3e' % (
        hit, ns, float(hull[0].rad()), not_growth, other))
    ok = ok and hit and not_growth and other < 1e-12 and abs(t - 1) < 1e-12

    hull, t, ns = go('logistic', '0.5', 1)
    exact = arb(1) / (arb(1) + e_minus)
    hit = contains(hull[0], exact)
    print('logistic y(1) = 1/(1+e^{-1}): contained %s in %d steps, radius %.3e' % (
        hit, ns, float(hull[0].rad())))
    ok = ok and hit and abs(t - 1) < 1e-12

    rad = '1e-6'
    hull, t, ns = go('decay', '1', 1, rad=rad)
    lo = (arb(1) - arb('1e-6')) * e_minus
    hi = (arb(1) + arb('1e-6')) * e_minus
    ends = contains(hull[0], lo) and contains(hull[0], hi)
    print('decay of the ball 1 ± 1e-6: both ends contained %s, radius %.3e' % (ends, float(hull[0].rad())))
    ok = ok and ends

    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
