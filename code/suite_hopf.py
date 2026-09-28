#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Lohner integrator on the supercritical Hopf normal form, embedded like suite_exact.

The first two components move,
    x' = mu x - omega y - x (x^2 + y^2)
    y' = omega x + mu y - y (x^2 + y^2)
with mu = omega = 1. Components 2, 3, 4 stay at their initial values and component 5 is the
constant parameter the integrator already carries.

The exact cycle of radius 1 is x = cos(t), y = sin(t). Integrate from t = 0 to t = pi/2 at
order 12. Exit status 0 means every check passed, including the negative control.
"""
import sys

from flint import arb, arb_mat, ctx

import lohner6 as L

ctx.prec = 128

MU = arb(1)
OMEGA = arb(1)


class HopfODE:
    """Taylor jet of the Hopf field, with gradients in the initial data."""

    def __init__(self):
        self.prec_aux = L.PREC_AUX

    def vals(self, x, p):
        return self.jet(x, p)[0]

    def f(self, x):
        v = self.vals(x, 1)
        return [v[i][1] for i in range(5)] + [arb(0)] * (len(x) - 5)

    def jet(self, x, p):
        """vals[i][k] and grads[i][k][j] = d(coeff k of component i) / d(initial component j)."""
        n = len(x)
        mu, om = MU, OMEGA

        def add(a, b):
            return [a[0] + b[0], [a[1][j] + b[1][j] for j in range(n)]]

        def sub(a, b):
            return [a[0] - b[0], [a[1][j] - b[1][j] for j in range(n)]]

        def mul(a, b):
            return [a[0] * b[0], [a[1][j] * b[0] + a[0] * b[1][j] for j in range(n)]]

        def divc(a, s):
            return [a[0] / s, [a[1][j] / s for j in range(n)]]

        def scale(a, s):
            return [s * a[0], [s * g for g in a[1]]]

        def conv(xs, ys, k):
            acc = None
            for j in range(k + 1):
                t = mul(xs[j], ys[k - j])
                acc = t if acc is None else add(acc, t)
            return acc

        gx = [arb(0)] * n
        gx[0] = arb(1)
        gy = [arb(0)] * n
        gy[1] = arb(1)
        ax = [[x[0], gx]]
        ay = [[x[1], gy]]
        r2s = []
        for k in range(p):
            r2 = add(conv(ax, ax, k), conv(ay, ay, k))
            r2s.append(r2)
            xr = conv(ax, r2s, k)
            yr = conv(ay, r2s, k)
            # (k+1) a_{k+1} = mu a_k - omega b_k - (x r^2)_k, and likewise for b
            fx = sub(sub(scale(ax[k], mu), scale(ay[k], om)), xr)
            fy = sub(add(scale(ax[k], om), scale(ay[k], mu)), yr)
            den = arb(k + 1)
            ax.append(divc(fx, den))
            ay.append(divc(fy, den))

        vals = [[ax[k][0] for k in range(p + 1)], [ay[k][0] for k in range(p + 1)]]
        grads = [[[ax[k][1][j] for j in range(n)] for k in range(p + 1)],
                 [[ay[k][1][j] for j in range(n)] for k in range(p + 1)]]
        for i in range(2, n):
            vals.append([x[i]] + [arb(0)] * p)
            row = []
            for k in range(p + 1):
                if k == 0:
                    g = [arb(0)] * n
                    g[i] = arb(1)
                    row.append(g)
                else:
                    row.append([arb(0)] * n)
            grads.append(row)
        return vals, grads


def point(x0, y0):
    n = 6
    xbar = [arb(x0), arb(y0), arb(0), arb(0), arb(0), arb(0)]
    C = arb_mat(n, n)
    B = arb_mat([[arb(1) if i == j else arb(0) for j in range(n)] for i in range(n)])
    return L.LSet(xbar, C, [L.ball(-1, 1)] * n, B, [arb(0)] * n)


def reach_half_pi(X, tol):
    """Integrate to the float just below pi/2, then one step across the exact remaining gap.

    Step sizes inside integrate are dyadic floats, so the landing time is not pi/2. The last
    step is taken at the arb gap pi/2 - t, which encloses the value at t = pi/2.
    """
    F = HopfODE()
    half = arb.pi() / 2
    X, t, ns = L.integrate(F, X, float(half), 12, tol, hmax=0.25)
    gap = half - t
    if not gap.lower() > 0:
        raise RuntimeError('stopped at or beyond pi/2 (gap %s)' % gap)
    X, _, _ = L.step(F, X, gap, 12, tol=tol)
    return X.hull(), ns


def polar_at(u0, t):
    """Closed form: r^2' = 2 r^2 (1 - r^2), theta' = 1, theta(0) = 0. Independent of the jet."""
    e = (2 * t).exp()
    u = u0 * e / (1 - u0 + u0 * e)
    r = u.sqrt()
    return r * t.cos(), r * t.sin(), u


def main():
    ok = True
    tol = 1e-16
    half = arb.pi() / 2
    x_exact, y_exact = half.cos(), half.sin()

    hull, ns = reach_half_pi(point('1', '0'), tol)
    hit = hull[0].contains(x_exact) and hull[1].contains(y_exact)
    other = max(float(hull[i].rad()) for i in range(2, 6))
    print('cycle (1, 0) at pi/2: cos contained %s, sin contained %s, radii %.3e %.3e, %d steps; idle radii %.3e' % (
        hull[0].contains(x_exact), hull[1].contains(y_exact),
        float(hull[0].rad()), float(hull[1].rad()), ns, other))
    ok = ok and hit and other < 1e-12

    # omega -> -omega sends the cycle to (cos t, -sin t) = (0, -1) at t = pi/2
    flipped = hull[0].contains(arb(0)) and hull[1].contains(-arb(1))
    print('negative control, sign of omega flipped: (0, -1) excluded %s' % (not flipped))
    ok = ok and not flipped

    hull2, ns2 = reach_half_pi(point('0.5', '0'), tol)
    xref, yref, _ = polar_at(arb('0.25'), half)
    r2 = hull2[0] * hull2[0] + hull2[1] * hull2[1]
    contains_ref = hull2[0].contains(xref) and hull2[1].contains(yref)
    excludes_cycle = not r2.contains(arb(1))
    # a vacuous box would contain the reference and also this clearly wrong value
    excludes_wrong = not (hull2[0].contains(arb(0)) and hull2[1].contains(-arb(1)))
    print('interior (0.5, 0) at pi/2: r^2 excludes 1 %s (r^2 %s), contains reference %s, '
          'excludes (0, -1) %s, radii %.3e %.3e, %d steps' % (
              excludes_cycle, r2.str(8, radius=True), contains_ref, excludes_wrong,
              float(hull2[0].rad()), float(hull2[1].rad()), ns2))
    ok = ok and contains_ref and excludes_cycle and excludes_wrong

    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
