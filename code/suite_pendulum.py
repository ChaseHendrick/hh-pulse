#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Lohner integrator on the simple pendulum, whose solution is a Jacobi elliptic function.

The first two components move,
    theta' = omega
    omega' = -sin theta
with theta(0) = theta_m = 1/2, omega(0) = 0. Components 2, 3 and 4 stay at 0, and
component 5 is the constant parameter the integrator already carries.

k = sin(theta_m / 2). K(k) = pi / (2 agm(1, sqrt(1 - k^2))) is the complete elliptic
integral of the first kind. The quarter period is K(k): at t = K(k) the bob is at the
bottom, theta = 0 and omega = -sqrt(2 (1 - cos theta_m)). The exact solution is
sin(theta(t) / 2) = k * sn(K(k) - t, k). flint does not expose Jacobi sn (arb.jacobi_p
is the Jacobi polynomial), so there is no intermediate check at t = K(k) / 2.

Exit status 0 means every check passed, including the negative control.
"""
import math
import sys

from flint import acb, arb, arb_mat, ctx

import lohner6 as L

ctx.prec = 128

THETA_M = arb('0.5')
ORDER = 12
TOL = 1e-14


def complete_K(k):
    """K(k) for the elliptic modulus k, via the arithmetic-geometric mean."""
    return arb.pi() / (2 * arb(1).agm((1 - k * k).sqrt()))


def jacobi_sn_binding():
    """Return a Jacobi sn callable if flint exposes one, else None.

    arb.jacobi_p is the Jacobi polynomial, not the elliptic sine. Smith normal form
    (snf) and the modular theta functions are not sn either.
    """
    for obj in (arb, acb):
        for name in ('jacobi_sn', 'elliptic_sn', 'sn'):
            fn = getattr(obj, name, None)
            if fn is not None:
                return fn
    return None


class PendulumODE:
    """Taylor jet of theta' = omega, omega' = -sin theta, with gradients in the initial data."""

    def __init__(self):
        self.prec_aux = L.PREC_AUX

    def _series(self, x, p):
        """Power series of (theta, omega) and of sin(theta), cos(theta), plus gradients.

        (k + 1) theta_{k+1} = omega_k, (k + 1) omega_{k+1} = -sin(theta)_k, and
        (k + 1) s_{k+1} = (cos(theta) * omega)_k, (k + 1) c_{k+1} = -(sin(theta) * omega)_k.
        grads[i][k][j] = d(coeff k of component i) / d(initial component j).
        """
        n = len(x)
        th = [x[0]]
        w = [x[1]]
        s = [x[0].sin()]
        c = [x[0].cos()]
        dth = [[arb(1) if j == 0 else arb(0) for j in range(n)]]
        dw = [[arb(1) if j == 1 else arb(0) for j in range(n)]]
        ds = [[c[0] * dth[0][j] for j in range(n)]]
        dc = [[-s[0] * dth[0][j] for j in range(n)]]
        for k in range(p):
            den = k + 1
            th.append(w[k] / den)
            w.append(-s[k] / den)
            dth.append([dw[k][j] / den for j in range(n)])
            dw.append([-ds[k][j] / den for j in range(n)])
            acc_s, acc_c = arb(0), arb(0)
            dacc_s = [arb(0)] * n
            dacc_c = [arb(0)] * n
            for j in range(k + 1):
                acc_s += c[j] * w[k - j]
                acc_c -= s[j] * w[k - j]
                for q in range(n):
                    dacc_s[q] += dc[j][q] * w[k - j] + c[j] * dw[k - j][q]
                    dacc_c[q] -= ds[j][q] * w[k - j] + s[j] * dw[k - j][q]
            s.append(acc_s / den)
            c.append(acc_c / den)
            ds.append([dacc_s[q] / den for q in range(n)])
            dc.append([dacc_c[q] / den for q in range(n)])
        return th, w, dth, dw

    def _pack(self, x, p, th, w):
        out = [th, w]
        for i in range(2, 5):
            out.append([x[i]] + [arb(0)] * p)
        for q in range(5, len(x)):
            out.append([x[q]] + [arb(0)] * p)
        return out

    def vals(self, x, p):
        th, w, _, _ = self._series(x, p)
        return self._pack(x, p, th, w)

    def f(self, x):
        v = self.vals(x, 1)
        return [v[i][1] for i in range(5)] + [arb(0)] * (len(x) - 5)

    def jet(self, x, p):
        n = len(x)
        th, w, dth, dw = self._series(x, p)
        grads = [dth, dw]
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
        return self._pack(x, p, th, w), grads


def point(theta0):
    n = 6
    xbar = [arb(theta0), arb(0), arb(0), arb(0), arb(0), arb(0)]
    C = arb_mat(n, n)
    B = arb_mat([[arb(1) if i == j else arb(0) for j in range(n)] for i in range(n)])
    return L.LSet(xbar, C, [L.ball(-1, 1)] * n, B, [arb(0)] * n)


def _float_below(target):
    """A float strictly less than the arb target, so the last step can cover the exact gap."""
    t = float(target.mid())
    for _ in range(8):
        if (target - arb(t)).lower() > 0:
            return t
        t = math.nextafter(t, 0.0)
    raise RuntimeError('no float strictly below the target')


def reach(X, target, tol):
    """Integrate to a float just below target, then one step across the exact arb gap."""
    F = PendulumODE()
    X, t, ns = L.integrate(F, X, _float_below(target), ORDER, tol, hmax=0.25)
    gap = target - t
    if not gap.lower() > 0:
        raise RuntimeError('stopped at or beyond the target (gap %s)' % gap)
    X, _, _ = L.step(F, X, gap, ORDER, tol=tol)
    return X.hull(), ns + 1


def main():
    ok = True
    k = (THETA_M / 2).sin()
    K = complete_K(k)
    omega_down = -(2 * (1 - THETA_M.cos())).sqrt()
    omega_up = -omega_down

    sn = jacobi_sn_binding()
    if sn is None:
        print('Jacobi sn: not in flint bindings; quarter-period energy check only')
    else:
        print('Jacobi sn: exposed as %s' % sn)

    hull, ns = reach(point(THETA_M), K, TOL)
    hit_th = hull[0].contains(arb(0))
    hit_w = hull[1].contains(omega_down)
    excluded = not hull[1].contains(omega_up)
    other = max(float(hull[i].rad()) for i in range(2, len(hull)))
    idle = other < 1e-12 and all(hull[i].contains(arb(0)) for i in range(2, len(hull)))

    print('theta(K) contains 0: %s, radius %.3e, %d steps' % (hit_th, float(hull[0].rad()), ns))
    print('omega(K) contains -sqrt(2(1-cos theta_m)): %s, radius %.3e' % (hit_w, float(hull[1].rad())))
    print('negative control, upward sign excluded: %s' % excluded)
    print('idle radii < 1e-12: %s, max %.3e' % (idle, other))
    ok = ok and hit_th and hit_w and excluded and idle

    if sn is not None:
        half = K / 2
        hull_h, ns_h = reach(point(THETA_M), half, TOL)
        # sin(theta/2) = k * sn(K - t, k) at t = K/2
        z = K - half
        try:
            sn_val = sn(z, k * k) if 'elliptic' in getattr(sn, '__name__', '') else sn(z, k)
            target = k * sn_val
            if hasattr(target, 'real'):
                target = target.real
            got = (hull_h[0] / 2).sin()
            hit_sn = got.contains(target) if hasattr(target, 'mid') else got.contains(arb(target))
        except (TypeError, ValueError, ArithmeticError) as exc:
            hit_sn = False
            print('sn(K/2) check failed to evaluate: %s' % exc)
        else:
            print('sin(theta(K/2)/2) contains k*sn(K/2, k): %s, %d steps' % (hit_sn, ns_h))
        ok = ok and hit_sn

    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
