#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Numerical (not rigorous): floating-point groundwork for the stability proof of the pulse (the design note,
Section 8.3, not part of this release). Nothing here is part of a proof; the rigorous programs
only take numbers from it (weights of cone coordinates, sizes of cells and segments) and check everything themselves.

Eigenvalue problem in the moving frame xi = t - x/theta, first-order form Y = (p, p', q_m, q_n, q_h):
    Y' = A(xi, lambda) Y,   A = Df(x(xi), K) + lambda E,   E = K e2 e1^T - (e3 e3^T + e4 e4^T + e5 e5^T).
Evans function D(lambda) = psi~(xm)^T phi~(xm), phi~ = e^{-nu xi} phi^- from v (xi = XL), psi~ = e^{nu xi} psi^+ from
w (xi = XR), nu the unstable eigenvalue of A_inf(lambda), w^T v = 1.

Usage:
  python3 stab_num.py profile [T]            # collocation profile with the printed E_l -> ../data/logs/stab_profile_<T>.npz
  python3 stab_num.py ess [T]                # edge of the essential spectrum
  python3 stab_num.py winding T x0 x1 y1     # winding number on the box [x0, x1] x [-y1, y1] (upper half, adaptive)
  python3 stab_num.py energy [T]             # the energy bound on Re lambda (constant gate scaling, optimized)
  python3 stab_num.py coneB0 T lam...        # cone condition on the closing block, optimized weights
  python3 stab_num.py conepath T lam...      # cone condition along the whole profile, optimized weights
"""
import math
import os
import sys
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import hhwave as H

EL = float(os.environ.get('HH_EL', '10.613'))
LOGS = '../data/logs'


def profile(T):
    import pulse_bvp as P
    W = H.Wave(T, EL=EL)
    lo, hi, _, _ = H.bisect_K(W, 3, 30, tol=1e-12)
    g = P.shoot_guess(W, lo)
    sol, geom = P.pulse(W, lo, guess=g, tol=1e-9, Tp=45.0 if T > 10 else 100.0)
    t, Y = P.profile(sol, geom, n=20001)
    os.makedirs(LOGS, exist_ok=True)
    np.savez('%s/stab_profile_%s.npz' % (LOGS, T), t=t, Y=Y, K=sol.p[0], rest=W.rest, phi=W.phi, EL=W.EL)
    print('status', sol.status, 'K', sol.p[0], 'nodes', sol.x.size)


class Model:
    def __init__(self, T):
        d = np.load('%s/stab_profile_%s.npz' % (LOGS, T))
        t, Y = d['t'], d['Y']
        keep = np.concatenate([[True], np.diff(t) > 0])
        self.t, self.Y = t[keep], Y[:, keep]
        self.K, self.rest, self.phi = float(d['K']), d['rest'], float(d['phi'])
        self.S = CubicSpline(self.t, self.Y, axis=1)
        self.E = np.zeros((5, 5))
        self.E[1, 0] = self.K
        self.E[2, 2] = self.E[3, 3] = self.E[4, 4] = -1
        self.A0 = self.Df(self.rest)
        self.XL, self.XR = self.t[0], self.t[-1]

    def Df(self, y):
        K, phi = self.K, self.phi
        u, w, m, n, h = y

        def psi(x):
            return 1 - x / 2 + x * x / 12 if abs(x) < 1e-6 else x / math.expm1(x)

        def dpsi(x):
            if abs(x) < 1e-4:
                return -0.5 + x / 6
            e = math.exp(x)
            return ((e - 1) - x * e) / (e - 1) ** 2
        x1, x2 = (25 - u) / 10, (10 - u) / 10
        am, dam = psi(x1), -dpsi(x1) / 10
        an, dan = 0.1 * psi(x2), -0.1 * dpsi(x2) / 10
        bm = 4 * math.exp(-u / 18)
        dbm = -bm / 18
        bn = 0.125 * math.exp(-u / 80)
        dbn = -bn / 80
        ah = 0.07 * math.exp(-u / 20)
        dah = -ah / 20
        e = math.exp((30 - u) / 10)
        bh = 1 / (e + 1)
        dbh = e / 10 / (e + 1) ** 2
        J = np.zeros((5, 5))
        J[0, 1] = 1
        J[1] = [K * (120 * m ** 3 * h + 36 * n ** 4 + 0.3), K, K * 360 * m ** 2 * h * (u - 115),
                K * 144 * n ** 3 * (u + 12), K * 120 * m ** 3 * (u - 115)]
        J[2, 0] = phi * (dam * (1 - m) - dbm * m)
        J[2, 2] = -phi * (am + bm)
        J[3, 0] = phi * (dan * (1 - n) - dbn * n)
        J[3, 3] = -phi * (an + bn)
        J[4, 0] = phi * (dah * (1 - h) - dbh * h)
        J[4, 4] = -phi * (ah + bh)
        return J

    def eig_u(self, lam):
        A = self.A0 + lam * self.E
        w, V = np.linalg.eig(A)
        i = int(np.argmax(w.real))
        wl, VL = np.linalg.eig(A.T)
        j = int(np.argmin(abs(wl - w[i])))
        v = V[:, i] / V[0, i]
        l = VL[:, j] / (VL[:, j] @ v)
        return w[i], v, l

    def evans(self, lam, xm=3.0, rtol=1e-10):
        nu, v, l = self.eig_u(lam)
        E, S, I5 = self.E, self.S, np.eye(5)
        a = solve_ivp(lambda s, z: (self.Df(S(s)) + lam * E - nu * I5) @ z, (self.XL, xm), v.astype(complex),
                      method='DOP853', rtol=rtol, atol=1e-14)
        b = solve_ivp(lambda s, z: -(self.Df(S(s)) + lam * E - nu * I5).T @ z, (self.XR, xm), l.astype(complex),
                      method='DOP853', rtol=rtol, atol=1e-14)
        return b.y[:, -1] @ a.y[:, -1]


def ess(T):
    M = Model(T)
    A, K = M.A0, M.K
    B = np.zeros((4, 4))
    B[0, 0] = -A[1, 0] / K
    B[0, 1:] = -A[1, 2:] / K
    B[1:, 0] = A[2:, 0]
    B[1:, 1:] = A[2:, 2:]
    best = -1e9
    for s in np.concatenate([np.linspace(0, 5, 2001), np.logspace(0.7, 7, 600)]):
        Ms = B.copy()
        Ms[0, 0] -= s
        best = max(best, np.linalg.eigvals(Ms).real.max())
    print('sup over s of the spectral abscissa of B - s e1 e1^T: %.7f; h gate rate at rest: %.7f' % (
        best, B[3, 3]))


def winding(T, x0, x1, y1):
    M = Model(T)

    def ad(a, b, fa, fb, depth=0):
        d = np.angle(fb / fa)
        if abs(d) < 0.3 or depth > 14:
            return d, min(abs(fa), abs(fb))
        m = (a + b) / 2
        fm = M.evans(m)
        d1, m1 = ad(a, m, fa, fm, depth + 1)
        d2, m2 = ad(m, b, fm, fb, depth + 1)
        return d1 + d2, min(m1, m2)
    path = [complex(x1, 0), complex(x1, y1), complex(x0, y1), complex(x0, 0)]
    tot, mn = 0.0, 1e300
    for a, b in zip(path[:-1], path[1:]):
        pts = [a + (b - a) * k / 40 for k in range(41)]
        vals = [M.evans(p) for p in pts]
        for i in range(40):
            d, m = ad(pts[i], pts[i + 1], vals[i], vals[i + 1])
            tot += d
            mn = min(mn, m)
    print('D(x0) = %s, D(x1) = %s' % (M.evans(complex(x0, 0)), M.evans(complex(x1, 0))))
    print('winding number (twice the upper half)/2pi = %.6f, min |D| on the contour %.4g' % (2 * tot / (2 * np.pi), mn))


def zmat(M, y, s):
    J = M.Df(y)
    K = M.K
    Z = np.zeros((4, 4))
    Z[0, 0] = -J[1, 0] / K
    Z[0, 1:] = -J[1, 2:] / K / s
    Z[1:, 0] = s * J[2:, 0]
    Z[1:, 1:] = np.diag(np.diag(J[2:, 2:]))
    return Z


def energy(T):
    M = Model(T)
    Ys = M.S(np.linspace(M.XL, M.XR, 20001)).T

    def lam_max(ls, sub):
        s = np.exp(ls)
        return max(np.linalg.eigvalsh((zmat(M, y, s) + zmat(M, y, s).T) / 2).max() for y in Ys[::sub])
    r = minimize(lambda ls: lam_max(ls, 20), np.zeros(3), method='Nelder-Mead', options={'maxiter': 600})
    print('scalings', np.exp(r.x), 'Lambda', lam_max(r.x, 1))


DG = np.diag([1, -1, -1, -1, -1.])


def eig_coords(M, lam, w):
    A = M.A0 + lam * M.E
    ev, V = np.linalg.eig(A)
    order = np.argsort(-ev.real)
    V = V[:, order]
    V = V / V[0, :]
    return np.diag(w) @ np.linalg.inv(V)


def cone_margin(M, lam, w, Js):
    C = eig_coords(M, lam, w)
    Ci = np.linalg.inv(C)
    A = C[None] @ (Js + lam * M.E) @ Ci[None]
    Hm = DG[None] @ A + np.conj(np.transpose(A, (0, 2, 1))) @ DG[None]
    return np.linalg.eigvalsh(Hm).min()


def best_weights(M, lam, Js, starts):
    best = (-1e18, None)
    for x0 in starts:
        r = minimize(lambda lw: -cone_margin(M, lam, np.exp(np.concatenate([[0.0], lw])), Js[::5]), x0,
                     method='Nelder-Mead', options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-6})
        w = np.exp(np.concatenate([[0.0], r.x]))
        m = cone_margin(M, lam, w, Js)
        if m > best[0]:
            best = (m, w)
    return best


STARTS = [np.log(np.array(s, float)) for s in ([37, 1.8, 1.8, 36], [1.7, 37, 1.9, 36], [1.2, 400, 100, 70.],
                                                 [1, 1e4, 1e3, 1e2], [1, 1e3, 1e4, 1e3], [1, 100, 100, 100])]


def sample_B0(M, T, N=3000, seed=1):
    blk = json.load(open('../data/closing_block_%s_El%s.json' % (T, os.environ.get('HH_EL', '10.613'))))
    T0 = np.array([[float.fromhex(v) for v in r] for r in blk['T_hex']])
    Mi = np.linalg.inv(np.diag(np.array(blk['weights'], float)) @ T0)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N, 4))
    z /= np.linalg.norm(z, axis=1)[:, None]
    rad = blk['rho'] * rng.random(N) ** 0.25
    rad[:N // 3] = blk['rho']
    Z = np.column_stack([rng.uniform(-1, 1, N) * blk['r'], z * rad[:, None]])
    return np.array([M.Df(M.rest + Mi @ zz) for zz in Z])


if __name__ == '__main__':
    what = sys.argv[1]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 18.5
    if what == 'profile':
        profile(T)
    elif what == 'ess':
        ess(T)
    elif what == 'winding':
        winding(T, *[float(x) for x in sys.argv[3:6]])
    elif what == 'energy':
        energy(T)
    elif what in ('coneB0', 'conepath'):
        M = Model(T)
        if what == 'coneB0':
            Js = sample_B0(M, T)
        else:
            xs = np.concatenate([np.linspace(-3, 1.5, 3000), np.linspace(1.5, 30, 1500)])
            Js = np.array([M.Df(y) for y in M.S(xs).T])
        for s in sys.argv[3:]:
            lam = complex(s)
            m, w = best_weights(M, lam, Js, STARTS)
            print(lam, 'smallest eigenvalue of the cone matrix %.4g' % m, 'weights', np.array2string(w, precision=4),
                  flush=True)
