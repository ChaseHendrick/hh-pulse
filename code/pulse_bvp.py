#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Numerical (not rigorous): the Hodgkin-Huxley pulse as a boundary-value problem on a truncated line.

Unknowns: the orbit y(t) on [t-, t+] and K. Conditions: y(t-) - rest lies in the unstable eigenspace of rest (4
conditions: the components along the 4 stable left eigenvectors vanish), y(t+) - rest lies in the stable
eigenspace (1 condition), and a phase condition u(0) = U0. Solved with scipy's collocation solver; the starting
guess comes from double-precision shooting (hhwave.py). Also finds the slow pulse by continuation in temperature.
"""
import sys
import json
import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eig
import hhwave as H

U0 = 50.0      # phase: the upstroke passes u = 50 mV at t = 0


def projections(W, K):
    J = W.jac(W.rest, K)
    lam, VL = eig(J.T)                 # left eigenvectors
    st = [k for k in range(5) if lam[k].real < 0]
    un = [k for k in range(5) if lam[k].real > 0]
    # real basis of the stable left space (complex pair -> real and imaginary parts)
    rows, seen = [], set()
    for k in st:
        if k in seen:
            continue
        v = VL[:, k]
        if abs(lam[k].imag) > 1e-12:
            j = [q for q in st if q != k and abs(lam[q] - np.conj(lam[k])) < 1e-9][0]
            seen.add(j)
            rows += [v.real, v.imag]
        else:
            rows.append(v.real)
        seen.add(k)
    return np.array(rows), VL[:, un[0]].real[None, :]


def fun_vec(W):
    phi, EL = W.phi, W.EL

    def f(t, y, p):
        K = p[0]
        u, w, m, n, h = y
        x1 = (25 - u) / 10
        x2 = (10 - u) / 10
        am = np.where(np.abs(x1) < 1e-6, 1 - x1 / 2, x1 / np.expm1(np.where(np.abs(x1) < 1e-6, 1.0, x1)))
        an = 0.1 * np.where(np.abs(x2) < 1e-6, 1 - x2 / 2, x2 / np.expm1(np.where(np.abs(x2) < 1e-6, 1.0, x2)))
        bm = 4 * np.exp(-u / 18)
        bn = 0.125 * np.exp(-u / 80)
        ah = 0.07 * np.exp(-u / 20)
        bh = 1 / (np.exp((30 - u) / 10) + 1)
        I = 120 * m ** 3 * h * (u - 115) + 36 * n ** 4 * (u + 12) + 0.3 * (u - EL)
        return np.array([w, K * (w + I), phi * (am * (1 - m) - bm * m), phi * (an * (1 - n) - bn * n),
                         phi * (ah * (1 - h) - bh * h)])
    return f


def pulse(W, K0, Tm=-6.0, Tp=40.0, tol=1e-7, guess=None, verbose=0):
    """Solve on [Tm, Tp] with t = 0 at u = U0 on the upstroke. The phase condition is imposed by splitting the
    interval at 0: we solve on [Tm, 0] and [0, Tp] as one 10-dimensional problem on [0, 1] (rescaled time)."""
    f = fun_vec(W)
    a, b = -Tm, Tp

    def F(s, z, p):
        y1, y2 = z[:5], z[5:]
        # y1(s) = y(-a (1 - s)) ... use y1(s) = y(Tm + a s) on [Tm, 0], y2(s) = y(b s) on [0, Tp]
        return np.vstack([a * f(0, y1, p), b * f(0, y2, p)])

    def bc(za, zb, p):
        Ls, Lu = projections(W, p[0])
        y1a, y1b, y2a, y2b = za[:5], zb[:5], za[5:], zb[5:]
        return np.concatenate([Ls @ (y1a - W.rest), y1b - y2a, [y2a[0] - U0], Lu @ (y2b - W.rest)])

    s = np.linspace(0, 1, 2001)
    if guess is None:
        raise ValueError('need a guess')
    tg, Yg = guess
    z = np.vstack([np.array([np.interp(Tm + a * s, tg, Yg[i]) for i in range(5)]),
                   np.array([np.interp(b * s, tg, Yg[i]) for i in range(5)])])
    sol = solve_bvp(F, bc, s, z, p=[K0], tol=tol, max_nodes=150000, verbose=verbose)
    return sol, (a, b, Tm)


def shoot_guess(W, K, Tm=-6.0, Tp=40.0):
    """A starting guess: the shooting orbit until it leaves, then relaxation to rest."""
    s, sol = W.shoot(K, delta=1e-9, dense=True)
    t = sol.t
    u = sol.y[0]
    i0 = np.argmax(u > U0)
    tt = t - t[i0]
    # cut where the orbit starts to leave (u at its minimum after the peak, before the escape)
    ipk = np.argmax(u)
    tail = np.arange(ipk, len(t))
    icut = tail[np.argmin(u[tail])] if s == -1 else tail[np.argmin(u[tail])]
    tg = list(tt[:icut])
    Y = [list(sol.y[i][:icut]) for i in range(5)]
    tn = np.linspace(tg[-1], Tp, 400)[1:]
    lam = 0.3
    for i in range(5):
        y0 = Y[i][-1]
        Y[i] += list(W.rest[i] + (y0 - W.rest[i]) * np.exp(-lam * (tn - tg[-1])))
    tg += list(tn)
    tg = np.array(tg)
    # extend to Tm with rest
    if tg[0] > Tm:
        tg = np.concatenate([[Tm], tg])
        Y = [[W.rest[i]] + Y[i] for i in range(5)]
    return tg, np.array(Y)


def profile(sol, geom, n=4001):
    a, b, Tm = geom
    s = np.linspace(0, 1, n)
    z = sol.sol(s)
    t = np.concatenate([Tm + a * s, b * s[1:]])
    Y = np.hstack([z[:5], z[5:, 1:]])
    return t, Y


if __name__ == '__main__':
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 18.5
    W = H.Wave(T)
    lo, hi, _, _ = H.bisect_K(W, 3, 30, tol=1e-12)
    g = shoot_guess(W, lo)
    sol, geom = pulse(W, lo, guess=g)
    print('T', T, 'status', sol.status, sol.message, 'K', sol.p[0], 'shooting K', lo, 'nodes', sol.x.size)
    print('speed m/s', H.speed_from_K(sol.p[0]))
    t, Y = profile(sol, geom)
    d = np.max(np.abs((Y - W.rest[:, None]) / np.array([100, 1000, 1, 1, 1])[:, None]), axis=0)
    ipk = np.argmax(Y[0])
    print('peak u %.3f mV at t = %.3f ms; min u %.3f mV at t = %.3f ms' % (Y[0][ipk], t[ipk], Y[0].min(), t[np.argmin(Y[0])]))
    for thr in [1e-2, 1e-3, 1e-4]:
        late = np.where((t > t[ipk]) & (d > thr))[0]
        print('scaled distance from rest last above %g at t = %.2f ms' % (thr, t[late[-1]] if late.size else float('nan')))
    np.savez('../data/pulse_%s.npz' % T, t=t, Y=Y, K=sol.p[0])
