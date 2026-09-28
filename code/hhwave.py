#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Numerical (not rigorous) shooting for Hodgkin and Huxley's propagated action potential.

Hodgkin and Huxley, J. Physiol. 117 (1952) 500-544, Part II, eq. (30) with K = 2 R2 theta^2 C / a:

    d^2V/dt^2 = K (dV/dt + I_ion / C),

a travelling wave V(x, t) = V(t - x/theta) of the cable equation (a / 2 R2) V_xx = C V_t + I_ion. The equation is
invariant under V -> -V with I_ion -> -I_ion, so we use the modern depolarization u = -V (mV) and

    u'' = K (u' + I(u, m, n, h)),   I = 120 m^3 h (u - 115) + 36 n^4 (u + 12) + 0.3 (u - E_l),
    x'  = phi (alpha_x(u) (1 - x) - beta_x(u) x),   phi = 3^((T - 6.3)/10),

with t in ms and C = 1 uF/cm^2. State y = (u, w = u', m, n, h). The speed is theta = sqrt(K a / (2 R2 C)).
"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig

A_CM = 238e-4        # fibre radius, cm
R2 = 35.4            # axoplasm resistivity, ohm cm
CM = 1e-6            # F/cm^2
EL_PRINTED = 10.613


def psi(x):
    return 1.0 - x / 2 + x * x / 12 if abs(x) < 1e-6 else x / math.expm1(x)


def rates(u):
    am = psi((25 - u) / 10)
    bm = 4 * math.exp(-u / 18)
    an = 0.1 * psi((10 - u) / 10)
    bn = 0.125 * math.exp(-u / 80)
    ah = 0.07 * math.exp(-u / 20)
    bh = 1 / (math.exp((30 - u) / 10) + 1)
    return am, bm, an, bn, ah, bh


def rest_gates():
    am, bm, an, bn, ah, bh = rates(0.0)
    return am / (am + bm), an / (an + bn), ah / (ah + bh)


def el_zero():
    """The leak potential that makes the resting current zero (10.5989...)."""
    m, n, h = rest_gates()
    return (36 * n ** 4 * 12 - 120 * m ** 3 * h * 115) / 0.3


def phi_of(T):
    return 3.0 ** ((T - 6.3) / 10)


def speed_from_K(K):
    """theta in m/s from K in 1/ms."""
    return math.sqrt(K * 1e3 * A_CM / (2 * R2 * CM)) / 100


def K_from_speed(theta):
    return (theta * 100) ** 2 * 2 * R2 * CM / A_CM / 1e3


class Wave:
    def __init__(self, T=18.5, EL=None):
        self.phi = phi_of(T)
        self.EL = el_zero() if EL is None else EL
        m, n, h = rest_gates()
        # rest: u = 0 only if EL = el_zero; otherwise solve I(u, x_inf(u)) = 0 near 0
        u = 0.0
        for _ in range(50):
            f = self.Iss(u)
            d = (self.Iss(u + 1e-6) - self.Iss(u - 1e-6)) / 2e-6
            u -= f / d
        self.urest = u
        am, bm, an, bn, ah, bh = rates(u)
        self.rest = np.array([u, 0.0, am / (am + bm), an / (an + bn), ah / (ah + bh)])

    def I(self, u, m, n, h):
        return 120 * m ** 3 * h * (u - 115) + 36 * n ** 4 * (u + 12) + 0.3 * (u - self.EL)

    def Iss(self, u):
        am, bm, an, bn, ah, bh = rates(u)
        return self.I(u, am / (am + bm), an / (an + bn), ah / (ah + bh))

    def f(self, t, y, K):
        u, w, m, n, h = y
        am, bm, an, bn, ah, bh = rates(u)
        p = self.phi
        return [w, K * (w + self.I(u, m, n, h)),
                p * (am * (1 - m) - bm * m), p * (an * (1 - n) - bn * n), p * (ah * (1 - h) - bh * h)]

    def jac(self, y, K, eps=1e-7):
        J = np.zeros((5, 5))
        for j in range(5):
            e = np.zeros(5); e[j] = eps
            J[:, j] = (np.array(self.f(0, y + e, K)) - np.array(self.f(0, y - e, K))) / (2 * eps)
        return J

    def eig_rest(self, K):
        lam, V = eig(self.jac(self.rest, K))
        return lam, V

    def unstable_dir(self, K):
        lam, V = self.eig_rest(K)
        i = [k for k in range(5) if lam[k].real > 0]
        if len(i) != 1:
            raise ValueError('rest does not have exactly one unstable eigenvalue: %s' % lam)
        v = V[:, i[0]].real
        if v[0] < 0:
            v = -v
        return lam[i[0]].real, v / abs(v[0])

    def shoot(self, K, delta=1e-6, tmax=60.0, rtol=1e-12, atol=1e-14, dense=False):
        """Integrate from rest + delta * (unstable eigenvector) and classify the escape:
        +1 if u runs away upward (K too small in HH's convention... reported, not assumed),
        -1 if it runs away downward, 0 if neither by tmax."""
        lam, v = self.unstable_dir(K)
        y0 = self.rest + delta * v

        def up(t, y, K): return y[0] - 150.0
        up.terminal = True
        def down(t, y, K): return y[0] + 60.0
        down.terminal = True
        sol = solve_ivp(self.f, (0, tmax), y0, args=(K,), method='DOP853', rtol=rtol, atol=atol,
                        events=(up, down), dense_output=dense)
        if sol.t_events[0].size:
            return +1, sol
        if sol.t_events[1].size:
            return -1, sol
        return 0, sol


def bisect_K(W, Klo, Khi, tol=1e-13, **kw):
    slo, _ = W.shoot(Klo, **kw)
    shi, _ = W.shoot(Khi, **kw)
    if slo == shi or 0 in (slo, shi):
        raise ValueError('no sign change: %s %s' % (slo, shi))
    while Khi - Klo > tol * Khi:
        Km = 0.5 * (Klo + Khi)
        s, _ = W.shoot(Km, **kw)
        if s == 0:
            break
        if s == slo:
            Klo = Km
        else:
            Khi = Km
    return Klo, Khi, slo, shi


def scan(W, Ks, **kw):
    return [(K, W.shoot(K, **kw)[0]) for K in Ks]
