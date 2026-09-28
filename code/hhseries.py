#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Taylor series of the Hodgkin-Huxley travelling-wave field in python-flint (arb_series).

The field is that of hhwave.py: y = (u, w, m, n, h), u' = w, w' = K (w + I), x' = phi (alpha_x (1 - x) - beta_x x).
Psi(x) = x / (e^x - 1) is evaluated on a series x(t) = x0 + s(t) either as x / (e^x - 1) when the constant term is
away from 0, or as 1 / G(x) with G(x) = (e^x - 1)/x = sum_n x^n/(n+1)!, whose Taylor coefficients at x0 are
G_k(x0) = sum_{n>=k} C(n, k) x0^(n-k) / (n+1)!, composed with s. With RIGOROUS = True the G_k carry a bound of the
tail n > N (|x0| <= 1/2: term ratio at most 1/2), so every coefficient is an enclosure.

This module is used by the rigorous integrator (lohner6.py, through hhjet6.py), which keeps the balls.
"""
from flint import arb, arb_series, ctx, fmpq

NTERMS = 400


def g_coeffs(x0, L, N=NTERMS):
    """Enclosures of G_k(x0), k < L, for a real ball x0 with |x0| <= 1/2."""
    r = abs(x0).upper() if hasattr(abs(x0), 'upper') else None
    r = arb(abs(x0).upper())
    if not r <= arb(1) / 2:
        raise ArithmeticError('g_coeffs needs |x0| <= 1/2')
    out = []
    fact = [arb(1)]
    for n in range(1, N + 3):
        fact.append(fact[-1] * n)
    for k in range(L):
        s = arb(0)
        binom = arb(1)
        xp = arb(1)
        for n in range(k, N + 1):
            if n > k:
                binom = binom * n / (n - k)
                xp = xp * x0
            s += binom * xp / fact[n + 1]
        # tail: term_n = C(n,k) r^(n-k)/(n+1)!; ratio term_{n+1}/term_n = (n+1) r / ((n+1-k)(n+2)) <= 1/2 for n >= N
        # when N >= 2k + 2 and r <= 1/2. Bound the tail by twice the first omitted term.
        if not N >= 2 * k + 2:                     # the tail bound below needs it (not an assert: python -O)
            raise ArithmeticError('Psi series: N = %d < 2k + 2 = %d' % (N, 2 * k + 2))
        bN = arb(1)
        for j in range(k):
            bN = bN * (N + 1 - j) / (j + 1)
        first = bN * r ** (N + 1 - k) / fact[N + 2]
        s += arb(0, (2 * first).upper())
        out.append(s)
    return out


def psi_series(x, L):
    """Psi(x(t)) for an arb_series x of length L."""
    c = x.coeffs()
    c += [arb(0)] * (L - len(c))
    x0 = c[0]
    if abs(x0) > arb(1) / 4:                                  # constant term away from 0 as a ball
        e = x.exp()
        return x * (e - 1).inv()
    gk = g_coeffs(x0, L)
    if all(ci == 0 for ci in c[1:L]):                  # constant series (flint refuses to compose with 0)
        G = arb_series([gk[0]])
    else:
        G = arb_series(gk)(arb_series([arb(0)] + c[1:L]))
    return G.inv()


def field_series(y, K, phi, EL):
    """The field applied to a list of 5 arb_series; returns 5 arb_series."""
    u, w, m, n, h = y
    L = ctx.cap
    am = psi_series((25 - u) / 10, L)
    bm = (u * (-arb(1) / 18)).exp() * 4
    an = psi_series((10 - u) / 10, L) / 10
    bn = (u * (-arb(1) / 80)).exp() / 8
    ah = (u * (-arb(1) / 20)).exp() * (arb(7) / 100)
    bh = (((30 - u) / 10).exp() + 1).inv()
    I = m ** 3 * h * (u - 115) * 120 + n ** 4 * (u + 12) * 36 + (u - EL) * (arb(3) / 10)
    return [w, (w + I) * K,
            (am * (1 - m) - bm * m) * phi, (an * (1 - n) - bn * n) * phi, (ah * (1 - h) - bh * h) * phi]


def taylor(y0, K, phi, EL, p):
    """Taylor coefficients to degree p of the solution through the point (list of arb) y0, by Picard iteration on
    series. Returns a list of 5 coefficient lists of length p + 1. Valid for balls y0, K, phi, EL: the result encloses
    the Taylor coefficients of every solution with data in the balls (each iteration is an exact identity on the
    truncated series, and after p iterations the first p + 1 coefficients are fixed)."""
    old = ctx.cap
    ctx.cap = p + 1
    try:
        y = [arb_series([c]) for c in y0]
        for _ in range(p):
            F = field_series(y, K, phi, EL)
            y = [arb_series([y0[i]]) + F[i].integral() for i in range(5)]
        out = []
        for s in y:
            c = s.coeffs()
            out.append(c + [arb(0)] * (p + 1 - len(c)))
        return out
    finally:
        ctx.cap = old
