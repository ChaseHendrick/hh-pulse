#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Taylor coefficients of the Hodgkin-Huxley wave flow together with their derivatives with respect to the initial
point, in ball arithmetic (python-flint arb_series).

A Dual is (v, [d_1..d_5]) with v and d_j arb_series in t: the value of a quantity along the solution x(t; x0) and
its partial derivatives with respect to x0_j. Picard iteration on truncated series,
    x <- x0 + int f(x),        D x <- I + int Df(x) D x,
done as one iteration on Duals, fixes one more coefficient per pass. With balls for x0 every coefficient is an
enclosure, over the ball, of the exact Taylor coefficient and of its gradient, because every operation below is an
exact identity on truncated series evaluated in ball arithmetic.
"""
from flint import arb, arb_series, ctx
from hhseries import g_coeffs

NV = 5


class Dual:
    __slots__ = ('v', 'd')

    def __init__(self, v, d):
        self.v, self.d = v, d

    @staticmethod
    def const(c):
        return Dual(arb_series([c]), [arb_series([arb(0)])] * NV)

    def _l(self, o):
        return o if isinstance(o, Dual) else Dual.const(arb(o))

    def __add__(self, o):
        o = self._l(o)
        return Dual(self.v + o.v, [a + b for a, b in zip(self.d, o.d)])
    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.d])

    def __sub__(self, o):
        return self + (-self._l(o))

    def __rsub__(self, o):
        return self._l(o) + (-self)

    def __mul__(self, o):
        if not isinstance(o, Dual):
            return Dual(self.v * o, [a * o for a in self.d])
        return Dual(self.v * o.v, [a * o.v + self.v * b for a, b in zip(self.d, o.d)])
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            return self * o.inv()
        return Dual(self.v / o, [a / o for a in self.d])

    def __pow__(self, n):
        r = self
        for _ in range(n - 1):
            r = r * self
        return r

    def exp(self):
        e = self.v.exp()
        return Dual(e, [e * a for a in self.d])

    def inv(self):
        iv = self.v.inv()
        iv2 = -(iv * iv)
        return Dual(iv, [iv2 * a for a in self.d])


def _const_term(s):
    c = s.coeffs()
    return c[0] if c else arb(0)


def psi(x):
    """Psi(x) = x / (e^x - 1) on a Dual x whose value has a real constant term."""
    L = ctx.cap
    x0 = _const_term(x.v)
    if arb(abs(x0).upper()) <= arb(1) / 2:
        c = x.v.coeffs() + [arb(0)] * L
        gk = g_coeffs(x0, L + 1)
        tail = c[1:L]
        if all(ci == 0 for ci in tail):
            G = arb_series([gk[0]])
            Gp = arb_series([gk[1]])
        else:
            s = arb_series([arb(0)] + tail)
            G = arb_series(gk[:L])(s)
            Gp = arb_series([gk[k + 1] * (k + 1) for k in range(L)])(s)
        Gd = Dual(G, [Gp * a for a in x.d])
        return Gd.inv()
    if x0.contains(0):
        raise ArithmeticError('Psi on a wide ball containing 0')
    return x * (x.exp() - 1).inv()


def field(y, K, phi, EL):
    u, w, m, n, h = y
    am = psi((25 - u) / 10)
    bm = (u * (-arb(1) / 18)).exp() * 4
    an = psi((10 - u) / 10) / 10
    bn = (u * (-arb(1) / 80)).exp() / 8
    ah = (u * (-arb(1) / 20)).exp() * (arb(7) / 100)
    bh = (((30 - u) / 10).exp() + 1).inv()
    I = m ** 3 * h * (u - 115) * 120 + n ** 4 * (u + 12) * 36 + (u - EL) * (arb(3) / 10)
    return [w, (w + I) * K,
            (am * (1 - m) - bm * m) * phi, (an * (1 - n) - bn * n) * phi, (ah * (1 - h) - bh * h) * phi]


def jet(x0, K, phi, EL, p):
    """Taylor coefficients (degree <= p) of x(t; x0) and of D_x0 x(t; x0), for x0 a list of 5 arb (balls allowed).
    Returns vals[i][k] and grads[i][k][j] = d x_{i,k} / d x0_j."""
    old = ctx.cap
    ctx.cap = p + 1
    try:
        one, zero = arb_series([arb(1)]), arb_series([arb(0)])
        y = [Dual(arb_series([x0[i]]), [one if j == i else zero for j in range(NV)]) for i in range(NV)]
        for _ in range(p):
            F = field(y, K, phi, EL)
            y = [Dual(arb_series([x0[i]]) + F[i].v.integral(),
                      [(one if j == i else zero) + F[i].d[j].integral() for j in range(NV)]) for i in range(NV)]

        def pad(s):
            c = s.coeffs()
            return c + [arb(0)] * (p + 1 - len(c))
        vals = [pad(y[i].v) for i in range(NV)]
        dl = [[pad(y[i].d[j]) for j in range(NV)] for i in range(NV)]
        grads = [[[dl[i][j][k] for j in range(NV)] for k in range(p + 1)] for i in range(NV)]
        return vals, grads
    finally:
        ctx.cap = old


def jacobian(x, K, phi, EL):
    """Df over the box x (list of arb): the degree-1 gradient coefficients of the jet."""
    _, g = jet(x, K, phi, EL, 1)
    return [[g[i][1][j] for j in range(NV)] for i in range(NV)]
