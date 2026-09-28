#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Taylor coefficients of the Hodgkin-Huxley wave flow in the extended state x = (u, w, m, n, h, K), K' = 0, with
their derivatives with respect to the initial point, in ball arithmetic (python-flint arb_series).

The same field and conventions as hhjet.py (u = -V, w = u', t in ms, phi = 3^((T - 6.3)/10), E_l a ball), with two
changes that do not affect rigour:
 * K is a sixth variable, so the jet also gives d/dK of every Taylor coefficient;
 * the Picard iteration x <- x0 + int f(x) is done with growing truncation: pass i (i = 0, 1, ...) works modulo t^(i+1),
   which is enough because after pass i the coefficients of degree <= i + 1 are exact (by induction: f(x) mod t^(i+1)
   depends only on the coefficients of x of degree <= i). Every operation used (sums, products, exp, inverse,
   composition) is an exact identity on power series truncated modulo t^L, evaluated in ball arithmetic, so each
   returned coefficient is an enclosure, over the input balls, of the exact Taylor coefficient and of its gradient.
 * Psi(x) = x / (e^x - 1) near x = 0 is 1/G(x), G(x) = (e^x - 1)/x, composed as G(x0 + s(t)) = sum_k G_k(x0) s^k with
   the Taylor coefficients G_k(x0) of hhseries.g_coeffs (which carry a rigorous tail bound); they are computed once
   per jet call and reused in every pass, since x0 (the constant term) does not change between passes.
"""
from flint import arb, arb_series, ctx
from hhseries import g_coeffs

NV = 6          # derivative directions: u, w, m, n, h, K
NS = 5          # state components that move


class Dual:
    __slots__ = ('v', 'd')

    def __init__(self, v, d):
        self.v, self.d = v, d

    @staticmethod
    def const(c, nd):
        z = arb_series([arb(0)])
        return Dual(arb_series([c]), [z] * nd)

    def _l(self, o):
        return o if isinstance(o, Dual) else Dual.const(arb(o), len(self.d))

    def __add__(self, o):
        if not isinstance(o, Dual):
            return Dual(self.v + o, self.d)
        return Dual(self.v + o.v, [a + b for a, b in zip(self.d, o.d)])
    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.d])

    def __sub__(self, o):
        if not isinstance(o, Dual):
            return Dual(self.v - o, self.d)
        return Dual(self.v - o.v, [a - b for a, b in zip(self.d, o.d)])

    def __rsub__(self, o):
        return Dual(o - self.v, [-a for a in self.d])

    def __mul__(self, o):
        if not isinstance(o, Dual):
            return Dual(self.v * o, [a * o for a in self.d])
        return Dual(self.v * o.v, [a * o.v + self.v * b for a, b in zip(self.d, o.d)])
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            return self * o.inv()
        return Dual(self.v / o, [a / o for a in self.d])

    def exp(self):
        e = self.v.exp()
        return Dual(e, [e * a for a in self.d])

    def inv(self):
        iv = self.v.inv()
        iv2 = -(iv * iv)
        return Dual(iv, [iv2 * a for a in self.d])


def _coeffs(s, L):
    c = s.coeffs()
    return c + [arb(0)] * (L - len(c))


class PsiCache:
    """G_k(x0) for the two Psi arguments, computed once per jet call (x0 fixed)."""

    def __init__(self, Lmax):
        self.Lmax, self.store = Lmax, {}

    def get(self, key, x0):
        if key not in self.store:
            self.store[key] = g_coeffs(x0, self.Lmax + 1)
        return self.store[key]


def psi(x, key, cache):
    """Psi(x) on a Dual x with a real constant term (the same branches as hhjet.psi)."""
    L = ctx.cap
    c = _coeffs(x.v, L)
    x0 = c[0]
    if arb(abs(x0).upper()) <= arb(1) / 2:
        gk = cache.get(key, x0)
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


ALPHA_M_PERTURB = None     # negative controls only: alpha_m -> alpha_m (1 + eps (u - ALPHA_M_CENTER)^2), with the
ALPHA_M_CENTER = 0         # centre the rest value u* (a ball), which leaves rest and the linearization there unchanged


def field(y, phi, EL, cache):
    """The field on Duals; y = (u, w, m, n, h, K) or (u, w, m, n, h, K, phi): with seven entries the temperature factor
    phi is the variable y[6] (tstrip.py) and the argument phi is ignored. Returns the 5 moving components."""
    u, w, m, n, h, K = y[:6]
    if len(y) == 7:
        phi = y[6]
    am = psi((25 - u) / 10, 'm', cache)
    if ALPHA_M_PERTURB is not None:
        du = u - ALPHA_M_CENTER
        am = am * (du * du * arb(ALPHA_M_PERTURB) + 1)
    bm = (u * (-arb(1) / 18)).exp() * 4
    an = psi((10 - u) / 10, 'n', cache) / 10
    bn = (u * (-arb(1) / 80)).exp() / 8
    ah = (u * (-arb(1) / 20)).exp() * (arb(7) / 100)
    bh = (((30 - u) / 10).exp() + 1).inv()
    m2 = m * m
    n2 = n * n
    I = (m2 * m) * h * (u - 115) * 120 + (n2 * n2) * (u + 12) * 36 + (u - EL) * (arb(3) / 10)
    return [w, (w + I) * K,
            (am * (1 - m) - bm * m) * phi, (an * (1 - n) - bn * n) * phi, (ah * (1 - h) - bh * h) * phi]


def jet(x0, phi, EL, p, nd=None):
    """Taylor coefficients (degree <= p) of x(t; x0) and of their derivatives with respect to x0_j, j < nd, for x0 a
    list of 6 arb (u, w, m, n, h, K) or 7 (u, w, m, n, h, K, phi), balls allowed; nd defaults to len(x0), and nd = 0
    gives the values only. Returns vals[i][k] (the parameter rows are constant) and, if nd > 0,
    grads[i][k][j] = d x_{i,k} / d x0_j."""
    n = len(x0)
    if nd is None:
        nd = n
    old = ctx.cap
    cache = PsiCache(p + 2)
    try:
        ctx.cap = 1
        one, zero = arb_series([arb(1)]), arb_series([arb(0)])
        # arb_series carries its own truncation order, so the constant parameters (whose series never change) are
        # created with the full order p + 1; the moving components are replaced in every pass by series of order
        # it + 2.
        ctx.cap = p + 1
        oneF, zeroF = arb_series([arb(1)]), arb_series([arb(0)])
        ypar = [Dual(arb_series([x0[q]]), [oneF if j == q else zeroF for j in range(nd)]) for q in range(NS, n)]
        ctx.cap = 1
        y = [Dual(arb_series([x0[i]]), [one if j == i else zero for j in range(nd)]) for i in range(NS)] + ypar
        for it in range(p):
            ctx.cap = it + 1                       # f(y) is needed modulo t^(it+1)
            F = field(y, phi, EL, cache)
            ctx.cap = it + 2
            new = []
            for i in range(NS):
                fv = _coeffs(F[i].v, it + 1)
                v = arb_series([x0[i]] + [fv[k] / (k + 1) for k in range(it + 1)])
                ds = []
                for j in range(nd):
                    fd = _coeffs(F[i].d[j], it + 1)
                    ds.append(arb_series([arb(1) if j == i else arb(0)] + [fd[k] / (k + 1) for k in range(it + 1)]))
                new.append(Dual(v, ds))
            y = new + ypar
        L = p + 1
        vals = [_coeffs(y[i].v, L) for i in range(NS)] + [[x0[q]] + [arb(0)] * p for q in range(NS, n)]
        if nd == 0:
            return vals, None
        grads = []
        for i in range(NS):
            dl = [_coeffs(y[i].d[j], L) for j in range(nd)]
            grads.append([[dl[j][k] for j in range(nd)] for k in range(L)])
        for q in range(NS, n):
            grads.append([[arb(1) if j == q else arb(0) for j in range(nd)]] + [[arb(0)] * nd for _ in range(p)])
        return vals, grads
    finally:
        ctx.cap = old


def values(x0, phi, EL, p):
    return jet(x0, phi, EL, p, nd=0)[0]


def vfield(x, phi, EL):
    """The field at a point or box x (len(x) components, the parameter components 0)."""
    v = values(x, phi, EL, 1)
    return [v[i][1] for i in range(NS)] + [arb(0)] * (len(x) - NS)


def jacobian(x, phi, EL):
    """Df over the box x (len(x) x len(x), the parameter rows 0)."""
    _, g = jet(x, phi, EL, 1)
    n = len(x)
    return [[g[i][1][j] for j in range(n)] for i in range(n)]
