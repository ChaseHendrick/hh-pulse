#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""A C^0 Lohner-type interval Taylor integrator for the Hodgkin-Huxley wave ODE with the speed parameter K carried as
a sixth state variable (K' = 0), in python-flint ball arithmetic. The jets come from hhjet6.py.

Set:  X = xbar + C r0 + B r,  r0 in R0, r in R  (xbar a point of R^6, C 6 x m, B 6 x 6 nearly orthogonal, R0 and R
boxes). One step of length h and order p:
 1. [X] = interval hull of X.
 2. A priori enclosure W:  [X] + [0, h] f(W) inside the interior of W (componentwise). Then every solution from [X]
    exists on [0, h] and stays in W (Picard-Lindeloef; the K component is constant, and its ball is kept as is).
 3. Lagrange remainder, per component: x_i(h) - sum_{k<=p} x_{i,k}(x0) h^k = x_{i,p+1}(x(xi_i)) h^(p+1) with x(xi_i) in
    W, so it lies in Rem_i = h^(p+1) x_{i,p+1}(W).
 4. Mean value form of Phi(x0) = sum_{k<=p} x_k(x0) h^k: Phi(x0) in Phi(xbar) + [J](x0 - xbar), [J] = sum_k h^k
    D x_k([X]), valid because [X] is convex and contains xbar and x0.
 5. y = Phi(xbar) + Rem; xbar' = mid(y); C' = mid([J] C); B' = Q factor of mid([J] B) (column pivoting);
    R' = [B'^-1](y - xbar' + ([J] C - C') R0) + ([B'^-1][J] B) R.
Mixed precision: Phi(xbar) and the linear algebra of step 5 are done at the working precision ctx.prec; W, Rem and
[J] may be computed at a lower precision PREC_AUX. Ball arithmetic at a lower precision returns wider balls that
still contain the exact values, so every enclosure above stays valid; the only effect is a slightly wider R'.
phi and E_l are balls held fixed; the enclosures hold for every value in them.
"""
import math
from flint import arb, arb_mat, ctx
import hhjet6

N = 6          # the default dimension (y, K); a set may carry more constant parameters (y, K, phi)
PREC_AUX = 128


def ball(lo, hi):
    return arb(lo).union(arb(hi))


def matvec(M, v):
    return [sum((M[i, j] * v[j] for j in range(M.ncols())), arb(0)) for i in range(M.nrows())]


def mid_mat(M):
    return arb_mat([[arb(M[i, j].mid()) for j in range(M.ncols())] for i in range(M.nrows())])


def horner(c, x):
    s = arb(0)
    for a in reversed(c):
        s = s * x + a
    return s


def inside(outer, inner):
    return all(o.lower() < i.lower() and i.upper() < o.upper() for o, i in zip(outer, inner))


def qr_orth(A):
    n = A.nrows()
    cols = [[arb(A[i, j].mid()) for i in range(n)] for j in range(n)]
    Q = []
    for c in cols:
        v = list(c)
        for _ in range(2):
            for q in Q:
                dot = sum((v[i] * q[i] for i in range(n)), arb(0))
                v = [arb((v[i] - dot * q[i]).mid()) for i in range(n)]
        nrm = sum((vi * vi for vi in v), arb(0)).sqrt()
        Q.append([arb((vi / nrm).mid()) for vi in v])
    return arb_mat([[Q[j][i] for j in range(n)] for i in range(n)])


class LSet:
    def __init__(self, xbar, C, R0, B, R):
        self.xbar, self.C, self.R0, self.B, self.R = xbar, C, R0, B, R

    def hull(self):
        a = matvec(self.C, self.R0)
        b = matvec(self.B, self.R)
        return [self.xbar[i] + a[i] + b[i] for i in range(len(self.xbar))]

    def affine_image_hull(self, M, shift):
        """hull of { M (x - shift) : x in X } as M (xbar - shift) + (M C) R0 + (M B) R; M is k x 6."""
        MC, MB = M * self.C, M * self.B
        d = [self.xbar[i] - shift[i] for i in range(len(self.xbar))]
        a, b, c = matvec(M, d), matvec(MC, self.R0), matvec(MB, self.R)
        return [a[i] + b[i] + c[i] for i in range(M.nrows())]

    def to_json(self):
        n = len(self.xbar)
        return {'xbar': [ser(v) for v in self.xbar], 'C': [[ser(self.C[i, j]) for j in range(self.C.ncols())]
                                                            for i in range(n)],
                'R0': [ser(v) for v in self.R0], 'B': [[ser(self.B[i, j]) for j in range(n)] for i in range(n)],
                'R': [ser(v) for v in self.R]}

    @staticmethod
    def from_json(d):
        C = d['C']
        return LSet([deser(v) for v in d['xbar']], arb_mat([[deser(v) for v in row] for row in C]),
                    [deser(v) for v in d['R0']], arb_mat([[deser(v) for v in row] for row in d['B']]),
                    [deser(v) for v in d['R']])


def ser(a):
    """Exact serialization of a ball: midpoint and radius as (mantissa, exponent) pairs."""
    m, e = a.mid().man_exp()
    r = arb(a.rad())
    rm, re_ = r.mid().man_exp()
    return [str(m), int(e), str(rm), int(re_)]


def deser(s):
    m, e, rm, re_ = s
    c = arb(int(m)) * arb(2) ** int(e)
    r = arb(int(rm)) * arb(2) ** int(re_)
    return c + arb(0, r) if int(rm) != 0 else c


class StepFailure(Exception):
    pass


class Field:
    def __init__(self, phi, EL, prec_aux=PREC_AUX):
        self.phi, self.EL, self.prec_aux = phi, EL, prec_aux

    def vals(self, x, p):
        return hhjet6.values(x, self.phi, self.EL, p)

    def jet(self, x, p):
        return hhjet6.jet(x, self.phi, self.EL, p)

    def f(self, x):
        return hhjet6.vfield(x, self.phi, self.EL)


def rough_enclosure(F, Xh, vx, h, tries=10):
    """W with Xh + [0, h] f(W) in int W (the first 5 components; K is constant and W[5] = Xh[5])."""
    hint = ball(0, h)
    W = []
    for i in range(5):
        w = horner(vx[i], hint) + (Xh[i] - arb(Xh[i].mid()))
        rad = arb(w.rad()) * arb('0.2') + arb(abs(w).upper()) * arb(2) ** (-(ctx.prec - 20)) + arb(2) ** -400
        W.append(w + ball(-rad, rad))
    W += list(Xh[5:])
    for _ in range(tries):
        try:
            f = F.f(W)
        except (ArithmeticError, ZeroDivisionError, ValueError):
            return None
        cand = [Xh[i] + hint * f[i] for i in range(5)]
        if inside(W[:5], cand):
            return W
        W = [W[i].union(cand[i]) for i in range(5)]
        W = [w + ball(-arb(w.rad()) * arb('0.2'), arb(w.rad()) * arb('0.2')) for w in W] + list(Xh[5:])
    return None


def refine_enclosure(F, vxh, W, h, qs=(3, 6, 12)):
    """Tighter enclosures of the solutions from the hull [X] on [0, h], given a valid one W (all at the current
    precision). By Taylor's theorem with the Lagrange remainder, for t in [0, h] and each component,
    x(t) = sum_{k<=q} x_k(x0) t^k + x_{q+1}(x(xi)) t^(q+1) with x(xi) in W, so
    x(t) in sum_{k<=q} x_k([X]) [0, h]^k + [0, h^(q+1)] x_{q+1}(W); intersected with W it is again valid.
    vxh: Taylor coefficients over the hull [X]."""
    hint = ball(0, h)
    for q in qs:
        vq = F.vals(W, q + 1)
        tq = ball(0, arb(h) ** (q + 1))
        Wn = []
        for i in range(5):
            w = horner(vxh[i][:q + 1], hint) + tq * vq[i][q + 1]
            wi = w.intersection(W[i])
            if not wi.is_finite():
                wi = W[i]
            Wn.append(wi)
        W = Wn + list(W[5:])
    return W


def remainder(F, vxh, W, h, p, nsub):
    """Rem_i = h^(p+1) x_{i,p+1}(x(xi_i)), xi_i in [0, h], enclosed over nsub subintervals [t_j, t_j+1] of [0, h]:
    there x(t) lies in W_j = sum_{k<=p} x_k([X]) [t_j, t_j+1]^k + [0, t_j+1^(p+1)] x_{p+1}(W) (Taylor's theorem again),
    and x(xi_i) lies in the union of the W_j."""
    hA = arb(h)
    vW = F.vals(W, p + 1)
    Rem = [None] * 5
    for j in range(nsub):
        tj = ball(hA * j / nsub, hA * (j + 1) / nsub)
        tp = ball(0, (hA * (j + 1) / nsub) ** (p + 1))
        Wj = [horner(vxh[i][:p + 1], tj) + tp * vW[i][p + 1] for i in range(5)]
        Wj = [Wj[i].intersection(W[i]) for i in range(5)] + list(W[5:])
        vj = F.vals(Wj, p + 1)
        for i in range(5):
            r = vj[i][p + 1]
            Rem[i] = r if Rem[i] is None else Rem[i].union(r)
    return [Rem[i] * hA ** (p + 1) for i in range(5)]


REM_FACTOR = 1e3     # a step whose enclosed remainder exceeds REM_FACTOR * tol is rejected (and h halved)
NSUB = 4


def step(F, X, h, p, vx=None, tol=None):
    """One step; returns (new set, W, Xh), W an enclosure of the solutions from Xh on [0, h]. vx: Taylor coefficients
    at xbar (working precision), if already known. tol: if given, the step is rejected (StepFailure) when the enclosed
    remainder exceeds REM_FACTOR * tol; this only affects the step size, never the validity."""
    if vx is None:
        vx = F.vals(X.xbar, p)
    hA = arb(h)
    y = [horner(vx[i][:p + 1], hA) for i in range(5)] + list(X.xbar[5:])
    Xh = X.hull()
    N = len(X.xbar)
    prec = ctx.prec
    ctx.prec = F.prec_aux
    try:
        W = rough_enclosure(F, Xh, vx, h)
        if W is None:
            raise StepFailure('a priori enclosure')
        try:
            vxh, g = F.jet(Xh, p)
            W = refine_enclosure(F, vxh, W, h)
            Rem = remainder(F, vxh, W, h, p, NSUB)
        except (ArithmeticError, ZeroDivisionError, ValueError):
            raise StepFailure('remainder or jet')
        if tol is not None and max(float(arb(r.abs_upper())) for r in Rem) > REM_FACTOR * tol:
            raise StepFailure('remainder too large')
        J = arb_mat(N, N)
        for i in range(N):
            for m in range(N):
                J[i, m] = horner([g[i][k][m] for k in range(p + 1)], hA)
    finally:
        ctx.prec = prec
    y = [y[i] + Rem[i] for i in range(5)] + y[5:]
    xbar2 = [arb(v.mid()) for v in y]
    JC = J * X.C
    C2 = mid_mat(JC)
    JB = J * X.B
    mJB = mid_mat(JB)
    keys = []
    for j in range(N):
        cn = math.sqrt(sum(float(mJB[i, j].mid()) ** 2 for i in range(N)))
        keys.append(cn * float(arb(X.R[j].rad()).mid()) + 1e-300 * cn)
    order = sorted(range(N), key=lambda j: -keys[j])
    B2 = qr_orth(arb_mat([[mJB[i, j] for j in order] for i in range(N)]))
    B2inv = B2.inv()
    err = [y[i] - xbar2[i] for i in range(N)]
    lin = matvec(JC - C2, X.R0)
    t1 = matvec(B2inv, [err[i] + lin[i] for i in range(N)])
    t2 = matvec(B2inv * JB, X.R)
    return LSet(xbar2, C2, X.R0, B2, [t1[i] + t2[i] for i in range(N)]), W, Xh


def choose_h(vx, p, tol, hmax):
    m = 1e-300
    for i in range(5):
        m = max(m, abs(float(vx[i][p].mid())), abs(float(vx[i][p - 1].mid())) ** (p / (p - 1.0)))
    return min(hmax, (tol / m) ** (1.0 / p))


def step_range(F, Xh, W, h, p, M, shift):
    """Enclosure of M (x(t) - shift) for all t in [0, h] and all x(0) in the box Xh:
    x(t) in sum_{k<=p} x_k(Xh) [0, h]^k + [0, h^(p+1)] x_{p+1}(W). M is k x 6 (its sixth column should be 0)."""
    hint = ball(0, h)
    prec = ctx.prec
    ctx.prec = F.prec_aux
    try:
        v = F.vals(Xh, p)
        vW = F.vals(W, p + 1)
    finally:
        ctx.prec = prec
    rem = [vW[i][p + 1] * ball(0, arb(h) ** (p + 1)) for i in range(5)] + [arb(0)] * (len(Xh) - 5)
    out = []
    for a in range(M.nrows()):
        co = []
        for k in range(p + 1):
            s = sum((M[a, i] * v[i][k] for i in range(5)), arb(0))
            if k == 0:
                s = s - sum((M[a, i] * shift[i] for i in range(5)), arb(0))
            co.append(s)
        out.append(horner(co, hint) + sum((M[a, i] * rem[i] for i in range(5)), arb(0)))
    return out


def integrate(F, X, T_end, p, tol, hmax=0.25, t0=0.0, callback=None, max_steps=100000, log=None, hmin=1e-7):
    """Integrate X from t0 to T_end, or until callback(tp, t, Xprev_hull, X, W, h) returns True. tol may be a number
    or a function of t (float). Step lengths are dyadic with 12 significant bits, so the times are exact."""
    t = arb(t0)
    Tend = arb(T_end)
    ns = 0
    while t < Tend and ns < max_steps:
        tf = float(t.mid())
        tl = tol(tf) if callable(tol) else tol
        vx = F.vals(X.xbar, p)
        h = choose_h(vx, p, tl, hmax)
        e = math.floor(math.log2(h))
        h = math.floor(h / 2.0 ** (e - 12)) * 2.0 ** (e - 12)
        if t + arb(h) > Tend:
            h = float((Tend - t).mid())
            vx = vx
        while True:
            try:
                Xn, W, Xh = step(F, X, h, p, vx, tl)
                break
            except StepFailure:
                h /= 2
                if h < hmin:
                    raise StepFailure('step size underflow at t = %s' % t)
        tp, t = t, t + arb(h)
        X = Xn
        ns += 1
        if log and ns % log == 0:
            hx = X.hull()
            wid = max(float(arb(x.rad()).mid()) for x in hx[:5])
            print('   step %d t = %.5f h = %.2e u = %s max radius %.2e' % (
                ns, float(t.mid()), h, hx[0].str(6, radius=False), wid), flush=True)
        if callback is not None and callback(tp, t, Xh, X, W, h):
            return X, t, ns
    return X, t, ns
