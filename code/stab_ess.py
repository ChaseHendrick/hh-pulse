#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): (E), the essential spectrum of the linearization about the pulse lies in Re lambda < -delta
(the design note, Section 8.3, not part of this release).

The linearization at rest in the moving frame, in first-order form, is Y' = A_inf(lambda) Y with
A_inf(lambda) = Df(y*, K) + lambda E. Write B for the space-clamped Jacobian at rest (4 x 4, variables u, m, n, h):
B = [[-a, -b^T], [c, -diag(kappa)]], a = dI/du, b = dI/d(m, n, h), c_i = phi dG_i/du, kappa_i = phi (alpha_i + beta_i).
A vector (p, i k p, q) (k real) is an eigenvector of A_inf(lambda) with eigenvalue i k if and only if, with mu = lambda
+ i k and s = k^2/K, (mu I - B + s e1 e1^T)(p, q) = 0: the second row of A_inf gives -k^2 p = K (lambda + a) p + i k K p
+ K b.q, the gate rows (i k + lambda + kappa_i) q_i = c_i p. So lambda belongs to Evans's Sigma(T) (Evans III,
Proposition 1: A_inf(lambda) has a purely imaginary eigenvalue; Corollary 1 is the same dispersion relation) if and only
if Re lambda = Re mu for an eigenvalue mu of B - s e1 e1^T with some s >= 0. The claim therefore is:

    for every s >= 0, every eigenvalue of B - s e1 e1^T has real part < -delta.

Proof method. det(z I - B + s e1 e1^T) = P_B(z) + s Q(z), P_B the characteristic polynomial of B and Q(z) =
prod_i (z + kappa_i) the cofactor of the (1, 1) entry. With z = w - delta this is a monic quartic
w^4 + a3(s) w^3 + a2(s) w^2 + a1(s) w + a0(s) whose coefficients are affine in s. All its roots have negative real part
if and only if a3, a2, a1, a0 > 0, a3 a2 - a1 > 0 and a3 a2 a1 - a1^2 - a3^2 a0 > 0 (the Routh-Hurwitz criterion for
quartics, Teschl 2012, p. 72, eq. (3.45); the paper's Lemma B.4 uses it too). These six quantities are polynomials in s
of degree <= 3. With s = sigma/(1 - sigma), sigma in [0, 1), each becomes (1 - sigma)^(-d) times a polynomial in sigma
of degree <= d; positivity of that polynomial on the closed interval [0, 1] (checked by ball evaluation on an adaptive
subdivision; a piece is accepted only when the ball is positive) gives positivity for all s >= 0.

Negative control: delta = 0.449 must fail (numerically the supremum of the real parts is -0.4486063, the h rate
phi (alpha_h + beta_h) at rest, approached as s -> infinity).

Usage: HH_EL=10.613 python3 stab_ess.py 18.5 [delta]     -> ../data/stab_ess_<tag>.txt
"""
import sys
from flint import arb, arb_mat, arb_poly, ctx
import certify_rest_wave as C
import hhjet6

ctx.prec = 256


def space_clamped(T):
    y, EL = C.rest_state()
    phi = C.phi_of(T)
    J = hhjet6.jacobian(list(y) + [arb(1)], phi, EL)       # K = 1: the w-row is (I_u, 1, I_m, I_n, I_h)
    Bm = arb_mat(4, 4)
    Bm[0, 0] = -J[1][0]
    for j in range(3):
        Bm[0, j + 1] = -J[1][j + 2]
        Bm[j + 1, 0] = J[j + 2][0]
        for i in range(3):
            Bm[j + 1, i + 1] = J[j + 2][i + 2]
    for j in range(3):
        for i in range(3):
            if i != j:
                assert J[j + 2][i + 2] == 0          # the gates are uncoupled
    return Bm, y, EL


def shift(p, d):
    """coefficients of p(w - d) (p an arb_poly)."""
    x = arb_poly([-d, 1])
    out = arb_poly([0])
    xp = arb_poly([1])
    for k in range(p.degree() + 1):
        out += p[k] * xp
        xp *= x
    return out


def hurwitz_polys(Bm, delta):
    """The six Routh-Hurwitz quantities as polynomials in s (arb_poly)."""
    PB = Bm.charpoly()
    Q = arb_poly([1])
    for i in range(1, 4):
        Q *= arb_poly([-Bm[i, i], 1])                  # z - B_ii = z + kappa_i
    Ps, Qs = shift(PB, delta), shift(Q, delta)
    assert PB.degree() == 4 and Q.degree() == 3
    a = [arb_poly([Ps[k], Qs[k] if k <= 3 else 0]) for k in range(5)]        # a_k(s) = Ps_k + s Qs_k
    a0, a1, a2, a3, a4 = a
    assert Ps[4] == 1 and Qs.degree() == 3            # monic quartic in w for every s
    return {'a3': a3, 'a2': a2, 'a1': a1, 'a0': a0, 'D2': a3 * a2 - a1, 'D3': a3 * a2 * a1 - a1 * a1 - a3 * a3 * a0}


def to_sigma(p):
    """(1 - sigma)^d p(sigma/(1 - sigma)), d = deg p: sum_k p_k sigma^k (1 - sigma)^(d - k)."""
    d = p.degree()
    out = arb_poly([0])
    for k in range(d + 1):
        term = arb_poly([p[k]]) * arb_poly([0, 1]) ** k * arb_poly([1, -1]) ** (d - k)
        out += term
    return out


def positive_on_unit(p, maxdepth=40):
    """True if p > 0 on [0, 1], by ball evaluation on an adaptive subdivision; returns (ok, pieces, min lower)."""
    stack = [(arb(0), arb(1), 0)]
    pieces = 0
    lowest = None
    while stack:
        a, b, dep = stack.pop()
        v = p(a.union(b))
        if v > 0:
            pieces += 1
            lo = float(v.lower())
            lowest = lo if lowest is None else min(lowest, lo)
            continue
        if dep >= maxdepth or not p(a) > 0 or not p(b) > 0:
            return False, pieces, lowest
        m = (a + b) / 2
        stack += [(a, m, dep + 1), (m, b, dep + 1)]
    return True, pieces, lowest


def check(T, delta):
    Bm, y, EL = space_clamped(T)
    H = hurwitz_polys(Bm, arb(delta))
    res = {}
    for k, p in H.items():
        res[k] = positive_on_unit(to_sigma(p))
    return all(r[0] for r in res.values()), res, Bm


def main():
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 18.5
    delta = sys.argv[2] if len(sys.argv) > 2 else '0.448'
    ok, res, Bm = check(T, delta)
    lines = ['T = %s C, E_l tag %s; space-clamped Jacobian at rest B:' % (T, C.tag(T))]
    for i in range(4):
        lines.append('  ' + '  '.join(Bm[i, j].str(10) for j in range(4)))
    lines.append('(E) delta = %s: every eigenvalue of B - s e1 e1^T has real part < -delta for every s >= 0: %s' % (
        delta, ok))
    for k, (o, n, lo) in res.items():
        lines.append('    %s(s) > 0 on s >= 0 (in sigma = s/(1+s)): %s, %d pieces, smallest lower bound %.4g' % (
            k, o, n, lo if lo is not None else float('nan')))
    okn, resn, _ = check(T, '0.449')
    failed = [k for k, r in resn.items() if not r[0]]
    lines.append('negative control delta = 0.449 rejected: %s (fails: %s)' % (not okn, ', '.join(failed)))
    allok = ok and not okn
    lines.append('ALL CHECKS PASSED' if allok else 'SOME CHECK FAILED')
    print('\n'.join(lines))
    open('../data/stab_ess_%s.txt' % C.tag(T), 'w').write('\n'.join(lines) + '\n')
    return allok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
