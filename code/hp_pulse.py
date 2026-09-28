#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Numerical (not rigorous): the speed parameter K* of the Hodgkin-Huxley pulse to about 50 digits, by multiple
shooting in high precision, to centre the K interval of the proof (hh_prove_pulse.py).

Unknowns: the states Y_1..Y_N at nodes and K. Equations:
  Y_1 = phi(tau_0; P_K(sigma0), K),   Y_{i+1} = phi(tau_i; Y_i, K),   l . (phi(tau_N; Y_N, K) - y*) = 0,
where P_K is a Taylor parametrization of the unstable manifold of rest (order NP, invariance equation solved order by
order, P_K(sigma) = y* + sigma v + ..., v with u-component 1), l is the left eigenvector of the unstable eigenvalue
(l . v = 1), and phi the flow (Taylor steps with the jets of hhjet6.py at working precision, midpoints only).
The final condition asks the unstable coordinate to vanish at t_end; on the true pulse it is only O(|z_s|^2) there,
which moves K by about that over dz1/dK(t_end), about 1e-50 at t_end = 10 ms. Newton's method, with the Jacobian of
the pieces from the jets; later iterations reuse it (chord method) while the residual keeps shrinking.

Output: data/hp_pulse_<T>.json (K*, the node states, the unstable coordinate history) and a log.
Usage: python3 hp_pulse.py [T] [t_end] [tolerance scale, default 1]
Each iteration is saved to data/logs/hp_pulse_<T>_state_<scale>.json, and a run resumes from that file. A run stops
after 12 iterations; if |dK| has not fallen below 1e-60 by then it exits with status 3 and writes no output, and the
same command resumes it. A run with a scale below 1 (1e-8 at 6.3 C, where the tolerance schedule written for the
growth at 18.5 C is too loose: see the manuscript, Section 5) starts from the converged state of the scale-1 run with new step
sequences, keeps the scale-1 output as data/hp_pulse_<T>_tol1.json and writes its own to data/hp_pulse_<T>.json.
"""
import json
import math
import sys
import time
import numpy as np
from scipy.interpolate import CubicSpline
from flint import arb, arb_mat, arb_series, ctx
import certify_rest_wave as C
import hhjet6
import hhseries

ctx.prec = 256
NP = 12            # order of the unstable-manifold parametrization
SIGMA0 = '1e-6'
P_ORD = 40


def mid(a):
    return arb(a.mid())


def eig_unstable(A, lam0):
    """Newton for (A - lam) v = 0, v[0] = 1, and the left vector l (A^T l = lam l, l.v = 1)."""
    n = 5
    lam = arb(lam0)
    Af = np.array([[float(A[i, j].mid()) for j in range(n)] for i in range(n)])
    w, V = np.linalg.eig(Af)
    i = int(np.argmax(w.real))
    v = [arb(float(x)) for x in (V[:, i].real / V[0, i].real)]
    lam = arb(float(w[i].real))                          # the float eigenvalue starts Newton (lam0 is not used)
    for _ in range(12):
        # F(v, lam) = (A - lam) v, with v[0] fixed to 1; unknowns v[1..4], lam
        r = [sum((A[a, b] * v[b] for b in range(n)), arb(0)) - lam * v[a] for a in range(n)]
        Jm = arb_mat(n, n)
        for a in range(n):
            for b in range(1, n):
                Jm[a, b - 1] = A[a, b] - (lam if a == b else 0)
            Jm[a, 4] = -v[a]
        d = Jm.solve(arb_mat([[x] for x in r]))
        v = [v[0]] + [mid(v[b] - d[b - 1, 0]) for b in range(1, n)]
        lam = mid(lam - d[4, 0])
    wl, VL = np.linalg.eig(Af.T)
    j = int(np.argmax(wl.real))
    l = [arb(float(x)) for x in VL[:, j].real]
    for _ in range(12):
        r = [sum((A[b, a] * l[b] for b in range(n)), arb(0)) - lam * l[a] for a in range(n)]
        M = arb_mat([[A[b, a] - (lam if a == b else 0) for b in range(n)] for a in range(n)])
        # l is determined up to scale: fix l.v = 1 by replacing the last equation
        for b in range(n):
            M[n - 1, b] = v[b]
        r[n - 1] = sum((l[b] * v[b] for b in range(n)), arb(0)) - 1
        d = M.solve(arb_mat([[x] for x in r]))
        l = [mid(l[b] - d[b, 0]) for b in range(n)]
    return lam, v, l


def manifold(K, phi, EL, y, A, lam, v):
    """Coefficients a_k (k = 0..NP) of P(sigma) = sum a_k sigma^k, a_0 = y*, a_1 = v, (k lam - A) a_k = N_k."""
    a = [list(y), list(v)]
    n = 5
    for k in range(2, NP + 1):
        old = ctx.cap
        ctx.cap = k + 1
        try:
            ser = [arb_series([a[j][i] for j in range(k)] + [arb(0)]) for i in range(n)]
            F = hhseries.field_series(ser, K, phi, EL)
            Nk = [F[i].coeffs()[k] if len(F[i].coeffs()) > k else arb(0) for i in range(n)]
        finally:
            ctx.cap = old
        M = arb_mat([[(k * lam if i == j else 0) - A[i, j] for j in range(n)] for i in range(n)])
        s = M.solve(arb_mat([[x] for x in Nk]))
        a.append([mid(s[i, 0]) for i in range(n)])
    return a


def p_of_sigma(a, sigma):
    return [sum((a[k][i] * sigma ** k for k in range(len(a))), arb(0)) for i in range(5)]


def dyadic_step(h):
    """h rounded down to 12 significant bits (an exact binary fraction with a short expansion)."""
    e = math.floor(math.log2(h))
    return math.floor(h / 2.0 ** (e - 12)) * 2.0 ** (e - 12)


def flow(x, K, tau, phi, EL, tol, want_jac=True, hmax=0.2, steps=None, order=None):
    """Integrate (x, K) for time tau (a float) with Taylor steps of order `order` (default P_ORD), keeping the time
    exactly: every step but the last has a 12-bit dyadic length, the elapsed time is summed exactly in arb, and the
    last step is tau minus that sum, so the flow is evaluated at time tau exactly (up to the working precision).
    Returns the end state and, if want_jac, the 5 x 6 Jacobian d x(tau) / d (x0, K). If steps is a list that is not
    empty, those step lengths are used (a fixed discretization, so that Newton's method solves one fixed discrete
    problem; the list stores the dyadic lengths and the last one is recomputed as tau minus the sum); if it is an
    empty list, the adaptive step lengths are appended to it."""
    p = order or P_ORD
    xs = [mid(c) for c in x] + [K]
    Phi = arb_mat([[1 if i == j else 0 for j in range(6)] for i in range(6)])
    t = arb(0)
    tauA = arb(tau)
    fixed = list(steps[:-1]) if steps else None
    record = steps is not None and not steps
    while True:
        if fixed is not None:
            if fixed:
                hA = arb(fixed.pop(0))
                last = False
            else:
                hA, last = tauA - t, True
        if want_jac:
            vals, g = hhjet6.jet(xs, phi, EL, p)
        else:
            vals = hhjet6.values(xs, phi, EL, p)
        if fixed is None:
            m = 1e-300
            for i in range(5):
                m = max(m, abs(float(vals[i][p].mid())), abs(float(vals[i][p - 1].mid())) ** (p / (p - 1.0)))
            tf = float(t.mid())
            tl = tol(tf) if callable(tol) else tol
            h = dyadic_step(min(hmax, (tl / m) ** (1.0 / p)))
            if t + arb(h) >= tauA:
                hA, last = tauA - t, True
            else:
                hA, last = arb(h), False
            if record:
                steps.append(float(hA.mid()))
        if want_jac:
            J = arb_mat(6, 6)
            for i in range(6):
                for j in range(6):
                    J[i, j] = mid(sum((g[i][k][j] * hA ** k for k in range(p + 1)), arb(0)))
            Phi = J * Phi
        xs = [mid(sum((vals[i][k] * hA ** k for k in range(p + 1)), arb(0))) for i in range(5)] + [K]
        t = t + hA
        if last:
            break
    if want_jac:
        return xs[:5], arb_mat([[mid(Phi[i, j]) for j in range(6)] for i in range(5)])
    return xs[:5], None


def main():
    T = C.temperature(sys.argv[1]) if len(sys.argv) > 1 else '18.5'     # a decimal string: phi is exact for it
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    phi = C.phi_of(T)
    y, EL = C.rest_state()
    d = np.load('../data/pulse_%s.npz' % T)                  # the zero-current profile, as the initial guess
    tp, Yp, K0 = d['t'], d['Y'], float(d['K'])
    import hhwave
    shift = hhwave.Wave(float(T), EL=float(C.HH_EL)).rest - hhwave.Wave(float(T)).rest if C.HH_EL else np.zeros(5)
    K = arb(K0)
    sigma0 = arb(SIGMA0)
    # start time in profile time: u_prof = sigma0 (log-linear interpolation on the early exponential part)
    iu = np.where(Yp[0] > float(SIGMA0))[0][0]
    ta, tb = tp[iu - 1], tp[iu]
    ua, ub = Yp[0, iu - 1], Yp[0, iu]
    t_s = ta + (math.log(float(SIGMA0)) - math.log(ua)) / (math.log(ub) - math.log(ua)) * (tb - ta)
    # nodes: every DT ms from t_s to t_end (profile time), DT = 0.25 ms at 18.5 C and scaled by 10.89/lambda_u
    # otherwise (about 2.7 units of unstable growth per piece)
    lam_f = float(eig_unstable(C.jac(y, K, phi, EL), 10.0)[0].mid())
    DT = 0.25 * 10.8923 / lam_f
    nodes = [t_s]
    while nodes[-1] + DT < t_end - 1e-9:
        nodes.append(nodes[-1] + DT)
    nodes.append(t_end)
    taus = [nodes[i + 1] - nodes[i] for i in range(len(nodes) - 1)]
    Nn = len(taus)                                   # pieces; unknown states Y_1..Y_{Nn-1}... see below
    spl = [CubicSpline(tp, Yp[i] + shift[i]) for i in range(5)]
    Y = [[arb(float(spl[i](nodes[j]))) for i in range(5)] for j in range(1, Nn)]
    log = open('../data/hp_pulse_%s.log' % C.tag(T), 'w')

    def say(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()
    say('T = %s C, t_end = %s ms (profile time), %d pieces, sigma0 = %s, NP = %d, order %d, %d bits' % (
        T, t_end, Nn, SIGMA0, NP, P_ORD, ctx.prec))

    def tol_of(t_abs):
        # local error budget: errors at profile time t move K by about err / S_K(t), S_K(t) ~ 200 exp(lambda_u t)
        return 1e-58 * max(1.0, 200 * math.exp(min(lam_f * t_abs, 600)))

    Jcache = None
    rhist = []
    stepseq = [[] for _ in range(Nn)]
    TOLSCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    state_file = '../data/logs/hp_pulse_%s_state_%s.json' % (C.tag(T), TOLSCALE)
    import os
    seed_file = '../data/logs/hp_pulse_%s_state_%s.json' % (C.tag(T), 1.0)
    os.makedirs('../data/logs', exist_ok=True)
    if os.path.exists(state_file):
        st = json.load(open(state_file))
        K = arb(st['K'])
        Y = [[arb(c) for c in row] for row in st['Y']]
        stepseq = st['steps']
        say('resumed from %s' % state_file)
    elif TOLSCALE < 1.0:
        if not os.path.exists('../data/hp_pulse_%s.json' % C.tag(T)):
            sys.exit('run the scale-1 computation to convergence first')
        st = json.load(open(seed_file))
        K = arb(st['K'])
        Y = [[arb(c) for c in row] for row in st['Y']]
        say('started from the converged scale-1 state %s, new step sequences' % seed_file)
    converged = False
    for it in range(12):
        t0 = time.time()
        full = Jcache is None
        A = C.jac(y, K, phi, EL)
        lam, v, l = eig_unstable(A, lam_f)
        a = manifold(K, phi, EL, y, A, lam, v)
        p0 = p_of_sigma(a, sigma0)
        # d p0 / dK by a difference quotient (the dependence is weak: p0 moves by about sigma0 * dv/dK)
        dK = arb('1e-40')
        A2 = C.jac(y, K + dK, phi, EL)
        lam2, v2, _ = eig_unstable(A2, lam_f)
        a2 = manifold(K + dK, phi, EL, y, A2, lam2, v2)
        dp0 = [(x2 - x1) / dK for x1, x2 in zip(p0, p_of_sigma(a2, sigma0))]
        starts = [p0] + Y
        ends, jacs = [], []
        for j in range(Nn):
            tt = nodes[j]
            e, Jp = flow(starts[j], K, taus[j], phi, EL, lambda s, tt=tt: TOLSCALE * tol_of(tt + s), want_jac=full,
                         steps=stepseq[j])
            ends.append(e)
            jacs.append(Jp)
        # residuals
        res = []
        for j in range(Nn - 1):
            res += [ends[j][i] - Y[j][i] for i in range(5)]
        z1 = sum((l[i] * (ends[-1][i] - y[i]) for i in range(5)), arb(0))
        res.append(z1)
        rn = max(abs(float(r.mid())) for r in res)
        if full:
            n = 5 * (Nn - 1) + 1
            Jm = arb_mat(n, n)
            # piece 0: Y_1 = phi(p0(K), K): d/dK = Phi_x dp0 + Phi_K
            for i in range(5):
                Jm[i, n - 1] = sum((jacs[0][i, b] * dp0[b] for b in range(5)), arb(0)) + jacs[0][i, 5]
                Jm[i, i] = -1
            for j in range(1, Nn - 1):
                for i in range(5):
                    for b in range(5):
                        Jm[5 * j + i, 5 * (j - 1) + b] = jacs[j][i, b]
                    Jm[5 * j + i, 5 * j + i] = -1
                    Jm[5 * j + i, n - 1] = jacs[j][i, 5]
            for b in range(5):
                Jm[n - 1, 5 * (Nn - 2) + b] = sum((l[i] * jacs[-1][i, b] for i in range(5)), arb(0))
            Jm[n - 1, n - 1] = sum((l[i] * jacs[-1][i, 5] for i in range(5)), arb(0))
            Jcache = Jm
        dx = Jcache.solve(arb_mat([[r] for r in res]))
        n = 5 * (Nn - 1) + 1
        Y = [[mid(Y[j][i] - dx[5 * j + i, 0]) for i in range(5)] for j in range(Nn - 1)]
        dKn = dx[n - 1, 0]
        K = mid(K - dKn)
        rhist.append(rn)
        json.dump({'K': K.str(80, radius=False), 'Y': [[c.str(80, radius=False) for c in row] for row in Y],
                   'steps': stepseq}, open(state_file, 'w'))
        say('iter %d (%s): max residual %.3e, |dK| %.3e, K = %s  (%.0f s)' % (
            it, 'Newton' if full else 'chord', rn, abs(float(dKn.mid())), K.str(60, radius=False), time.time() - t0))
        if (len(rhist) >= 2 and rhist[-1] > 0.1 * rhist[-2]) or it == 2:
            Jcache = None                                   # refresh once after two chord steps, or on a stall
        if abs(float(dKn.mid())) < 1e-60:
            converged = True
            break
    if not converged:
        say('not converged after 12 iterations; run the same command again to resume')
        sys.exit(3)
    # history of the unstable coordinate along the final orbit at the nodes
    out = {'T': float(T), 'phi': phi.str(60), 't_end': t_end, 'K': K.str(70, radius=False), 'sigma0': SIGMA0,
           'NP': NP, 'order': P_ORD,
           'prec': ctx.prec, 'nodes_profile_time': nodes, 'residual_history': rhist,
           'Y': [[c.str(70, radius=False) for c in row] for row in Y],
           'p0': [c.str(70, radius=False) for c in p0], 'lambda_u': lam.str(40, radius=False)}
    out['tol_scale'] = TOLSCALE
    out['steps_per_piece'] = [len(q) for q in stepseq]
    target = '../data/hp_pulse_%s.json' % C.tag(T)
    if TOLSCALE < 1.0:
        os.replace(target, '../data/hp_pulse_%s_tol1.json' % C.tag(T))
    elif TOLSCALE > 1.0:
        target = '../data/hp_pulse_%s_tol%g.json' % (C.tag(T), TOLSCALE)
    json.dump(out, open(target, 'w'), indent=1)
    say('K* = %s' % K.str(60, radius=False))


if __name__ == '__main__':
    main()
