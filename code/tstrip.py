#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): Hodgkin-Huxley pulses for every temperature in an interval [T_lo, T_hi], by a chain of
windows (h-sets with one exit direction) along a reference pulse, with the temperature factor phi = 3^((T - 6.3)/10)
and the speed parameter carried as variables. Design and argument: REPORT.md, Section 8.1.

Parameters. phi in [phi_lo, phi_hi] = phi([T_lo, T_hi]); K = K_c + a (phi - phi_c) + s, s in [-sigma, sigma]. In the
Lohner sets they are the coordinates p1 = (phi - phi_m) / dphi and p2 = s / sigma in [-1, 1] (phi_m the midpoint,
dphi the half-width), so the set is (u, w, m, n, h, K, phi) = xbar + C r0 + B r with r0 in [-1, 1]^7.

Windows. W_i = { c_i + F_i D_i q + g_i p1 + h_i p2 : q in [-1, 1]^5, p in [-1, 1]^2 }, F_i a frame whose first column
is the exit direction, D_i = diag(w_i, s_i1, ..., s_i4), g_i, h_i shears with zero exit component. W_0 is the exit set
of Lemma B (exit coordinate fixed at z1 = r_B). Checks, for i = 0, ..., m-1, from one Lohner run of W_i over
[t_i, t_{i+1}] (the set is valid for each value of r0, so the images of the faces q_1 = +-1 are read off the same run):
 (S) the entry coordinates of the image lie strictly inside those of W_{i+1};
 (X) (i >= 1) the image of the face q_1 = +1 has exit coordinate > 0 in W_{i+1}, that of q_1 = -1 has < 0;
 and for the last stage (F) the image of W_{m-1} lies in the interior of the closing block B0.
W_{i+1} is defined from the stage-i image (entry widths = the image's entry extent times 1 + MARGIN, shears = the
image's entry dependence on p1 and p2, w_{i+1} = 0.95 times the smaller exit distance of the two face images), so (S)
holds by construction and (X) is checked as w_{i+1} > 0. (P) The endpoint runs: for p2 = +1 (resp. -1) and every p1
the orbit from the exit set reaches exit coordinate > w_i (resp. < -w_i) at some stage, and never the other side before.
With Lemmas A and B, the transversality and the block conditions, all over the parameter box, the argument of
REPORT 8.1 gives a pulse for every T in [T_lo, T_hi].

Usage:
  python3 tstrip.py reference T_c r_B T_enter_profile          # numerical reference pulse and stage times (256 bits)
  python3 tstrip.py piece T_lo T_hi T_c sigma [shift]         # the rigorous checks on one piece -> data/tstrip_*.json
  python3 tstrip.py piece ... ; with TSTRIP_NEG=model the alpha_m perturbation of the negative control is switched on
The printed leak potential is used when HH_EL is set (as everywhere in this folder).
"""
import json
import math
import os
import sys
import time
from flint import arb, arb_mat, ctx
import certify_rest_wave as C
import hhjet6
import lohner6 as L
import block0

PREC = 128
ORDER = 24
TOL = 1e-22
MARGIN = 1e-3
DATA = '../data'
ctx.prec = PREC


def ball(lo, hi):
    return arb(lo).union(arb(hi))


def phi_of(T):
    return arb(3) ** ((arb(T) - arb('6.3')) / 10)


def mid(a):
    return arb(a.mid())


def midmat(M):
    return arb_mat([[mid(M[i, j]) for j in range(M.ncols())] for i in range(M.nrows())])


def ref_path(Tc):
    return '%s/tstrip_ref_%s.json' % (DATA, C.tag(Tc))


# ---------------------------------------------------------------------------------------------------------------
# numerical reference (not rigorous): the pulse at T_c from hp_pulse's multiple-shooting nodes, stage times, frames

def reference(Tc, rB, Tenter_prof):
    import hp_pulse as HP
    ctx.prec = 256
    phi = C.phi_of(Tc)
    ystar, EL = C.rest_state()
    d = json.load(open('%s/hp_pulse_%s.json' % (DATA, C.tag(Tc))))
    K = arb(d['K'])
    nodes, Y, p0 = d['nodes_profile_time'], [[arb(v) for v in row] for row in d['Y']], [arb(v) for v in d['p0']]
    starts = [p0] + Y
    A0 = C.jac(ystar, K, phi, EL)
    import hh_prove_pulse as PP
    TB = PP.lemmaB_T(A0)
    TBi = TB.inv()

    def z1(x):
        return sum((TB[0, j] * (x[j] - ystar[j]) for j in range(5)), arb(0))

    def state_at(tp):
        """the reference pulse at profile time tp (flow from the last node before tp)"""
        j = max(k for k in range(len(nodes) - 1) if nodes[k] <= tp + 1e-12)
        x, _ = HP.flow(starts[j], K, tp - nodes[j], phi, EL, 1e-60, want_jac=False, hmax=0.02) if tp > nodes[j] else \
            (starts[j], None)
        return x
    # crossing of z1 = r_B (bisection in profile time, numerical)
    a_, b_ = nodes[0], nodes[0] + 1.0
    for _ in range(60):
        m_ = (a_ + b_) / 2
        if float(z1(state_at(m_)).mid()) < rB:
            a_ = m_
        else:
            b_ = m_
    tx = b_
    # stage times: the exit direction grows by about e^1.1 per stage, 0.002 <= dt <= 0.25 ms
    F = arb_mat([[mid(TBi[i, j]) for j in range(5)] for i in range(5)])
    t, times, centres, frames = 0.0, [0.0], [state_at(tx)], [F]
    Tend = Tenter_prof - tx
    while t < Tend - 2.0 ** -12:
        dt = 0.1
        x0 = centres[-1]
        while True:
            dt = min(dt, Tend - t)
            x1, J = HP.flow(x0, K, dt, phi, EL, 1e-60, want_jac=True, hmax=0.02)
            G = arb_mat([[J[i, j] for j in range(5)] for i in range(5)]) * frames[-1]
            grow = math.sqrt(sum(float(G[i, 0].mid()) ** 2 for i in range(5))) / \
                math.sqrt(sum(float(frames[-1][i, 0].mid()) ** 2 for i in range(5)))
            if (grow <= math.exp(1.2) or dt <= 0.002) or dt < 1e-9:
                break
            dt = max(0.002, dt * 1.1 / math.log(grow))
        dt = max(2.0 ** -12, math.floor(dt * 4096) / 4096.0)      # stage lengths are multiples of 2^-12 ms (exact)
        if t + dt > Tend:
            dt = max(2.0 ** -12, math.floor((Tend - t) * 4096) / 4096.0)
        x1, J = HP.flow(x0, K, dt, phi, EL, 1e-60, want_jac=True, hmax=0.02)
        G = arb_mat([[J[i, j] for j in range(5)] for i in range(5)]) * frames[-1]
        t += dt
        x1 = state_at(tx + t)                  # re-centre on the pulse (from the nodes), not on the drifting flow
        Q = L.qr_orth(midmat(G))
        times.append(t)
        centres.append(x1)
        frames.append(Q)
        print('reference: stage %d t = %.5f dt = %.4f growth %.2f u = %.4f' % (len(times) - 1, t, dt, grow,
                                                                              float(x1[0].mid())), flush=True)
    out = {'T_c': Tc, 'tag': C.tag(Tc), 'K_c': d['K'], 'r_B': rB, 'profile_time_of_exit': tx, 'T_enter_profile':
           Tenter_prof, 'times': times, 'centres': [[L.ser(mid(v)) for v in c] for c in centres],
           'frames': [[[L.ser(Fm[i, j]) for j in range(5)] for i in range(5)] for Fm in frames],
           'TB': [[L.ser(TB[i, j]) for j in range(5)] for i in range(5)]}
    json.dump(out, open(ref_path(Tc), 'w'))
    print('reference written: %d stages, T_enter = %.4f after the exit' % (len(times) - 1, Tend))


# ---------------------------------------------------------------------------------------------------------------
# rigorous part

class Piece:
    def __init__(self, T_lo, T_hi, Tc, sigma, shift=0.0):
        ctx.prec = PREC
        self.ref = json.load(open(ref_path(Tc)))
        self.T_lo, self.T_hi, self.Tc = T_lo, T_hi, Tc
        self.phi_lo, self.phi_hi, self.phi_c = phi_of(T_lo), phi_of(T_hi), phi_of(Tc)
        self.phi_m = mid((self.phi_lo + self.phi_hi) / 2)
        self.dphi = arb(((self.phi_hi - self.phi_lo) / 2).upper())
        # the half-width is rounded up and the midpoint exact, so phi_m + dphi [-1, 1] covers [phi_lo, phi_hi]
        assert bool(ball(self.phi_m - self.dphi, self.phi_m + self.dphi).contains(self.phi_lo.union(self.phi_hi)))
        self.phi_ball = self.phi_lo.union(self.phi_hi)
        self.K_c = mid(arb(self.ref['K_c'])) + arb(shift)
        self.a = arb(SLOPE.get(C.tag(Tc), '2.1017'))
        self.sigma = arb(sigma)
        Kmid = mid(self.K_c + self.a * (self.phi_m - self.phi_c))
        self.K_mid = Kmid
        self.K_ball = (Kmid + arb(0, (abs(self.a) * self.dphi + self.sigma).upper()))
        self.ystar, self.EL = C.rest_state()
        self.rB = arb(self.ref['r_B'])
        self.TB = arb_mat([[L.deser(v) for v in row] for row in self.ref['TB']])
        self.F = L.Field(None, self.EL, prec_aux=PREC)

    def param_rows(self):
        """rows (K, phi) of the Lohner C matrix for the columns (p1, p2)"""
        return [[self.a * self.dphi, self.sigma], [self.dphi, arb(0)]]

    def lohner_set(self, centre, cols, R=None):
        """xbar = (centre, K_mid, phi_m); C = [cols (5 x 5 state columns scaled) | parameter columns]"""
        xbar = [mid(v) for v in centre] + [self.K_mid, self.phi_m]
        Cm = arb_mat(7, 7)
        for i in range(5):
            for j in range(7):
                Cm[i, j] = cols[i][j]
        pr = self.param_rows()
        for j in range(2):
            Cm[5, 5 + j] = pr[0][j]
            Cm[6, 5 + j] = pr[1][j]
        Bm = arb_mat([[1 if i == j else 0 for j in range(7)] for i in range(7)])
        Rr = R if R is not None else [arb(0)] * 7
        return L.LSet(xbar, Cm, [ball(-1, 1)] * 7, Bm, Rr)


SLOPE = {}          # numerical dK*/dphi by tag; the secant between the 6.3 and 18.5 C speeds is used otherwise


def coords(X, Fi_inv, c, g, h):
    """Affine enclosure of the window coordinates q = F^-1 (x - c - g p1 - h p2) of the set X:
    q in qbar + A r0 + Bq r, with r0 in [-1, 1]^7 (p1 = r0[5], p2 = r0[6]) and r in X.R. Returns (qbar, A, Bq)."""
    Cs = arb_mat([[X.C[i, j] for j in range(7)] for i in range(5)])
    Bs = arb_mat([[X.B[i, j] for j in range(7)] for i in range(5)])
    qbar = L.matvec(Fi_inv, [X.xbar[i] - c[i] for i in range(5)])
    A = Fi_inv * Cs
    Gh = L.matvec(Fi_inv, g)
    Hh = L.matvec(Fi_inv, h)
    for k in range(5):
        A[k, 5] = A[k, 5] - Gh[k]
        A[k, 6] = A[k, 6] - Hh[k]
    return qbar, A, Fi_inv * Bs


def row_range(qbar, A, Bq, R, k, fix=None):
    """enclosure of coordinate k over r0 in [-1, 1]^7 (with r0[0] fixed to fix, if given) and r in R"""
    s = qbar[k]
    for j in range(7):
        s = s + A[k, j] * (arb(fix) if (fix is not None and j == 0) else ball(-1, 1))
    for j in range(7):
        s = s + Bq[k, j] * R[j]
    return s


def run_piece(T_lo, T_hi, Tc, sigma, shift=0.0, tag_extra=''):
    t_start = time.time()
    P = Piece(T_lo, T_hi, Tc, sigma, shift)
    ref = P.ref
    name = 'tstrip%s_%s_%s%s' % (('_El%s' % C.HH_EL) if C.HH_EL else '', T_lo, T_hi, tag_extra)
    out = {'T_lo': T_lo, 'T_hi': T_hi, 'T_c': Tc, 'sigma': sigma, 'shift': shift, 'prec': PREC, 'order': ORDER,
           'EL': C.HH_EL or 'zero-current', 'alpha_m_perturbation': hhjet6.ALPHA_M_PERTURB}
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)
    say('piece T in [%s, %s] C (phi in %s), K = K_c + a (phi - phi_c) + s, K_c = %s, a = %s, |s| <= %s' % (
        T_lo, T_hi, P.phi_ball.str(15), P.K_c.str(25), P.a.str(5), sigma))
    # (A) Lemma A, (B) Lemma B with the parameter box, (B') transversality, (C) the block
    Kf = float(P.K_ball.mid())
    A = C.lemma_A(P.K_ball, P.phi_ball, P.EL, Kf * 0.9, Kf * 1.2)
    ok = {'A': bool(A['ok'])}
    rB = float(P.rB.mid())
    s_scale = 1.0
    B = None
    for _ in range(40):
        s = (arb(2e-9 * s_scale * (rB / 1e-4) ** 2), arb(1e-7 * s_scale * (rB / 1e-4) ** 2),
             arb(6e-9 * s_scale * (rB / 1e-4) ** 2))
        B = C.lemma_B(P.K_ball, P.phi_ball, P.EL, P.rB, s, Tf=P.TB)
        if B['ok']:
            break
        s_scale *= 2
    ok['B'] = bool(B['ok'])
    Ti = B['Ti']
    s2, s3, s5 = B['s']
    zb = [P.rB, ball(-s2, s2), ball(-s3, s3), ball(-s3, s3), ball(-s5, s5)]
    Ebox = [P.ystar[i] + sum((Ti[i, k] * zb[k] for k in range(5)), arb(0)) for i in range(5)]
    f = hhjet6.vfield(Ebox + [P.K_ball, P.phi_ball], None, P.EL)
    z1dot = sum((P.TB[0, i] * f[i] for i in range(5)), arb(0))
    ok['B_transversal'] = bool(z1dot > 0)
    blk = json.load(open(block0.block_file(Tc)))
    M, Minv, rho, r, _ = block0.load_block(Tc)
    res = block0.run(Tc, P.K_ball, M, Minv, rho, r, log=lambda s: None, phi=P.phi_ball)
    ok['C'] = bool(res['ok'])
    say('(A) %s  (B) %s with stable box %s  (B\') %s  (C) block %s (%d + %d cells)' % (
        ok['A'], ok['B'], [x.str(3) for x in B['s']], ok['B_transversal'], ok['C'], res['cone']['cells'],
        res['entrance']['cells']))
    # W_0: the exit set times the parameter box
    centre0 = [P.ystar[i] + Ti[i, 0] * P.rB for i in range(5)]
    wid0 = [arb(0), s2, s3, s3, s5]
    cols = [[mid(Ti[i, k]) * wid0[k] if k else arb(0) for k in range(5)] + [arb(0), arb(0)] for i in range(5)]
    R0 = [centre0[i] - mid(centre0[i]) + sum(((Ti[i, k] - mid(Ti[i, k])) * zb[k] for k in range(1, 5)), arb(0))
          for i in range(5)] + [arb(0), arb(0)]
    X0 = P.lohner_set(centre0, cols, R0)
    times = ref['times']
    cents = [[L.deser(v) for v in c] for c in ref['centres']]
    frames = [arb_mat([[L.deser(v) for v in row] for row in Fm]) for Fm in ref['frames']]
    m = len(times) - 1
    X = X0
    win = []          # windows W_1..W_m: (c, F, Finv, D, g, h)
    stage_ok = True
    for i in range(m):
        Xn, t, ns = L.integrate(P.F, X, times[i + 1], ORDER, TOL, hmax=0.05, t0=times[i])
        assert bool(t == arb(times[i + 1])), 'stage time not exact'
        if i == m - 1:
            # (F): the image of W_{m-1} in the interior of B0
            M6 = arb_mat([[M[a_, b_] if b_ < 5 else 0 for b_ in range(7)] for a_ in range(5)])
            zeta = Xn.affine_image_hull(M6, list(P.ystar) + [arb(0), arb(0)])
            ns_up = sum((arb(v.abs_upper()) ** 2 for v in zeta[1:]), arb(0)).sqrt()
            inB = bool(arb(zeta[0].abs_upper()) < arb(r)) and bool(ns_up < arb(rho))
            say('stage %d -> B0 at t = %.4f: zeta_1 in %s, |zeta_s| <= %s: in int B0 %s' % (
                i, float(t.mid()), zeta[0].str(5), ns_up.str(5), inB))
            ok['F'] = inB
            break
        c1, F1 = cents[i + 1], frames[i + 1]
        F1inv = F1.inv()
        zero = [arb(0)] * 5
        qbar, A0m, Bq = coords(Xn, F1inv, c1, zero, zero)
        # shears: the entry dependence on p1, p2 (exit component left out), as state vectors F1 (0, A[1:, j])
        g = L.matvec(F1, [arb(0)] + [mid(A0m[k, 5]) for k in range(1, 5)])
        h = L.matvec(F1, [arb(0)] + [mid(A0m[k, 6]) for k in range(1, 5)])
        g, h = [mid(v) for v in g], [mid(v) for v in h]
        qbar, Am, Bq = coords(Xn, F1inv, c1, g, h)
        ent = [row_range(qbar, Am, Bq, Xn.R, k) for k in range(1, 5)]
        s_new = [arb(arb(e.abs_upper()) * (1 + MARGIN) + arb(2) ** -100).upper() for e in ent]
        if i == 0:
            ex = row_range(qbar, Am, Bq, Xn.R, 0)
            w_new = arb(ex.abs_upper()) * 2 + arb(2) ** -100            # W_0 has no exit faces: any w_1 > 0
            xplus = xminus = None
        else:
            ep = row_range(qbar, Am, Bq, Xn.R, 0, fix=1)
            em = row_range(qbar, Am, Bq, Xn.R, 0, fix=-1)
            xplus, xminus = float(ep.lower()), float(em.upper())
            if not (ep > 0 and em < 0):
                say('stage %d: (X) FAILS: image of the + face has exit coordinate >= %.3e, the - face <= %.3e' % (
                    i, xplus, xminus))
                stage_ok = False
                break
            w_new = arb(min(ep.lower(), -em.upper())) * arb('0.95')
        w_new = arb(w_new.lower())
        D = [w_new] + [arb(v) for v in s_new]
        win.append((c1, F1, F1inv, D, g, h))
        if (i + 1) % 10 == 0 or i < 3:
            say('stage %d t = %.4f: w = %.3e, entry widths %s, exit face images %s / %s, steps %d (%.0f s)' % (
                i + 1, times[i + 1], float(w_new.mid()), ['%.2e' % float(v.mid()) for v in s_new],
                '%.2e' % xplus if xplus is not None else '-', '%.2e' % xminus if xminus is not None else '-', ns,
                time.time() - t_start))
        # the next set: W_{i+1} exactly
        cols = [[sum((F1[a_, b_] * D[b_] * (1 if b_ == j else 0) for b_ in range(5)), arb(0)) for j in range(5)] +
                [g[a_], h[a_]] for a_ in range(5)]
        X = P.lohner_set(c1, cols)
    ok['S_X'] = stage_ok
    ok.setdefault('F', False)
    # (P) endpoint runs
    for sign in (+1, -1):
        if not stage_ok:
            ok['P%+d' % sign] = False
            continue
        colsE = [[mid(Ti[i, k]) * wid0[k] if k else arb(0) for k in range(5)] + [arb(0), arb(0)] for i in range(5)]
        XE = P.lohner_set(centre0, colsE, R0)
        # fix p2 = sign: shift the K centre and zero the s column
        XE.xbar[5] = mid(XE.xbar[5] + P.sigma * sign)
        XE.C[5, 6] = arb(0)
        XE.R[5] = XE.R[5] + arb(0, arb(2) ** -120)
        res_e = None
        for i in range(len(win)):
            XE, t, ns = L.integrate(P.F, XE, times[i + 1], ORDER, TOL, hmax=0.05, t0=times[i])
            assert bool(t == arb(times[i + 1])), 'stage time not exact'
            c1, F1, F1inv, D, g, h = win[i]
            # p2 is fixed at sign: the window coordinates use h * sign; r0[6] has a zero column in XE
            qbar, Am, Bq = coords(XE, F1inv, c1, g, [v * sign for v in h])
            e = row_range(qbar, Am, Bq, XE.R, 0)
            if sign * e > D[0]:
                res_e = (i + 1, e)
                break
            if sign * e < -D[0] or not (sign * e > -D[0]):
                res_e = ('wrong side or undecided', i + 1, e)
                break
        ok['P%+d' % sign] = isinstance(res_e, tuple) and isinstance(res_e[0], int)
        say('(P) endpoint s = %+d sigma: %s' % (sign, 'leaves through the %s face at stage %d' % (
            '+' if sign > 0 else '-', res_e[0]) if ok['P%+d' % sign] else 'FAILS (%s)' % (res_e,)))
    out['checks'] = ok
    out['all'] = all(ok.values())
    out['stages'] = len(win) + 1
    out['secs'] = round(time.time() - t_start)
    out['lines'] = lines
    say('PIECE %s in %d s' % ('PASSED' if out['all'] else 'FAILED', out['secs']))
    json.dump(out, open('%s/%s.json' % (DATA, name), 'w'), indent=1)
    return out['all']


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'reference':
        reference(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif what == 'piece':
        if os.environ.get('TSTRIP_NEG') == 'model':
            hhjet6.ALPHA_M_CENTER = C.rest_state()[0][0]
            hhjet6.ALPHA_M_PERTURB = '1e-8'
        T_lo, T_hi, Tc, sigma = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
        shift = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        extra = ('_shift%g' % shift if shift else '') + ('_negmodel' if os.environ.get('TSTRIP_NEG') == 'model' else '')
        sys.exit(0 if run_piece(T_lo, T_hi, Tc, sigma, shift, extra) else 1)
