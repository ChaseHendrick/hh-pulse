#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): computer-assisted existence proof of the propagated action potential of Hodgkin and
Huxley (J. Physiol. 117 (1952), eq. (31)) at their 1952 rate functions and constants, as an orbit homoclinic to rest
of the travelling-wave ODE (hhwave.py conventions: u = -V, w = u', t in ms, phi = 3^((T - 6.3)/10) for the decimal
temperature T as typed). E_l is the leak potential that makes the resting current zero, with rest at u = 0, or, with
HH_EL=10.613, Hodgkin and Huxley's printed value, with rest at the enclosed equilibrium u* (certify_rest_wave.py).

The argument (written out in the manuscript, Section 4) has these computed hypotheses, each checked here:
 (H2) Lemma B of certify_rest_wave.py at the radius r_B, in coordinates z = T_B (y - y*) that diagonalize Df(y*) to
     about 1e-70 (a rigorous eigen-decomposition at the working precision, then exact dyadic midpoints), for every K
     in [K1, K2]: (i) the cone condition on the box B (Gershgorin on the interval matrix D A + A^T D), (ii) the stable
     faces are inflowing, and (iii) z1' > 0 on the exit set E = { y* + T_B^-1 (r_B e1 + z') : |z'_2| <= s2,
     |(z'_3, z'_4)| <= s3, |z'_5| <= s5 }. Also checked: u > u* on all of E, so the branch of the unstable manifold
     that leaves B through E does so with u above its rest value.
 (H1) follows from (H2)(i) (Lemma 0 of the manuscript); the characteristic polynomial check of certify_rest_wave.lemma_A
     (one simple real unstable eigenvalue, the other four in Re < 0) runs as an independent check that the proof does
     not use.
 (H3) The closing block B0 of block0.py (cone and entrance conditions), for every K in [K1, K2].
 (H4) The interval run: the Lohner set containing { (y, K) : y in E, K in [K1, K2] }, integrated from t = 0 to
     t = T_enter, lies in the interior of B0 at t = T_enter.
 (H5) The endpoint runs: for K = K1 (resp. K2) the set E is integrated to T_enter, lies in int B0 there, and is then
     integrated further with enclosures of the whole path over every step, which stay in int B0, until the set lies
     in K- = {L > 0, zeta_1 < 0} (resp. K+ = {L > 0, zeta_1 > 0}).
Conclusion: some K* in (K1, K2) has an orbit that leaves rest along the branch z1 > 0 of W^u and stays in B0 for all
t >= T_enter, hence tends to rest: a homoclinic orbit, the pulse, with speed theta = sqrt(K* a / (2 R_2 C_M)).

Negative controls (each must fail, for its stated reason): the interval run for a K interval that does not contain the
pulse speed (shifted by 40 half-widths; it must reach T_enter outside int B0), and for the model with alpha_m
multiplied by 1 + 1e-12 (u - u*)^2 at the true interval (the whole set must escape below u = -60 mV); Lemma B with
faces 100 times too thin; the block with a radius 1.5 times too large (block0.py).

Stages (run separately; each writes data/pulse_proof_<tag>_<stage>.json and checkpoints to data/ckpt/):
  python3 hh_prove_pulse.py <T> config <delta> <r_B> <T_enter> <tol_final> <tol_min>   # from data/hp_pulse_<tag>.json
  python3 hh_prove_pulse.py <T> setup      # (H2), (H3), the check of (H1), and their negative controls
  python3 hh_prove_pulse.py <T> interval   # (H4)
  python3 hh_prove_pulse.py <T> K1         # (H5) at K1
  python3 hh_prove_pulse.py <T> K2         # (H5) at K2
  python3 hh_prove_pulse.py <T> neg-shift  # negative control: shifted K interval
  python3 hh_prove_pulse.py <T> neg-model  # negative control: perturbed alpha_m
  python3 hh_prove_pulse.py <T> summary    # checks every certificate; exit status 0 iff everything is as expected
  python3 hh_prove_pulse.py <T> summary-control   # the summary must reject planted stale or wrong certificates
T is a decimal string (18.5, 6.3), never read as a float. Every certificate records its provenance: the sha256 of the
configuration, of the closing block and of the programs (PROGRAMS), the python-flint version and the phi and E_l balls;
the summary recomputes them and refuses a certificate that does not match, whose K is not the configuration's, or
whose verdict contradicts its own fields.
Exit status of a stage: 0 if it passed, 1 if it wrote the verdict FAIL, 2 on any error (no certificate is written).
"""
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import flint
from flint import arb, acb_mat, arb_mat, ctx
import certify_rest_wave as C
import hhjet6
import lohner6 as L
import block0

PREC = 256
ORDER = 40
DATA = '../data'
HERE = os.path.dirname(os.path.abspath(__file__))
# the programs whose code the certificates depend on (their sha256 goes into every certificate)
PROGRAMS = ('hh_prove_pulse.py', 'certify_rest_wave.py', 'lohner6.py', 'hhjet6.py', 'hhjet.py', 'hhseries.py',
            'block0.py', 'hhwave.py')
STAGES = ('setup', 'interval', 'K1', 'K2', 'neg-shift', 'neg-model')
# what each negative control must fail with: the stated reason, as recorded in the certificate's 'fail' field
NEG_REASON = {'neg-shift': 'outside_int_B0_at_T_enter', 'neg-model': 'escaped_below_-60'}
ctx.prec = PREC            # the imports above set other precisions; every stage below also sets it


class CheckFailed(Exception):
    pass


def ball(lo, hi):
    return arb(lo).union(arb(hi))


def require(cond, msg):
    """A failed check is an error (exit status 2), never a FAIL verdict: no certificate is written."""
    if not cond:
        raise CheckFailed(msg)


def cfg_path(T, data=DATA):
    return '%s/pulse_proof_%s_config.json' % (data, C.tag(T))


def out_path(T, stage, data=DATA):
    return '%s/pulse_proof_%s_%s.json' % (data, C.tag(T), stage)


def block_path(T, data=DATA):
    return '%s/closing_block_%s.json' % (data, C.tag(T))


def sha256_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def code_sha256():
    """One hash of the programs in PROGRAMS (each file's name and sha256, in that order)."""
    return hashlib.sha256(''.join('%s %s\n' % (f, sha256_file(os.path.join(HERE, f))) for f in PROGRAMS)
                          .encode()).hexdigest()


def provenance(T, data=DATA):
    """What a certificate was computed from; the summary requires every certificate's copy to equal the current one."""
    old = ctx.prec
    ctx.prec = PREC
    phi = C.phi_of(T)
    _, EL = C.rest_state()
    out = {'config_sha256': sha256_file(cfg_path(T, data)), 'block_sha256': sha256_file(block_path(T, data)),
           'code_sha256': code_sha256(), 'python_flint': flint.__version__, 'phi': phi.str(50), 'E_l': EL.str(50)}
    ctx.prec = old
    return out


# ---------------------------------------------------------------------------------------------------------------
# configuration

def make_config(T, delta, r_B, T_enter, tol_final, tol_min):
    ctx.prec = PREC
    hp = json.load(open('%s/hp_pulse_%s.json' % (DATA, C.tag(T))))
    Ks = arb(hp['K'])
    # K1, K2: exact dyadic numbers (midpoints) at distance about delta from the numerical K*
    K1 = arb((Ks - arb(delta)).mid())
    K2 = arb((Ks + arb(delta)).mid())
    require(bool(abs((K2 - K1) / (2 * arb(delta)) - 1) < arb('1e-6')), 'K1, K2 not at the intended distance')
    sc = (float(r_B) / 1e-4) ** 2
    cfg = {'T': float(T), 'T_decimal': T, 'phi': C.phi_of(T).str(50), 'K_numerical': hp['K'], 'delta': delta,
           'K1': ser(K1), 'K2': ser(K2), 'K1_dec': K1.str(70, radius=False), 'K2_dec': K2.str(70, radius=False),
           'r_B': r_B, 's': ['%.3e' % (2e-9 * sc), '%.3e' % (1e-7 * sc), '%.3e' % (6e-9 * sc)],
           'T_enter': T_enter, 'order': ORDER, 'prec': PREC, 'prec_aux': L.PREC_AUX,
           'tol_final': tol_final, 'tol_min': tol_min}
    json.dump(cfg, open(cfg_path(T), 'w'), indent=1)
    return cfg


def ser(a):
    return L.ser(a)


def deser(s):
    return L.deser(s)


def cfg_hash(cfg, extra=''):
    return hashlib.sha256((json.dumps(cfg, sort_keys=True) + extra).encode()).hexdigest()[:16]


def kpart(K1, K2, stage):
    """The K ball (or point) a stage integrates, from the configuration's K1 and K2."""
    if stage in ('interval', 'neg-model'):
        return K1.union(K2)
    if stage == 'neg-shift':
        d = K2 - K1
        return (K2 + 19 * d).union(K2 + 20 * d)
    if stage == 'K1':
        return K1
    if stage == 'K2':
        return K2
    raise CheckFailed('unknown stage ' + stage)


# ---------------------------------------------------------------------------------------------------------------
# Lemma B coordinates: a rigorous eigen-decomposition, then exact dyadic midpoints

def lemmaB_T(A0):
    """Real eigenbasis (unstable, fast real, Re and Im of the complex pair, slow real) of the 5 x 5 arb matrix A0 from
    acb_mat.eig at the working precision; columns normalized as in certify_rest_wave.real_basis; returns T = an exact
    dyadic matrix close to the inverse of that basis. Only its exactness matters for rigour (Lemma B is verified for
    whatever T is used); its accuracy makes the linear part nearly diagonal, which the check at tiny r_B needs."""
    E, R = acb_mat(A0).eig(right=True)
    ev = [(float(e.real.mid()), float(e.imag.mid()), k) for k, e in enumerate(E)]
    iu = max(ev, key=lambda t: t[0])[2]
    reals = sorted([t for t in ev if abs(t[1]) < 1e-9 and t[2] != iu], key=lambda t: t[0])
    cpx = [t for t in ev if t[1] > 1e-9][0][2]
    fast, slow = reals[0][2], reals[1][2]

    def col(k):
        return [R[i, k] for i in range(5)]
    vu = col(iu)
    vu = [x / vu[0] for x in vu]
    vc = col(cpx)
    vc = [x / vc[0] for x in vc]

    def realnorm(v):
        i = max(range(5), key=lambda i: abs(float(v[i].real.mid())))
        return [x / v[i] for x in v]
    vf, vs = realnorm(col(fast)), realnorm(col(slow))
    cols = [[x.real for x in vu], [x.real for x in vf], [x.real for x in vc], [x.imag for x in vc],
            [x.real for x in vs]]
    P = arb_mat([[arb(cols[j][i].mid()) for j in range(5)] for i in range(5)])
    Pi = P.inv()
    return arb_mat([[arb(Pi[i, j].mid()) for j in range(5)] for i in range(5)])


# ---------------------------------------------------------------------------------------------------------------
# common setup

class Setup:
    def __init__(self, T, perturb=None):
        ctx.prec = PREC
        self.cfg = json.load(open(cfg_path(T)))
        self.T = T
        self.phi = C.phi_of(T)
        require(self.cfg.get('T_decimal') == T and self.cfg.get('phi') == self.phi.str(50),
                'the configuration was made for another temperature or phi')
        self.ystar, self.EL = C.rest_state()
        self.K1, self.K2 = deser(self.cfg['K1']), deser(self.cfg['K2'])
        self.Kball = self.K1.union(self.K2)
        A0 = C.jac(self.ystar, arb(self.Kball.mid()), self.phi, self.EL)
        self.TB = lemmaB_T(A0)
        self.rB = arb(self.cfg['r_B'])
        self.s = tuple(arb(v) for v in self.cfg['s'])
        M, Minv, rho, r, d = block0.load_block(T)
        self.M, self.Minv, self.rho, self.r = M, Minv, arb(rho), arb(r)
        self.M6 = arb_mat([[M[i, j] if j < 5 else 0 for j in range(6)] for i in range(5)])
        self.shift6 = list(self.ystar) + [arb(0)]
        self.T_enter = self.cfg['T_enter']
        self.lam = None

    def lemma_B(self, s=None):
        return C.lemma_B(self.Kball, self.phi, self.EL, self.rB, s or self.s, Tf=self.TB)

    def exit_set(self, Kpart, B):
        """Lohner set for { (y* + T_B^-1 (r_B e1 + z'), K) : z' in the stable box, K in Kpart } (Kpart a ball)."""
        Ti = B['Ti']
        s2, s3, s5 = B['s']
        R0 = [ball(-s2, s2), ball(-s3, s3), ball(-s3, s3), ball(-s5, s5)]
        centre = [self.ystar[i] + Ti[i, 0] * self.rB for i in range(5)]
        xbar = [arb(c.mid()) for c in centre] + [arb(Kpart.mid())]
        kr = arb(Kpart.rad())
        m = 5 if kr > 0 else 4
        Cm = arb_mat(6, m)
        for i in range(5):
            for k in range(4):
                Cm[i, k] = arb(Ti[i, k + 1].mid())
        if m == 5:
            Cm[5, 4] = arb(kr.mid()) if arb(kr.mid()) >= kr else arb(kr.upper())
            R0 = R0 + [ball(-1, 1)]
        Rr = [centre[i] - xbar[i] + sum(((Ti[i, k + 1] - Cm[i, k]) * R0[k] for k in range(4)), arb(0))
              for i in range(5)]
        Rr.append(arb(0))
        # the K ball must be covered: K in xbar_K + Cm[5, 4] [-1, 1]
        if m == 5:
            require(bool(ball(xbar[5] - Cm[5, 4], xbar[5] + Cm[5, 4]).contains(Kpart)), 'K ball not covered')
        else:
            require(bool(Kpart == xbar[5]), 'K point not exact')
        Bm = arb_mat([[1 if i == j else 0 for j in range(6)] for i in range(6)])
        return L.LSet(xbar, Cm, R0, Bm, Rr)

    def zeta(self, X):
        return X.affine_image_hull(self.M6, self.shift6)

    def norm_s_upper(self, z):
        return sum((arb(v.abs_upper()) ** 2 for v in z[1:]), arb(0)).sqrt()

    def in_int_B0(self, z):
        return bool(arb(z[0].abs_upper()) < self.r) and bool(self.norm_s_upper(z) < self.rho)

    def in_cone(self, z, sign):
        v = z[0] if sign > 0 else -z[0]
        return bool(arb(v.lower()) > self.norm_s_upper(z))

    def tol(self, lam):
        tf, tm, Te = self.cfg['tol_final'], self.cfg['tol_min'], self.T_enter

        def f(t):
            return max(tm, tf * math.exp(-lam * max(Te - t, 0.0)))
        return f


# ---------------------------------------------------------------------------------------------------------------
# stages

def stage_setup(T):
    S = Setup(T)
    out = {'stage': 'setup', 'T': float(T), 'prec': PREC, 'K1': S.cfg['K1_dec'], 'K2': S.cfg['K2_dec']}
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)
    say('T = %s C, phi = 3^((T - 6.3)/10) = %s, E_l = %s' % (T, S.phi.str(30), S.EL.str(40)))
    say('K1 = %s\nK2 = %s' % (S.cfg['K1_dec'], S.cfg['K2_dec']))
    # (A)
    Kf = float(S.Kball.mid())
    A = C.lemma_A(S.Kball, S.phi, S.EL, Kf * 0.9, Kf * 1.2)
    bad = C.lemma_A(S.Kball, S.phi, S.EL, Kf * 1.2, Kf * 2.4)
    say('(A) one real simple unstable eigenvalue in %s, the other four in Re < 0 (Hurwitz: D2 = %s, D3 = %s): %s; '
        'negative control (bracket above lambda_u) rejected: %s' % (A['lam_u'].str(30), A['D2'].str(8),
                                                                   A['D3'].str(8), A['ok'], not bad['ok']))
    out['A'] = bool(A['ok']) and not bad['ok']
    out['lambda_u'] = A['lam_u'].str(40)
    # (B)
    B = S.lemma_B()
    say('(B) Lemma B at r_B = %s, stable box %s: %s; inflow bounds %s; cone Gershgorin bounds %s' % (
        S.cfg['r_B'], S.cfg['s'], B['ok'], [v.str(5) for v in B['inflow'].values()], [g.str(5) for g in B['gersh']]))
    badB = S.lemma_B(tuple(x / 100 for x in S.s))
    say('    negative control (stable faces 100 times thinner) rejected: %s' % (not badB['ok']))
    out['B'] = bool(B['ok']) and not badB['ok']
    # (B') transversality: z1' = (T_B f(y))_1 > 0 on E for every K in the ball
    Ti = B['Ti']
    s2, s3, s5 = B['s']
    zb = [S.rB, ball(-s2, s2), ball(-s3, s3), ball(-s3, s3), ball(-s5, s5)]
    Ebox = [S.ystar[i] + sum((Ti[i, k] * zb[k] for k in range(5)), arb(0)) for i in range(5)]
    f = hhjet6.vfield(Ebox + [S.Kball], S.phi, S.EL)
    z1dot = sum((S.TB[0, i] * f[i] for i in range(5)), arb(0))
    say("(B') z1' on the exit set: %s (needs > 0): %s" % (z1dot.str(8), bool(z1dot > 0)))
    out['B_transversal'] = bool(z1dot > 0)
    # the exit set lies where u > u*: the branch that leaves B through E does so with u above its rest value
    du = Ebox[0] - S.ystar[0]
    say("(B'') u - u* on the exit set: %s (needs > 0): %s" % (du.str(8), bool(du > 0)))
    out['B_exit_u_above_rest'] = bool(du > 0)
    # (C) the closing block for every K in the ball, and its negative control
    ctx.prec = 128
    res = block0.run(T, S.Kball, S.M, S.Minv, float(S.rho.mid()), float(S.r.mid()), log=lambda s: None)
    neg = block0.run(T, S.Kball, S.M, S.Minv, 1.5 * float(S.rho.mid()), 1.5 * float(S.r.mid()), log=lambda s: None)
    ctx.prec = PREC
    say('(C) block B0 (rho = %s, r = %s): cone %s on %d cells, entrance %s on %d cells; negative control (radius x '
        '1.5) rejected: %s' % (S.rho.str(10), S.r.str(10), res['cone']['ok'], res['cone']['cells'],
                                res['entrance']['ok'], res['entrance']['cells'], not neg['ok']))
    out['C'] = bool(res['ok']) and not neg['ok']
    out['ok'] = all(out[k] for k in ('A', 'B', 'B_transversal', 'B_exit_u_above_rest', 'C'))
    out['lines'] = lines
    out['provenance'] = provenance(T)
    json.dump(out, open(out_path(T, 'setup'), 'w'), indent=1)
    say('SETUP ' + ('PASSED' if out['ok'] else 'FAILED'))
    return out['ok']


def jac6(y, K, phi, EL):
    """Df at rest or over a box, from hhjet6 (so that the negative control's perturbation reaches Lemmas A and B)."""
    J = hhjet6.jacobian(list(y) + [K], phi, EL)
    return arb_mat([[J[i][j] for j in range(5)] for i in range(5)])


C.jac = jac6


def perturb_model(eps):
    """alpha_m -> alpha_m (1 + eps (u - u*)^2) everywhere, u* the enclosed rest value (negative control only; rest and
    the linearization there are unchanged, so Lemmas A and B still apply)."""
    ctx.prec = PREC
    hhjet6.ALPHA_M_CENTER = C.rest_state()[0][0]
    hhjet6.ALPHA_M_PERTURB = eps


def run_stage(T, stage):
    t_start = time.time()
    if stage == 'neg-model':
        perturb_model('1e-12')
    S = Setup(T)
    cfg = S.cfg
    prov = provenance(T)
    setup = json.load(open(out_path(T, 'setup')))
    require(setup.get('ok') is True and setup.get('provenance') == prov,
            'the setup certificate is missing, failed, or was made from another configuration, block or program')
    lam = float(arb(setup['lambda_u']).mid())
    B = S.lemma_B()
    require(B['ok'], 'Lemma B')
    Kpart = kpart(S.K1, S.K2, stage)
    F = L.Field(S.phi, S.EL)
    ck = '%s/ckpt/pulse_%s_%s.json' % (DATA, C.tag(T), stage)
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    h = cfg_hash(cfg, stage + prov['code_sha256'] + prov['block_sha256'])      # a checkpoint of other code is ignored
    state = {'phase': 'approach', 'steps': 0, 'umax_lower': -1e9, 'umax_upper_steps': -1e9}
    t0 = 0.0
    X = S.exit_set(Kpart, B)
    if os.path.exists(ck):
        d = json.load(open(ck))
        if d.get('hash') == h:
            X = L.LSet.from_json(d['X'])
            t0 = float(deser(d['t_ser']))
            state = d['state']
            print('%s: resumed from checkpoint at t = %s (%d steps)' % (stage, t0, state['steps']), flush=True)
    tolf = S.tol(lam)
    log = {'stage': stage, 'T': float(T), 'K': Kpart.str(70), 'T_enter': S.T_enter, 'order': ORDER, 'prec': PREC}
    last_ck = [time.time()]

    def cb(tp, t, Xh, Xn, W, hh):
        state['steps'] += 1
        hx = Xn.hull()
        ul = float(hx[0].lower())
        if arb(ul) > hx[0].lower():                  # float() rounds to nearest: step down to a lower bound
            ul = math.nextafter(ul, -math.inf)
        state['umax_lower'] = max(state['umax_lower'], ul)
        state['umax_upper_steps'] = max(state['umax_upper_steps'], float(hx[0].upper()))
        wid = max(float(arb(x.rad()).mid()) for x in hx[:5])
        if state['steps'] % 25 == 0:
            z = S.zeta(Xn)
            print('%s t=%.4f steps=%d h=%.2e u=%s maxrad=%.2e zeta1=%s |zeta_s|<=%.3e (%.0fs)' % (
                stage, float(t.mid()), state['steps'], hh, hx[0].str(6, radius=False), wid, z[0].str(4),
                float(S.norm_s_upper(z).mid()), time.time() - t_start), flush=True)
        if wid > 1e3 or not all(x.is_finite() for x in hx):
            state['phase'] = 'blew_up'
            return True
        if hx[0] < -60 or hx[0] > 150:        # the whole set has escaped
            state['phase'] = 'escaped'
            state['escape'] = 'u < -60 mV' if hx[0] < -60 else 'u > 150 mV'
            state['t_escape'] = float(t.mid())
            return True
        if state['phase'] == 'inside':
            zr = L.step_range(F, Xh, W, arb((t - tp).upper()), ORDER, S.M6, S.shift6)
            if not S.in_int_B0(zr):
                state['phase'] = 'left_int_B0'
                return True
            z = S.zeta(Xn)
            sign = -1 if stage == 'K1' else 1
            if S.in_cone(z, sign):
                state['phase'] = 'in_cone'
                state['t_cone'] = float(t.mid())
                state['zeta_cone'] = [v.str(10) for v in z]
                return True
            if S.in_cone(z, -sign):
                state['phase'] = 'wrong_cone'
                return True
        if time.time() - last_ck[0] > 120:
            require(bool(arb(float(t.mid())) == t), 'time not a float')
            json.dump({'hash': h, 't_ser': ser(t), 'X': Xn.to_json(), 'state': state}, open(ck + '.tmp', 'w'))
            os.replace(ck + '.tmp', ck)
            last_ck[0] = time.time()
        return False

    if state['phase'] == 'approach':
        X, t, ns = L.integrate(F, X, S.T_enter, ORDER, tolf, hmax=0.25, t0=t0, callback=cb)
        if state['phase'] == 'blew_up':
            log['verdict'] = 'FAIL'
            log['fail'] = 'blew_up'
            log['reason'] = 'the set blew up at t = %s' % t.str(8)
        elif state['phase'] == 'escaped':
            log['verdict'] = 'FAIL'
            log['fail'] = 'escaped_below_-60' if state['escape'].startswith('u <') else 'escaped_above_150'
            log['reason'] = 'the whole set escaped (%s) at t = %s, before T_enter' % (state['escape'], t.str(8))
        else:
            require(bool(t == arb(S.T_enter)), 'did not reach T_enter exactly')
            z = S.zeta(X)
            inB = S.in_int_B0(z)
            log['zeta_at_T_enter'] = [v.str(12) for v in z]
            log['|zeta_s|_upper_at_T_enter'] = S.norm_s_upper(z).str(10)
            log['in_int_B0_at_T_enter'] = inB
            log['steps_to_T_enter'] = state['steps']
            log['u_max_lower_bound'] = state['umax_lower']
            print('%s: at T_enter = %s: zeta = %s, |zeta_s| <= %s, in int B0: %s' % (
                stage, S.T_enter, [v.str(6) for v in z], S.norm_s_upper(z).str(6), inB), flush=True)
            if stage in ('K1', 'K2') and inB:
                state['phase'] = 'inside'
                json.dump({'hash': h, 't_ser': ser(t), 'X': X.to_json(), 'state': state}, open(ck, 'w'))
                t0 = float(t.mid())
            else:
                log['verdict'] = 'PASS' if inB else 'FAIL'
                if not inB:
                    log['fail'] = 'outside_int_B0_at_T_enter'
    if state['phase'] in ('inside',):
        # short steps after T_enter: zeta_1 grows by about exp(lambda_u h) per step, which must not carry the set past
        # |zeta_1| = r between two checks of the cone (the path check would then fail, safely but uselessly)
        X, t, ns = L.integrate(F, X, t0 + 20.0, ORDER, cfg['tol_final'], hmax=2.0 ** -7, t0=t0, callback=cb)
        log['phase2'] = state['phase']
        if state['phase'] == 'in_cone':
            log['t_cone'] = state['t_cone']
            log['zeta_at_t_cone'] = state['zeta_cone']
        log['verdict'] = 'PASS' if state['phase'] == 'in_cone' else 'FAIL'
        log['steps_total'] = state['steps']
        log['u_max_lower_bound'] = state['umax_lower']
    elif state['phase'] in ('in_cone', 'left_int_B0', 'wrong_cone'):
        log['phase2'] = state['phase']
        log['verdict'] = 'PASS' if state['phase'] == 'in_cone' else 'FAIL'
    require('verdict' in log, 'the stage ended in phase %s without a verdict' % state['phase'])
    if log['verdict'] == 'FAIL' and 'fail' not in log:
        log['fail'] = state['phase']
    log['provenance'] = prov
    log['secs'] = round(time.time() - t_start)
    json.dump(log, open(out_path(T, stage), 'w'), indent=1)
    print('%s VERDICT %s (%d s)' % (stage, log['verdict'], log['secs']), flush=True)
    return log['verdict'] == 'PASS'


SETUP_CHECKS = ('A', 'B', 'B_transversal', 'B_exit_u_above_rest', 'C')


def contradictions(st, d):
    """Fields of a certificate that contradict its own verdict (the summary recomputes the verdict from them)."""
    out = []
    if st == 'setup':
        if d.get('ok') != all(d.get(k) is True for k in SETUP_CHECKS):
            out.append('ok is not the conjunction of %s' % ', '.join(SETUP_CHECKS))
        return out
    v, inB = d.get('verdict'), d.get('in_int_B0_at_T_enter')
    if v not in ('PASS', 'FAIL'):
        out.append('no verdict')
    if st == 'interval' and (v == 'PASS') != (inB is True):
        out.append('the verdict disagrees with in_int_B0_at_T_enter')
    if st in ('K1', 'K2') and v == 'PASS' and not (inB is True and d.get('phase2') == 'in_cone' and 't_cone' in d):
        out.append('PASS without reaching the cone from int B0')
    if st == 'neg-shift' and v == 'FAIL' and d.get('fail') == NEG_REASON[st] and inB is not False:
        out.append('the stated reason disagrees with in_int_B0_at_T_enter')
    if st == 'neg-model' and v == 'FAIL' and d.get('fail') == NEG_REASON[st] and not str(
            d.get('reason', '')).startswith('the whole set escaped (u < -60 mV)'):
        out.append('the stated reason disagrees with the recorded one')
    if v == 'PASS' and 'fail' in d:
        out.append('PASS with a reason for failure')
    return out


def check_certificates(T, data=DATA):
    """Every certificate of the proof, checked against the current configuration, closing block, programs, phi and E_l.
    Returns (ok, lines)."""
    ctx.prec = PREC
    ok = True
    lines = []
    try:
        prov = provenance(T, data)
        cfg = json.load(open(cfg_path(T, data)))
    except (OSError, ValueError) as e:
        return False, ['configuration or closing block unreadable: %s' % e]
    # phi must enclose 3^((T - 6.3)/10) for the decimal temperature (computed here again at twice the precision)
    ctx.prec = 2 * PREC
    phi_exact = arb(3) ** ((arb(C.temperature(T)) - arb('6.3')) / 10)
    ctx.prec = PREC
    phi_ok = cfg.get('T_decimal') == T and cfg.get('phi') == prov['phi'] and bool(arb(prov['phi']).contains(phi_exact))
    ok = ok and phi_ok
    lines.append('%-10s %s' % ('phi', ('encloses 3^((%s - 6.3)/10): %s' % (T, prov['phi'][:40])) if phi_ok
                              else 'DOES NOT MATCH the decimal temperature %s' % T))
    K1, K2 = deser(cfg['K1']), deser(cfg['K2'])
    for st in STAGES:
        p = out_path(T, st, data)
        if not os.path.exists(p):
            lines.append('%-10s MISSING' % st)
            ok = False
            continue
        d = json.load(open(p))
        why = []
        if d.get('provenance') != prov:
            got = d.get('provenance') or {}
            why.append('made from another ' + (', '.join(k for k in prov if got.get(k) != prov[k]) or 'record'))
        if d.get('stage') != st or d.get('T') != float(T) or d.get('prec') != PREC:
            why.append('stage, temperature or precision differ')
        if st == 'setup':
            if (d.get('K1'), d.get('K2')) != (cfg['K1_dec'], cfg['K2_dec']):
                why.append('K1, K2 differ from the configuration')
            passed = d.get('ok') is True
        else:
            if d.get('K') != kpart(K1, K2, st).str(70) or d.get('order') != ORDER:
                why.append('K or order differ from the configuration')
            passed = d.get('verdict') == 'PASS'
        why += contradictions(st, d)
        if st in NEG_REASON:
            good = d.get('verdict') == 'FAIL' and d.get('fail') == NEG_REASON[st]
            what = 'failed (%s), a negative control' % d.get('fail') if not passed else 'passed, a negative control'
        else:
            good = passed
            what = 'passed' if passed else 'failed'
        good = good and not why
        ok = ok and good
        lines.append('%-10s %s (%s)%s' % (st, 'as expected' if good else 'NOT AS EXPECTED', what,
                                          ('; STALE, FOREIGN OR INCONSISTENT: ' + '; '.join(why)) if why else ''))
    lines.append('certificates made from config %s, closing block %s, programs %s, python-flint %s (sha256, first 16)'
                 % (prov['config_sha256'][:16], prov['block_sha256'][:16], prov['code_sha256'][:16],
                    prov['python_flint']))
    return ok, lines


def stage_summary(T, data=DATA, write=True):
    ok, lines = check_certificates(T, data)
    ctx.prec = PREC
    cfg = json.load(open(cfg_path(T, data)))
    K1, K2 = deser(cfg['K1']), deser(cfg['K2'])

    def theta(K):
        # theta = sqrt(K a / (2 R_2 C_M)): K in 1/ms, a = 0.0238 cm, R_2 = 35.4 ohm cm, C_M = 1e-6 F/cm^2; in m/s
        return (K * 1000 * arb('0.0238') / (2 * arb('35.4') * arb('1e-6'))).sqrt() / 100
    lines.append('K1 = %s /ms, K2 = %s /ms (K2 - K1 = %s)' % (cfg['K1_dec'][:60], cfg['K2_dec'][:60],
                                                           (K2 - K1).str(5)))
    # outward-rounded decimal bounds, with enough digits to separate theta(K1) from theta(K2)
    from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
    getcontext().prec = 400
    t1, t2 = theta(K1), theta(K2)
    d = int(-float((t2 - t1).mid().log()) / 2.302585) + 3
    q = Decimal(1).scaleb(-d)
    lo = Decimal((t1.lower() - arb(10) ** -d).str(d + 10, radius=False)).quantize(q, rounding=ROUND_FLOOR)
    hi = Decimal((t2.upper() + arb(10) ** -d).str(d + 10, radius=False)).quantize(q, rounding=ROUND_CEILING)
    require(bool(arb(str(lo)) < t1) and bool(arb(str(hi)) > t2), 'the decimal speed bounds are not outward')
    lines.append('speed theta in (%s, %s) m/s for a = 238 um, R_2 = 35.4 ohm cm, C_M = 1 uF/cm^2' % (lo, hi))
    lines.append('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    if write:
        print('\n'.join(lines))
        open('%s/pulse_proof_%s_summary.txt' % (data, C.tag(T)), 'w').write('\n'.join(lines) + '\n')
    return ok


def stage_summary_control(T):
    """A negative control for the harness itself: copies of the certificates with one planted defect each (the kind a
    crashed or skipped rerun would leave behind) must each be refused by the summary, and the unaltered copy must
    pass. Writes data/pulse_proof_<tag>_summary_control.txt; exit status 0 iff every case behaves as expected."""
    names = [cfg_path(T), block_path(T)] + [out_path(T, st) for st in STAGES]

    def edit(path, f):
        d = json.load(open(path))
        f(d)
        json.dump(d, open(path, 'w'), indent=1)

    def cert(data, st):
        return out_path(T, st, data)

    def set_prov(key, val):
        return lambda d: d['provenance'].__setitem__(key, val)

    cases = [
        ('unaltered copy', True, lambda data: None),
        ('K1 certificate missing (the stage crashed)', False, lambda data: os.remove(cert(data, 'K1'))),
        ('K2 certificate made by other programs', False,
         lambda data: edit(cert(data, 'K2'), set_prov('code_sha256', '0' * 64))),
        ('interval certificate made from another configuration', False,
         lambda data: edit(cert(data, 'interval'), set_prov('config_sha256', '0' * 64))),
        ('K1 certificate made with another closing block', False,
         lambda data: edit(cert(data, 'K1'), set_prov('block_sha256', '0' * 64))),
        ('setup certificate with the phi of the binary number 6.3', False,
         lambda data: edit(cert(data, 'setup'), set_prov('phi', (arb(3) ** ((arb(6.3) - arb('6.3')) / 10)).str(50)))),
        ('K2 certificate for another K', False,
         lambda data: edit(cert(data, 'K2'), lambda d: d.__setitem__('K', kpart(*[deser(json.load(open(
             cfg_path(T, data)))[k]) for k in ('K1', 'K2')], 'K1').str(70)))),
        ('configuration changed after the stages ran', False,
         lambda data: edit(cfg_path(T, data), lambda d: d.__setitem__('K_numerical', d['K_numerical'] + '1'))),
        ('closing block changed after the stages ran', False,
         lambda data: edit(block_path(T, data), lambda d: d.__setitem__('K_ref', d['K_ref'] + '1'))),
        ('neg-model failed for another reason (the set blew up)', False,
         lambda data: edit(cert(data, 'neg-model'), lambda d: d.__setitem__('fail', 'blew_up'))),
        ('setup certificate with "C": false and "ok": true', False,
         lambda data: edit(cert(data, 'setup'), lambda d: d.__setitem__('C', False))),
        ('interval PASS with in_int_B0_at_T_enter false', False,
         lambda data: edit(cert(data, 'interval'), lambda d: d.__setitem__('in_int_B0_at_T_enter', False))),
        ('K1 PASS without reaching the cone', False,
         lambda data: edit(cert(data, 'K1'), lambda d: d.__setitem__('phase2', 'left_int_B0'))),
        ('neg-shift passed', False,
         lambda data: edit(cert(data, 'neg-shift'), lambda d: (d.__setitem__('verdict', 'PASS'), d.pop('fail', None)))),
    ]
    lines = ['summary control at T = %s C (%s): each planted defect must make the summary fail' % (T, C.tag(T))]
    allok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, want, plant in cases:
            data = os.path.join(tmp, 'data')
            shutil.rmtree(data, ignore_errors=True)
            os.makedirs(data)
            for f in names:
                shutil.copy(f, data)
            plant(data)
            try:
                got, why = check_certificates(T, data)
            except (OSError, ValueError, KeyError, CheckFailed) as e:
                got, why = False, ['error: %s' % e]
            good = got == want
            allok = allok and good
            bad = [w for w in why if 'NOT AS EXPECTED' in w or 'MISSING' in w or 'DOES NOT' in w or 'error' in w]
            lines.append('%-55s summary %s: %s%s' % (name, 'passes' if got else 'fails',
                                                      'as expected' if good else 'NOT AS EXPECTED',
                                                      (' (' + bad[0].strip()[:90] + ')') if bad else ''))
    lines.append('ALL CHECKS PASSED' if allok else 'SOME CHECK FAILED')
    print('\n'.join(lines))
    open('%s/pulse_proof_%s_summary_control.txt' % (DATA, C.tag(T)), 'w').write('\n'.join(lines) + '\n')
    return allok


def main(argv):
    T = C.temperature(argv[1])            # a decimal string: phi is computed from it exactly
    stage = argv[2]
    if stage == 'config':
        # python3 hh_prove_pulse.py T config delta r_B T_enter tol_final tol_min
        make_config(T, argv[3], argv[4], float(argv[5]), float(argv[6]), float(argv[7]))
        return 0
    if stage == 'setup':
        return 0 if stage_setup(T) else 1
    if stage == 'summary':
        return 0 if stage_summary(T) else 1
    if stage == 'summary-control':
        return 0 if stage_summary_control(T) else 1
    if stage not in STAGES:
        raise CheckFailed('unknown stage ' + stage)
    return 0 if run_stage(T, stage) else 1


if __name__ == '__main__':
    try:
        status = main(sys.argv)
    except CheckFailed as e:
        print('CHECK FAILED: %s' % e, flush=True)
        status = 2
    except Exception:
        traceback.print_exc()
        print('ERROR (exit status 2)', flush=True)
        status = 2
    sys.exit(status)
