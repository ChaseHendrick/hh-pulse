#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Rigorous (ball arithmetic): the record (R) of the stability proof (the design note, Section 8.3, not part of this release).
It reruns the interval stage of prove_pulse.py (the same set, integrator, order, precision and step
tolerances, so the same computation as hypothesis (H4) of the existence proof) and keeps, for every step j:
  t_j, h_j, the hull X_j of the Lohner set at the step start (6 balls: y and K), and the a priori enclosure W_j of
  every solution from X_j on [t_j, t_j + h_j] (Lemma A.1 of the paper).
For every pulse of Theorem 1 (K* in [K1, K2], leaving rest through the exit set E at xi = 0 and staying in B0 from
T_enter on), x(xi) lies in W_j for xi in [t_j, t_j + h_j] and in X_j at xi = t_j (Appendix A of the paper).

It also finds T_c: the start of the earliest step from which every step's whole path (Lemma A.4, lohner6.step_range,
mapped to zeta = M (y - y*)) lies in the interior of B0, up to T_enter. Then every pulse of Theorem 1 lies in B0 for all
xi >= T_c (from T_enter on by Theorem 1's proof, Lemma 2). The check is repeated for every step and stops the program if
the enclosure at T_enter is not in int B0 (as (H4) requires).

Output: ../data/logs/stab_record_<tag>.pkl (the boxes, exact serialization; regenerated, not tracked) and
../data/stab_record_<tag>.json (steps, T_c, the sha256 of the pickle, the zeta hull at T_enter).
Usage: HH_EL=10.613 python3 stab_record.py 18.5
"""
import hashlib
import json
import os
import pickle
import sys
import time
from flint import arb, ctx
import prove_pulse as PP
import lohner6 as L
import certify_rest_wave as C


def main(T):
    t_start = time.time()
    ctx.prec = PP.PREC
    S = PP.Setup(T)
    lam = float(arb(json.load(open(PP.out_path(T, 'setup')))['lambda_u']).mid())
    B = S.lemma_B()
    PP.require(B['ok'], 'Lemma B')
    X = S.exit_set(S.Kball, B)
    F = L.Field(S.phi, S.EL)
    recs = []

    def cb(tp, t, Xh, Xn, W, hh):
        recs.append((L.ser(tp), L.ser(t - tp), [L.ser(v) for v in Xh], [L.ser(v) for v in W],
                     [L.ser(v) for v in Xn.hull()]))
        if len(recs) % 100 == 0:
            print('step %d t = %.4f (%.0f s)' % (len(recs), float(t.mid()), time.time() - t_start), flush=True)
        hx = Xn.hull()
        wid = max(float(arb(x.rad()).mid()) for x in hx[:5])
        PP.require(wid < 1e3 and all(x.is_finite() for x in hx), 'the set blew up')
        return False
    X, t, ns = L.integrate(F, X, S.T_enter, PP.ORDER, S.tol(lam), hmax=0.25, t0=0.0, callback=cb)
    PP.require(bool(t == arb(S.T_enter)), 'did not reach T_enter exactly')
    z = S.zeta(X)
    PP.require(S.in_int_B0(z), 'the enclosure at T_enter is not in int B0')
    print('T_enter = %s reached in %d steps; zeta = %s, |zeta_s| <= %s' % (S.T_enter, ns, [v.str(6) for v in z],
                                                                           S.norm_s_upper(z).str(6)), flush=True)
    # T_c: scan the steps backward from T_enter while the whole path of the step is in int B0
    j0 = len(recs)
    paths = []
    ctx.prec = PP.PREC
    for j in range(len(recs) - 1, -1, -1):
        tp, hs, Xh, W, _ = recs[j]
        Xh = [L.deser(v) for v in Xh]
        W = [L.deser(v) for v in W]
        h = L.deser(hs)
        zr = L.step_range(F, Xh, W, arb(h.upper()), PP.ORDER, S.M6, S.shift6)
        if not S.in_int_B0(zr):
            break
        paths.append({'t': float(L.deser(tp).mid()), 'zeta1': zr[0].str(6), 'norm_s_upper': S.norm_s_upper(zr).str(8)})
        j0 = j
    PP.require(j0 < len(recs), 'no step before T_enter has its path in int B0')
    T_c = L.deser(recs[j0][0])
    print('T_c = %s: the paths of steps %d..%d (to T_enter) lie in int B0' % (T_c.str(10), j0, len(recs) - 1),
          flush=True)
    os.makedirs('../data/logs', exist_ok=True)
    pk = '../data/logs/stab_record_%s.pkl' % C.tag(T)
    blob = pickle.dumps({'T': T, 'recs': recs, 'j_c': j0, 'cfg': S.cfg})
    open(pk, 'wb').write(blob)
    out = {'T': T, 'tag': C.tag(T), 'steps': len(recs), 'T_enter': S.T_enter, 'j_c': j0, 'T_c': T_c.str(20),
           'T_c_float': float(T_c.mid()), 'zeta_at_T_enter': [v.str(10) for v in z],
           'norm_s_upper_at_T_enter': S.norm_s_upper(z).str(10), 'paths_in_B0_from_T_c': paths[::-1],
           'sha256_pkl': hashlib.sha256(blob).hexdigest(), 'K1': S.cfg['K1_dec'], 'K2': S.cfg['K2_dec'],
           'secs': round(time.time() - t_start)}
    json.dump(out, open('../data/stab_record_%s.json' % C.tag(T), 'w'), indent=1)
    print('RECORD WRITTEN (%d s)' % out['secs'])
    return True


if __name__ == '__main__':
    sys.exit(0 if main(float(sys.argv[1])) else 1)
