#!/bin/sh
# Tests of tstrip.py (printed leak potential, reference temperature 18.5 C). Exit status 0 iff all behave as expected.
#  1. degenerate interval T = [18.5, 18.5], |s| <= 1e-6: must pass (a pulse with K within 1e-6 of the numerical
#     centre, consistent with the proof of hh_prove_pulse.py, whose [K1, K2] lies inside that range);
#  2. negative control: the same with the speed interval shifted by 1e-3 (it no longer contains the pulse): must fail;
#  3. negative control: alpha_m multiplied by 1 + 1e-8 u^2 (TSTRIP_NEG=model), which moves the pulse speed by far more
#     than 1e-6: must fail.
# Needs data/tstrip_ref_18.5_El10.613.json (python3 tstrip.py reference 18.5 1e-5 7.97, with HH_EL=10.613).
cd "$(dirname "$0")"
export HH_EL=10.613
ok=0
nice -n 19 timeout 3600 python3 tstrip.py piece 18.5 18.5 18.5 1e-6 > ../data/logs/test_tstrip_1.out 2>&1 || ok=1
nice -n 19 timeout 3600 python3 tstrip.py piece 18.5 18.5 18.5 1e-6 1e-3 > ../data/logs/test_tstrip_2.out 2>&1 && ok=1
TSTRIP_NEG=model nice -n 19 timeout 3600 python3 tstrip.py piece 18.5 18.5 18.5 1e-6 > ../data/logs/test_tstrip_3.out 2>&1 && ok=1
python3 - <<'EOF'
import json
from flint import arb, ctx
ctx.prec = 256
import lohner6 as L
d = json.load(open('../data/tstrip_El10.613_18.5_18.5.json'))
cfg = json.load(open('../data/pulse_proof_18.5_El10.613_config.json'))
ref = json.load(open('../data/tstrip_ref_18.5_El10.613.json'))
K1, K2, Kc = L.deser(cfg['K1']), L.deser(cfg['K2']), arb(ref['K_c'])
inside = bool(abs(K1 - Kc) < arb('1e-6')) and bool(abs(K2 - Kc) < arb('1e-6'))
print('degenerate piece passed:', d['all'], '; [K1, K2] of hh_prove_pulse.py inside K_c +- 1e-6:', inside)
for f, name in (('../data/tstrip_El10.613_18.5_18.5_shift0.001.json', 'shifted speed interval'),
                ('../data/tstrip_El10.613_18.5_18.5_negmodel.json', 'perturbed alpha_m')):
    e = json.load(open(f))
    print('negative control (%s) failed as it must:' % name, not e['all'])
EOF
[ $ok -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TEST FAILED"
exit $ok
