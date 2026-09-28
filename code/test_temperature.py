#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the temperature factor phi = 3^((T - 6.3)/10) (notes/review-1.md, M1).

The programs once read the temperature with float(sys.argv[1]); the float 6.3 is 6.29999999999999982236431605997... C,
so phi was 1 - 1.95e-17, a ball that excludes 1 at 256 bits, and the 6.3 C proof was a proof at that temperature.
Now the temperature is the decimal string as typed, from the command line to phi. Checks:
 1. certify_rest_wave.temperature('6.3') is '6.3', and phi_of('6.3') contains 1, with a radius below 1e-70;
 2. phi_of(6.3), given a float, also contains 1 (a float is read as its shortest decimal, repr(6.3) = '6.3');
 3. phi_of('18.5') is the same ball, midpoint and radius, as the old float path gave (18.5 is a binary number, so
    the 18.5 C certificates are not affected);
 4. hh_block_check_iv.phi_iv('6.3') (mpmath intervals, separate code) contains 1;
 5. no program that computes phi parses the temperature with float(): hp_pulse.py, block0.py, hh_prove_pulse.py and
    certify_rest_wave.py call temperature(sys.argv[1]) (or argv[1]); hh_block_check_iv.py checks the decimal string;
 6. temperature() refuses a float and a string that is not a plain decimal number.
Negative control: on the old float path, arb(3) ** ((arb(6.3) - arb('6.3')) / 10) excludes 1, so check 1 or 2 fails
on it (this test would have caught the defect).
Output: data/test_temperature.txt; exit status 0 iff every check passes and the control fails.
"""
import os
import re
import sys
from flint import arb, ctx
import certify_rest_wave as C
import hh_block_check_iv as BIV

ctx.prec = 256
HERE = os.path.dirname(os.path.abspath(__file__))
lines = []


def check(name, ok):
    lines.append('%-78s %s' % (name, 'ok' if ok else 'FAILED'))
    return bool(ok)


def main():
    ok = True
    T = C.temperature('6.3')
    p = C.phi_of(T)
    ok &= check('1. temperature(\'6.3\') == \'6.3\'; phi_of(\'6.3\') - 1 = %s contains 0' % (p - 1).str(3),
                T == '6.3' and p.contains(1) and p.rad() < arb('1e-70'))
    ok &= check('2. phi_of(6.3) (a float) contains 1', C.phi_of(6.3).contains(1))
    old18 = arb(3) ** ((arb(18.5) - arb('6.3')) / 10)
    new18 = C.phi_of('18.5')
    ok &= check('3. phi_of(\'18.5\') equals the old ball %s' % old18.str(12),
                old18.mid() == new18.mid() and old18.rad() == new18.rad())
    ok &= check('4. hh_block_check_iv.phi_iv(\'6.3\') contains 1', 1 in BIV.phi_iv('6.3'))
    bad = []
    for f in ('hp_pulse.py', 'block0.py', 'hh_prove_pulse.py', 'certify_rest_wave.py', 'hh_block_check_iv.py'):
        src = open(os.path.join(HERE, f)).read()
        if re.search(r'float\(\s*(sys\.)?argv\[1\]\s*\)', src):
            bad.append(f)
    ok &= check('5. no program parses the temperature with float()%s' % (': ' + ', '.join(bad) if bad else ''),
                not bad)
    refused = 0
    for v in (6.3, '6.3e0', '6,3', ' 6.3', ''):
        try:
            C.temperature(v)
        except ValueError:
            refused += 1
    ok &= check('6. temperature() refuses a float and non-decimal strings (%d of 5)' % refused, refused == 5)
    old63 = arb(3) ** ((arb(6.3) - arb('6.3')) / 10)
    ctrl = not old63.contains(1)
    lines.append('negative control: the old float path gives phi - 1 = %s, which excludes 0: %s' % ((old63 - 1).str(5),
                                                                                                  ctrl))
    ok = ok and ctrl
    lines.append('ALL TESTS PASSED' if ok else 'SOME TEST FAILED')
    print('\n'.join(lines))
    open(os.path.join(HERE, '..', 'data', 'test_temperature.txt'), 'w').write('\n'.join(lines) + '\n')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
