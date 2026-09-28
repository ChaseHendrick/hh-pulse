#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Recompute the axon speeds from the K endpoint decimals, without flint.

The speed in the pulse summaries is

    theta = sqrt(K * 1000 * a_cm / (2 * R2 * CM)) / 100

with a_cm = 0.0238, R2 = 35.4, CM = 1e-6, K in 1/ms and theta in m/s.
The constants and the formula are typed here. This program does not import
the proof.

K1 and K2 in the configuration JSON are dyadic mantissas. The decimal
endpoints are the strings K1_dec and K2_dec. Each is turned into the exact
rational value of the radicand. decimal.sqrt is not used: the stored speed
intervals are only a few units in the last place wide, so a rounded point can
fall on the wrong side of an endpoint. The square root is enclosed by finding
an integer L with

    L^2 * denom <= radicand_numer * 10^{2N} < (L + 1)^2 * denom,

which is L * 10^{-N} <= sqrt(radicand) < (L + 1) * 10^{-N}. Dividing by 100
is a further exact shift of the decimal point. The working Decimal precision
is at least 80 and is high enough that those enclosure endpoints are exact
Decimals.

Exit 0 only if, at both 18.5 C and 6.3 C, each endpoint's enclosure lies in
the stored speed interval, the images are ordered as the K endpoints are,
and K * (1 + 1e-6) leaves that interval.
"""
import json
import math
import os
import re
import sys
from decimal import Decimal, getcontext

# Above 80 so an enclosure a little finer than the 6.3 certificate is exact.
getcontext().prec = 120

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

# Fractional digits of the square-root enclosure. theta = sqrt / 100, so the
# speed enclosure is one unit in the 10^{-(SQRT_PLACES + 2)} place.
SQRT_PLACES = 88

CASES = (
    ('18.5_El10.613', '18.5 C'),
    ('6.3_El10.613', '6.3 C'),
)


def scaled_digits(value):
    """Return (digits, places) with value = digits * 10^{-places}, places >= 0."""
    parts = Decimal(value).as_tuple()
    if parts.sign:
        raise ValueError('expected a non-negative decimal, got %s' % value)
    if not isinstance(parts.exponent, int):
        raise ValueError('expected a finite decimal, got %s' % value)
    numer = 0
    for digit in parts.digits:
        numer = numer * 10 + digit
    if parts.exponent >= 0:
        return numer * 10 ** parts.exponent, 0
    return numer, -parts.exponent


def decimal_pair(value):
    """Exact (numer, denom) for a finite non-negative Decimal."""
    numer, places = scaled_digits(value)
    return numer, 10 ** places


def radicand(k_numer, k_denom):
    """Exact numer/denom of K * 1000 * a_cm / (2 * R2 * CM)."""
    a_n, a_d = decimal_pair(Decimal('0.0238'))
    r_n, r_d = decimal_pair(Decimal('35.4'))
    c_n, c_d = decimal_pair(Decimal('1e-6'))
    numer = k_numer * 1000 * a_n * r_d * c_d
    denom = k_denom * a_d * 2 * r_n * c_n
    g = math.gcd(numer, denom)
    return numer // g, denom // g


def sqrt_lower_index(numer, denom, places):
    """Largest L with (L / 10^places)^2 <= numer/denom < ((L+1) / 10^places)^2.

    math.isqrt(numer * 10^{2 places} // denom) is that L: if q is the
    quotient, isqrt(q)^2 <= q and (isqrt(q)+1)^2 >= q + 1, so multiplying
    through by the positive denominator puts the scaled radicand in
    [L^2, (L+1)^2).
    """
    if numer <= 0 or denom <= 0:
        raise ValueError('the radicand must be positive')
    target = numer * 10 ** (2 * places)
    L = math.isqrt(target // denom)
    if L * L * denom > target or (L + 1) * (L + 1) * denom <= target:
        raise RuntimeError('integer square-root enclosure is not proved')
    return L


def speed_enclosure(k_numer, k_denom):
    """Rigorous [lower, upper) for theta, as integers over 10^scale.

    lower/10^scale <= theta < upper/10^scale, upper = lower + 1.
    """
    numer, denom = radicand(k_numer, k_denom)
    L = sqrt_lower_index(numer, denom, SQRT_PLACES)
    scale = SQRT_PLACES + 2  # divide the square root by 100
    return L, L + 1, scale


def dec_from_scaled(numer, scale):
    """Exact Decimal numer * 10^{-scale}. Precision must hold every digit."""
    out = Decimal(numer).scaleb(-scale)
    if out.scaleb(scale) != Decimal(numer):
        raise RuntimeError('Decimal precision %s rounded an enclosure endpoint' % getcontext().prec)
    return out


def scaled_str(numer, scale):
    return format(dec_from_scaled(numer, scale), 'f')


def less_scaled(a_numer, a_scale, b_numer, b_scale):
    """a_numer / 10^a_scale < b_numer / 10^b_scale, exactly."""
    return a_numer * 10 ** b_scale < b_numer * 10 ** a_scale


def inside_open(lo_numer, hi_numer, scale, stored_lo, stored_hi):
    """True when [lo, hi) / 10^scale lies in (stored_lo, stored_hi)."""
    slo_n, slo_p = scaled_digits(stored_lo)
    shi_n, shi_p = scaled_digits(stored_hi)
    return (less_scaled(slo_n, slo_p, lo_numer, scale)
            and less_scaled(hi_numer, scale, shi_n, shi_p))


def above_open(lo_numer, scale, stored_hi):
    """True when lo/10^scale > stored_hi, so the whole enclosure is above it."""
    shi_n, shi_p = scaled_digits(stored_hi)
    return less_scaled(shi_n, shi_p, lo_numer, scale)


def agreement(stored_s, lo_numer, hi_numer, scale):
    """Leading significant digits shared with both ends of the enclosure."""
    def sig_prefix(text):
        n = 0
        for left, right in zip(stored_s, text):
            if left != right:
                break
            n += 1
        body = stored_s[:n]
        if body.endswith('.'):
            body = body[:-1]
        return len(body.replace('.', '').lstrip('+-')), body

    lo_n, lo_body = sig_prefix(scaled_str(lo_numer, scale))
    hi_n, hi_body = sig_prefix(scaled_str(hi_numer, scale))
    if lo_n <= hi_n:
        return lo_n, lo_body
    return hi_n, hi_body


def load_case(tag):
    cfg = json.load(open(os.path.join(DATA, 'pulse_proof_%s_config.json' % tag)))
    summary = open(os.path.join(DATA, 'pulse_proof_%s_summary.txt' % tag)).read()
    found = re.search(r'speed theta in \(([0-9.]+), ([0-9.]+)\) m/s', summary)
    if not found:
        raise RuntimeError('%s summary has no speed interval' % tag)
    if 'K1_dec' not in cfg or 'K2_dec' not in cfg:
        raise RuntimeError('%s config has no decimal K endpoints' % tag)
    return cfg['K1_dec'], cfg['K2_dec'], found.group(1), found.group(2)


def main():
    if getcontext().prec < 80:
        print('decimal precision %s is below 80' % getcontext().prec)
        return 1
    lines = []

    def say(text=''):
        lines.append(text)

    say('theta = sqrt(K * 1000 * a_cm / (2 * R2 * CM)) / 100')
    say('a_cm = 0.0238, R2 = 35.4, CM = 1e-6')
    say('K in 1/ms; theta in m/s')
    say('decimal precision %d' % getcontext().prec)
    say('square root enclosed by integer squares at 10^{-%d}; theta enclosure width 1e-%d'
        % (SQRT_PLACES, SQRT_PLACES + 2))
    say('K endpoints are the config strings K1_dec and K2_dec')

    one_plus = Decimal(1) + Decimal('1e-6')
    shift_n, shift_d = decimal_pair(one_plus)
    if Decimal(shift_n) / Decimal(shift_d) != one_plus:
        say('1 + 1e-6 is not the rational 1000001/1000000 at this precision')
        print('\n'.join(lines))
        return 1

    ok = True
    for tag, title in CASES:
        say()
        say('%s (%s)' % (title, tag))
        k1_s, k2_s, stored_lo_s, stored_hi_s = load_case(tag)
        k1 = Decimal(k1_s)
        k2 = Decimal(k2_s)
        stored_lo = Decimal(stored_lo_s)
        stored_hi = Decimal(stored_hi_s)
        k1_n, k1_d = decimal_pair(k1)
        k2_n, k2_d = decimal_pair(k2)
        lo1, hi1, scale = speed_enclosure(k1_n, k1_d)
        lo2, hi2, scale2 = speed_enclosure(k2_n, k2_d)
        if scale != scale2:
            raise RuntimeError('enclosures were built at different scales')

        in1 = inside_open(lo1, hi1, scale, stored_lo, stored_hi)
        in2 = inside_open(lo2, hi2, scale, stored_lo, stored_hi)
        n1, pref1 = agreement(stored_lo_s, lo1, hi1, scale)
        n2, pref2 = agreement(stored_hi_s, lo2, hi2, scale)
        same_order = ((k1 < k2 and lo2 > hi1) or (k2 < k1 and lo1 > hi2))

        sk1_n, sk1_d = k1_n * shift_n, k1_d * shift_d
        sk2_n, sk2_d = k2_n * shift_n, k2_d * shift_d
        slo1, shi1, sscale = speed_enclosure(sk1_n, sk1_d)
        slo2, shi2, sscale2 = speed_enclosure(sk2_n, sk2_d)
        if sscale != scale or sscale2 != scale:
            raise RuntimeError('shifted enclosure scale drifted')
        leaves = above_open(slo1, scale, stored_hi) and above_open(slo2, scale, stored_hi)

        say('  K1 = %s /ms' % k1_s)
        say('  K2 = %s /ms' % k2_s)
        say('  stored speed (%s, %s) m/s' % (stored_lo_s, stored_hi_s))
        say('  theta(K1) enclosure [%s, %s)' % (scaled_str(lo1, scale), scaled_str(hi1, scale)))
        say('  theta(K2) enclosure [%s, %s)' % (scaled_str(lo2, scale), scaled_str(hi2, scale)))
        say('  theta(K1) inside the stored interval: %s' % ('yes' if in1 else 'NO'))
        say('  theta(K2) inside the stored interval: %s' % ('yes' if in2 else 'NO'))
        say('  theta(K1) agrees with the stored lower endpoint for %d leading digits' % n1)
        say('    %s' % pref1)
        say('  theta(K2) agrees with the stored upper endpoint for %d leading digits' % n2)
        say('    %s' % pref2)
        say('  larger K gives larger speed: %s' % ('yes' if same_order else 'NO'))
        say('  negative control K * (1 + 1e-6)')
        say('    theta(K1 shifted) enclosure [%s, %s)' % (scaled_str(slo1, scale), scaled_str(shi1, scale)))
        say('    theta(K2 shifted) enclosure [%s, %s)' % (scaled_str(slo2, scale), scaled_str(shi2, scale)))
        say('  shifted speed leaves the stored interval: %s' % ('yes' if leaves else 'NO'))
        case_ok = in1 and in2 and same_order and leaves
        say('  %s' % ('ok' if case_ok else 'FAILED'))
        ok = ok and case_ok

    say()
    say('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    print('\n'.join(lines))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
