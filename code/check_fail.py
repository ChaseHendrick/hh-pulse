#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Check the Hodgkin–Huxley pulse FAIL certificates against the manuscript.

Quoted from paper/paper.md, "(H3) The closing block" and
"(H4) The interval run" (the entrance condition at T_enter):
  B0 = {|zeta_1| <= r, |zeta_s|_2 <= rho},
  rho = 0.8, r = 0.84 (18.5 C) and rho = 0.6, r = 0.63 (6.3 C),
  as binary floating-point numbers that the programs use exactly:
  rho = 0.8000000000000000444..., r = 0.8 x 1.05 rounded to
  0.8400000000000000799..., and rho = 0.5999999999999999777...,
  r = 0.6300000000000000044....
  (H4) encloses the set at T_enter in the interior of B0, so the condition
  checked here is |zeta_1| < r and |zeta_s| < rho.
The numbers below are read from that paragraph. They are not invented, and
they are not the achieved interval-run enclosures printed later in the tables.
"""
import json
import os
import re
import struct
import sys

from flint import arb, ctx

ctx.prec = 256

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')
PAPER = os.path.join(ROOT, 'paper', 'paper.md')

# An entrance miss (zeta_1 or |zeta_s| outside int B0) is stored under this
# fail string. It is the string those certificates use, not a label made up here.
ENTRANCE_FAIL = 'outside_int_B0_at_T_enter'
ESCAPE_FAIL = 'escaped_below_-60'
# Parser control only. Never the bound the certificates are required to meet.
LOOSE = 1000


def binary64_from_prefix(prefix):
    """The unique IEEE-754 binary64 whose decimal expansion begins with prefix."""
    center = float(prefix)
    bits = struct.unpack('<Q', struct.pack('<d', center))[0]
    found = []
    for delta in range(-5, 6):
        raw = (bits + delta) & ((1 << 64) - 1)
        f = struct.unpack('<d', struct.pack('<Q', raw))[0]
        if format(f, '.80f').startswith(prefix):
            found.append(f)
    if len(found) != 1:
        raise SystemExit('manuscript prefix %r matched %s' % (prefix, found))
    return found[0]


def read_bounds(path):
    """Entrance radii at T_enter, from the (H3)/(H4) paragraph of the manuscript."""
    flat = re.sub(r'\s+', ' ', open(path, encoding='utf-8').read())
    found = re.search(
        r'let B0 = \{\|zeta_1\| <= r, \|zeta_s\|_2 <= rho\}'
        r'.*?rho = 0\.8, r = 0\.84 \(18\.5 C\) and rho = 0\.6, r = 0\.63 \(6\.3 C\), '
        r'as binary floating-point numbers that the programs use exactly: '
        r'rho = (0\.8[0-9]*)\.\.\., r = 0\.8 x 1\.05 rounded to (0\.84[0-9]*)\.\.\., and rho = '
        r'(0\.59[0-9]*)\.\.\., r = (0\.63[0-9]*)\.\.\.',
        flat)
    if not found:
        raise SystemExit('entrance bounds not found in %s' % path)
    if 'at T_enter, in the interior of B0' not in flat:
        raise SystemExit('%s does not state the interior entrance at T_enter' % path)
    rho185_p, r185_p, rho63_p, r63_p = found.groups()
    rho185 = binary64_from_prefix(rho185_p)
    r185 = binary64_from_prefix(r185_p)
    rho63 = binary64_from_prefix(rho63_p)
    r63 = binary64_from_prefix(r63_p)
    # The manuscript names r at 18.5 C as 0.8 x 1.05 rounded to that expansion.
    if r185 != (0.8 * 1.05) or rho185 != 0.8 or rho63 != 0.6 or r63 != 0.63:
        raise SystemExit('parsed radii are not the binary64 values named in the manuscript')
    return {
        18.5: {'r': r185, 'rho': rho185, 'r_prefix': r185_p, 'rho_prefix': rho185_p},
        6.3: {'r': r63, 'rho': rho63, 'r_prefix': r63_p, 'rho_prefix': rho63_p},
    }


def bounds_for(T, table):
    for key, val in table.items():
        if T == key:
            return val
    raise SystemExit('no manuscript entrance bound for T = %r' % (T,))


def parse_ball(text):
    """Flint enclosure of a stored '[mid +/- rad]' ball. Parsing only widens it."""
    try:
        ball = arb(text)
    except (TypeError, ValueError) as exc:
        raise SystemExit('cannot parse ball %r (%s)' % (text, exc))
    if not ball.is_finite():
        raise SystemExit('ball %r is not finite' % text)
    return ball


def norm_s(components):
    total = sum((v * v for v in components), arb(0))
    return total.sqrt()


def abs_inside(z, bound):
    return bool(z.abs_upper() < arb(bound))


def abs_outside(z, bound):
    """True when the whole interval lies strictly outside (-bound, bound)."""
    return bool(z.abs_lower() > arb(bound))


def classify(z1, zs_upper, zs_components, r, rho):
    """Status of the two entrance quantities against |zeta_1| < r and |zeta_s| < rho."""
    out = {}
    if abs_outside(z1, r):
        out['zeta_1'] = ('outside', z1.abs_lower() - arb(r))
    elif abs_inside(z1, r):
        out['zeta_1'] = ('inside', arb(r) - z1.abs_upper())
    else:
        out['zeta_1'] = ('unknown', None)
    nrm = norm_s(zs_components)
    # |zeta_s|_upper proves the norm is inside when the whole upper-bound ball is < rho.
    # The component intervals prove it is outside when their norm lies above rho.
    if bool(zs_upper < arb(rho)) or abs_inside(nrm, rho):
        slack = arb(rho) - zs_upper.abs_upper()
        out['|zeta_s|'] = ('inside', slack)
    elif bool(nrm > arb(rho)):
        out['|zeta_s|'] = ('outside', nrm.abs_lower() - arb(rho))
    else:
        out['|zeta_s|'] = ('unknown', None)
    return out


def rel(path):
    return os.path.relpath(path, ROOT)


def sibling(path, stage, new_stage):
    base = os.path.basename(path)
    suffix = '_%s.json' % stage
    if not base.startswith('pulse_proof_') or not base.endswith(suffix):
        raise SystemExit('unexpected certificate name %s' % base)
    tag = base[len('pulse_proof_'):-len(suffix)]
    return os.path.join(os.path.dirname(path), 'pulse_proof_%s_%s.json' % (tag, new_stage))


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def zeta_parts(cert, path):
    raw = cert.get('zeta_at_T_enter')
    if not isinstance(raw, list) or len(raw) != 5:
        raise SystemExit('%s zeta_at_T_enter is not five intervals' % rel(path))
    z = [parse_ball(s) for s in raw]
    key = '|zeta_s|_upper_at_T_enter'
    if key not in cert:
        raise SystemExit('%s has zeta but no %s' % (rel(path), key))
    return z, parse_ball(cert[key]), raw


def main():
    bounds = read_bounds(PAPER)
    lines = []
    bad = False

    def say(text=''):
        lines.append(text)

    say('entrance bounds from paper/paper.md')
    say('  phrase: "(H3) The closing block" and "(H4) The interval run"')
    say('  at T_enter the enclosure lies in the interior of B0: |zeta_1| < r and |zeta_s| < rho')
    for T in (18.5, 6.3):
        b = bounds[T]
        say('  %s C: r = %s... (binary64 %s), rho = %s... (binary64 %s)' % (
            T, b['r_prefix'], b['r'].hex(), b['rho_prefix'], b['rho'].hex()))

    paths = []
    for name in sorted(os.listdir(DATA)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(DATA, name)
        cert = load_json(path)
        if isinstance(cert, dict) and cert.get('verdict') == 'FAIL':
            paths.append(path)
    if not paths:
        say('no FAIL certificates')
        print('\n'.join(lines))
        return 1

    guilty_names = []
    n_shift = 0
    n_escape = 0

    for path in paths:
        cert = load_json(path)
        say()
        say('FAIL %s' % rel(path))
        reason = cert.get('reason', '')
        escaped = ('escap' in reason) and ('-60' in reason)
        has_zeta = 'zeta_at_T_enter' in cert
        if not has_zeta and not escaped:
            say('  neither zeta_at_T_enter nor an escape below -60')
            bad = True
            continue

        if has_zeta:
            n_shift += 1
            b = bounds_for(cert['T'], bounds)
            z, zs_upper, raw = zeta_parts(cert, path)
            status = classify(z[0], zs_upper, z[1:], b['r'], b['rho'])
            guilty = [name for name, (kind, _) in status.items() if kind == 'outside']
            unknown = [name for name, (kind, _) in status.items() if kind == 'unknown']
            guilty_names.append(tuple(guilty))
            say('  T = %s C, T_enter = %s' % (cert['T'], cert['T_enter']))
            for i, text in enumerate(raw):
                say('  zeta_%d = %s' % (i + 1, text))
            say('  |zeta_s|_upper = %s' % cert['|zeta_s|_upper_at_T_enter'])
            for name in ('zeta_1', '|zeta_s|'):
                kind, margin = status[name]
                if kind == 'outside':
                    say('  %s misses the entrance condition; margin (how far outside) = %s' % (
                        name, margin.str(12)))
                elif kind == 'inside':
                    say('  %s meets the entrance condition' % name)
                else:
                    say('  %s is not decided against the entrance condition' % name)
            if unknown or not guilty:
                say('  computed coordinate: %s' % (', '.join(guilty) if guilty else 'none'))
                say('  file fail string: %s' % cert.get('fail'))
                bad = True
            else:
                say('  guilty coordinate: %s' % ', '.join(guilty))
                fail = cert.get('fail')
                flag = cert.get('in_int_B0_at_T_enter')
                say('  in_int_B0_at_T_enter: %s' % flag)
                say('  fail string: %s' % fail)
                if flag is not False or fail != ENTRANCE_FAIL:
                    say('  MISMATCH computed coordinate: %s' % ', '.join(guilty))
                    say('  MISMATCH file fail string: %s' % fail)
                    bad = True

            loose = classify(z[0], zs_upper, z[1:], b['r'] * LOOSE, b['rho'] * LOOSE)
            loose_in = all(kind == 'inside' for kind, _ in loose.values())
            say('  loosened entrance bound x%s (parser control, not required of the certificate): inside = %s' % (
                LOOSE, 'yes' if loose_in else 'NO'))
            if not loose_in:
                say('  the stored zeta is still outside the loosened bound; the parser is wrong')
                bad = True

            k1_path = sibling(path, cert.get('stage', ''), 'K1')
            if not os.path.isfile(k1_path):
                say('  no PASS K1 certificate at %s' % rel(k1_path))
                bad = True
            else:
                k1 = load_json(k1_path)
                if k1.get('verdict') != 'PASS' or k1.get('T') != cert.get('T'):
                    say('  %s is not a PASS certificate at the same temperature' % rel(k1_path))
                    bad = True
                else:
                    kz, kzs, _ = zeta_parts(k1, k1_path)
                    kb = bounds_for(k1['T'], bounds)
                    kstatus = classify(kz[0], kzs, kz[1:], kb['r'], kb['rho'])
                    k_ok = all(kind == 'inside' for kind, _ in kstatus.values())
                    say('  PASS %s meets the manuscript entrance bounds: %s' % (
                        rel(k1_path), 'yes' if k_ok else 'NO'))
                    if not k_ok:
                        for name, (kind, _) in kstatus.items():
                            say('    %s: %s' % (name, kind))
                        bad = True

        if escaped:
            n_escape += 1
            say('  reason: %s' % reason)
            found = re.search(r't\s*=\s*(\[[^\]]+\])', reason)
            if not found:
                say('  the reason does not contain a time')
                bad = True
            else:
                t_raw = found.group(1)
                t = parse_ball(t_raw)
                Te = arb(cert['T_enter'])
                gap = Te - t
                before = bool(t.upper() < Te)
                say('  time t = %s' % t_raw)
                say('  gap T_enter - t = %s' % gap.str(12))
                say('  t is before T_enter: %s' % ('yes' if before else 'NO'))
                if not before:
                    bad = True
            fail = cert.get('fail')
            if fail != ESCAPE_FAIL:
                say('  MISMATCH computed miss: escaped below -60')
                say('  MISMATCH file fail string: %s' % fail)
                bad = True

    say()
    if guilty_names and all(names == ('zeta_1',) for names in guilty_names):
        say('shift controls: guilty coordinate is zeta_1')
    else:
        say('shift controls: guilty coordinates %s' % (guilty_names,))
    if n_shift == 0 or n_escape == 0:
        say('expected both a zeta FAIL and an escape-below--60 FAIL')
        bad = True
    say('ALL CHECKS PASSED' if not bad else 'SOME CHECK FAILED')
    print('\n'.join(lines))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
