#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Check the abstract speeds, K widths, and printed-E_l speed shift.

Standard library only. Endpoints come from the proof summaries and the
decimal strings in the configuration JSON. Nothing here imports the proof.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')
PAPER = os.path.join(ROOT, 'paper', 'paper.md')

SPEED_LINE = re.compile(r'speed theta in \(([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+)\) m/s')
SCI = re.compile(r'([+-]?)(\d+)(?:\.(\d+))?[eE]([+-]?\d+)')


class Dec:
    """Exact decimal, value = num / 10**scale, trailing zeros removed."""

    def __init__(self, num, scale):
        num = int(num)
        scale = int(scale)
        if scale < 0:
            num *= 10 ** (-scale)
            scale = 0
        if num == 0:
            self.num = 0
            self.scale = 0
            return
        sign = -1 if num < 0 else 1
        num = abs(num)
        while scale > 0 and num % 10 == 0:
            num //= 10
            scale -= 1
        self.num = sign * num
        self.scale = scale

    @staticmethod
    def parse(text):
        s = text.strip()
        sign = 1
        if s[0] == '-':
            sign = -1
            s = s[1:]
        elif s[0] == '+':
            s = s[1:]
        if not re.fullmatch(r'\d+(\.\d+)?', s):
            raise ValueError('not a decimal: %r' % text)
        if '.' in s:
            whole, frac = s.split('.')
            return Dec(sign * int(whole + frac), len(frac))
        return Dec(sign * int(s), 0)

    @staticmethod
    def parse_sci(token):
        found = SCI.fullmatch(token.strip())
        if not found:
            raise ValueError('not scientific notation: %r' % token)
        sign_s, whole, frac, exp = found.groups()
        frac = frac or ''
        num = int(whole + frac)
        if sign_s == '-':
            num = -num
        return Dec(num, len(frac) - int(exp))

    def __add__(self, other):
        scale = max(self.scale, other.scale)
        a = self.num * 10 ** (scale - self.scale)
        b = other.num * 10 ** (scale - other.scale)
        return Dec(a + b, scale)

    def __sub__(self, other):
        scale = max(self.scale, other.scale)
        a = self.num * 10 ** (scale - self.scale)
        b = other.num * 10 ** (scale - other.scale)
        return Dec(a - b, scale)

    def __abs__(self):
        return Dec(abs(self.num), self.scale)

    def __lt__(self, other):
        return (self - other).num < 0

    def div2(self):
        if self.num % 2 == 0:
            return Dec(self.num // 2, self.scale)
        return Dec(self.num * 5, self.scale + 1)

    def exact_sci(self):
        if self.num == 0:
            return '0e0'
        s = str(abs(self.num))
        exp = len(s) - 1 - self.scale
        mant = s if len(s) == 1 else s[0] + '.' + s[1:]
        sign = '-' if self.num < 0 else ''
        return '%s%se%d' % (sign, mant, exp)

    def round_sig(self, sig):
        """Round half-even to `sig` significant digits. Returns (mantissa, exp)."""
        if sig < 1:
            raise ValueError('sig')
        if self.num == 0:
            return '0' if sig == 1 else '0.' + '0' * (sig - 1), 0
        s = str(abs(self.num))
        exp = len(s) - 1 - self.scale
        if len(s) <= sig:
            head = s + ('0' * (sig - len(s)))
            return head, exp
        head, rest = s[:sig], s[sig:]
        first = int(rest[0])
        if first > 5:
            up = True
        elif first < 5:
            up = False
        elif any(c != '0' for c in rest[1:]):
            up = True
        else:
            up = (int(head[-1]) % 2 == 1)
        extra = 0
        if up:
            bumped = int(head) + 1
            if len(str(bumped)) > sig:
                head = '1' + '0' * (sig - 1)
                extra = 1
            else:
                head = str(bumped).zfill(sig)
        return head, exp + extra


def format_sig(head, exp):
    mant = head if len(head) == 1 else head[0] + '.' + head[1:]
    return '%se%d' % (mant, exp)


def canonical_sci(token):
    found = SCI.fullmatch(token.strip())
    if not found:
        return None
    sign_s, whole, frac, exp = found.groups()
    frac = frac or ''
    mant = whole if not frac else whole + '.' + frac
    if sign_s == '+':
        sign_s = ''
    return '%s%se%d' % (sign_s, mant, int(exp))


def sig_digits(token):
    found = SCI.fullmatch(token.strip())
    if not found:
        raise ValueError(token)
    whole, frac = found.group(2), found.group(3) or ''
    return len((whole + frac).lstrip('0') or '0')


def shared_prefix(lo, hi):
    """Longest shared prefix. A digit the endpoints do not share is not counted."""
    n = 0
    while n < len(lo) and n < len(hi) and lo[n] == hi[n]:
        n += 1
    prefix = lo[:n]
    while prefix and not prefix[-1].isdigit():
        prefix = prefix[:-1]
    return prefix


def first_disagreement(shown, lo, hi):
    """None if `shown` is exactly the shared prefix, else a disagreement record."""
    prefix = shared_prefix(lo, hi)
    if shown == prefix:
        return None
    for i, ch in enumerate(shown):
        lo_ch = lo[i] if i < len(lo) else None
        hi_ch = hi[i] if i < len(hi) else None
        if lo_ch is None or hi_ch is None or lo_ch != hi_ch or ch != lo_ch:
            return {
                'kind': 'digit', 'index': i, 'abstract': ch, 'lo': lo_ch, 'hi': hi_ch,
                'shown': shown, 'stored': (lo, hi), 'prefix': prefix,
            }
    i = len(shown)
    return {
        'kind': 'early', 'index': i,
        'abstract': None,
        'lo': lo[i] if i < len(lo) else None,
        'hi': hi[i] if i < len(hi) else None,
        'shown': shown, 'stored': (lo, hi), 'prefix': prefix,
    }


def disagreement_text(label, rec):
    lo, hi = rec['stored']
    if rec['kind'] == 'early':
        nxt = rec['prefix'][rec['index']] if rec['index'] < len(rec['prefix']) else None
        return ('%s stops early at index %d: next shared digit %r (endpoints %r and %r); '
                'abstract %s; stored (%s, %s); shared prefix %s'
                % (label, rec['index'], nxt, rec['lo'], rec['hi'], rec['shown'], lo, hi, rec['prefix']))
    return ('%s first disagreement at index %d: abstract %r, endpoints %r and %r; '
            'abstract %s; stored (%s, %s); shared prefix %s'
            % (label, rec['index'], rec['abstract'], rec['lo'], rec['hi'],
               rec['shown'], lo, hi, rec['prefix']))


def bump_last_digit(s):
    """Copy of s with the last digit increased by 1 (carries if that digit is 9)."""
    chars = list(s)
    i = len(chars) - 1
    while i >= 0 and not chars[i].isdigit():
        i -= 1
    if i < 0:
        raise ValueError('no digit in %r' % s)
    while True:
        if chars[i].isdigit() and chars[i] != '9':
            chars[i] = str(int(chars[i]) + 1)
            return ''.join(chars)
        if chars[i].isdigit():
            chars[i] = '0'
        i -= 1
        while i >= 0 and not chars[i].isdigit():
            i -= 1
        if i < 0:
            return '1' + ''.join(chars)


def flatten(text):
    return re.sub(r'\s+', ' ', text)


def load_speed(name):
    path = os.path.join(DATA, name)
    text = open(path, encoding='utf-8').read()
    found = SPEED_LINE.search(text)
    if not found:
        raise SystemExit('%s has no speed interval' % path)
    return found.group(1), found.group(2)


def load_k(name):
    path = os.path.join(DATA, name)
    cfg = json.load(open(path, encoding='utf-8'))
    for a, b in (('K1_dec', 'K2_dec'), ('K1_decimal', 'K2_decimal'), ('k1_dec', 'k2_dec')):
        if a in cfg and b in cfg:
            return str(cfg[a]), str(cfg[b])
    if isinstance(cfg.get('K1'), str) and isinstance(cfg.get('K2'), str):
        return cfg['K1'], cfg['K2']
    raise SystemExit('%s has no decimal K endpoints (keys: %s)' % (path, sorted(cfg)))


def paper_interval(flat, pattern, label):
    found = re.search(pattern, flat)
    if not found:
        return None, '%s speed interval not found in the manuscript' % label
    return (found.group(1), found.group(2)), None


def main():
    paper = open(PAPER, encoding='utf-8').read()
    flat = flatten(paper)
    abs_m = re.search(r'## Abstract\b(.*?)(?:## |\Z)', paper, re.S)
    if not abs_m:
        print('abstract not found')
        print('FAIL')
        return 1
    abstract = flatten(abs_m.group(1))

    lines = []
    ok = True

    # --- check 1: abstract speeds are the shared prefixes of the stored endpoints ---
    speeds = re.search(
        r'the conduction speed begins\s+([0-9]+\.[0-9]+)\s+m/s\s+at 18\.5 C\b'
        r'.*?\s+([0-9]+\.[0-9]+)\s+m/s\s+at 6\.3 C\b',
        abstract)
    stored = {
        '18.5 C': load_speed('pulse_proof_18.5_El10.613_summary.txt'),
        '6.3 C': load_speed('pulse_proof_6.3_El10.613_summary.txt'),
    }
    shown = None
    bits = []
    if not speeds:
        bits.append('abstract speeds under "the conduction speed begins" were not found')
        ok = False
    else:
        shown = {'18.5 C': speeds.group(1), '6.3 C': speeds.group(2)}
        for label in ('18.5 C', '6.3 C'):
            lo, hi = stored[label]
            msg = first_disagreement(shown[label], lo, hi)
            if msg:
                bits.append(disagreement_text(label, msg))
                ok = False
    theorems = (
        ('18.5 C', r'Theorem 1 \(18\.5 C\).*?speed lies in \(([0-9.]+), ([0-9.]+)\) m/s'),
        ('6.3 C', r'Theorem 2 \(6\.3 C\).*?speed lies in \(([0-9.]+), ([0-9.]+)\) m/s'),
    )
    for label, pat in theorems:
        got, err = paper_interval(flat, pat, label)
        lo, hi = stored[label]
        if err:
            bits.append(err)
            ok = False
        elif got != (lo, hi):
            bits.append('%s stored speed (%s, %s) but the paper prints (%s, %s)'
                        % (label, lo, hi, got[0], got[1]))
            ok = False
    if bits:
        lines.append('check 1: ' + ' | '.join(bits))
    else:
        lines.append('check 1: abstract speeds are the shared prefixes '
                     '(18.5 C %s, 6.3 C %s)' % (shown['18.5 C'], shown['6.3 C']))

    # --- check 2: printed K widths equal K2-K1 rounded to the printed sig digits ---
    width_m = re.search(
        r'interval of width ([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+) at 18\.5 C '
        r'and of width ([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+) at 6\.3 C',
        abstract)
    k_diff = {}
    rounded = {}
    printed_w = {}
    wbits = []
    if not width_m:
        wbits.append('abstract K widths were not found')
        ok = False
    else:
        printed_w = {'18.5 C': width_m.group(1), '6.3 C': width_m.group(2)}
        configs = {
            '18.5 C': 'pulse_proof_18.5_El10.613_config.json',
            '6.3 C': 'pulse_proof_6.3_El10.613_config.json',
        }
        for label, cfg_name in configs.items():
            k1, k2 = load_k(cfg_name)
            diff = Dec.parse(k2) - Dec.parse(k1)
            k_diff[label] = diff
            token = printed_w[label]
            canon = canonical_sci(token)
            if canon is None or diff.num <= 0:
                wbits.append('%s paper width %s, exact K2-K1 %s'
                             % (label, token, diff.exact_sci()))
                ok = False
                continue
            sig = sig_digits(token)
            head, exp = diff.round_sig(sig)
            form = format_sig(head, exp)
            rounded[label] = form
            if form != canon:
                wbits.append('%s paper width %s, exact K2-K1 %s, rounded to %d sig %s'
                             % (label, token, diff.exact_sci(), sig, form))
                ok = False
        if not wbits:
            lines.append(
                'check 2: 18.5 C exact K2-K1 %s rounds to %s (%d sig); '
                '6.3 C exact K2-K1 %s rounds to %s (%d sig)'
                % (k_diff['18.5 C'].exact_sci(), rounded['18.5 C'], sig_digits(printed_w['18.5 C']),
                   k_diff['6.3 C'].exact_sci(), rounded['6.3 C'], sig_digits(printed_w['6.3 C'])))
    if wbits:
        lines.append('check 2: ' + ' | '.join(wbits))

    # --- check 3: printed E_l moves the speed by 2.7e-4, to 2 significant digits ---
    claim_m = re.search(
        r'The printed E_l moves the speed by ([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+) m/s \(to 2 digits\)\.',
        flat)
    r_lo, r_hi = load_speed('pulse_proof_18.5_summary.txt')
    t_lo, t_hi = stored['18.5 C']
    raw = (Dec.parse(r_lo) + Dec.parse(r_hi)).div2() - (Dec.parse(t_lo) + Dec.parse(t_hi)).div2()
    raw = abs(raw)
    head, exp = raw.round_sig(2)
    rounded_shift = format_sig(head, exp)
    neighbors = ('2.6e-4', '2.7e-4', '2.8e-4')
    dists = {name: abs(raw - Dec.parse_sci(name)) for name in neighbors}
    closer = dists['2.7e-4'] < dists['2.6e-4'] and dists['2.7e-4'] < dists['2.8e-4']
    remark, remark_err = paper_interval(
        flat,
        r'Remark 1 \(the zero-current leak potential\).*?speed in \(([0-9.]+), ([0-9.]+)\) m/s',
        'Remark 1')
    cbits = []
    if claim_m is None:
        cbits.append('sentence not found; raw %s rounds to %s' % (raw.exact_sci(), rounded_shift))
        ok = False
    else:
        claim = canonical_sci(claim_m.group(1))
        if claim != rounded_shift or sig_digits(claim_m.group(1)) != 2:
            cbits.append('paper %s, raw %s, rounded to 2 sig %s'
                         % (claim_m.group(1), raw.exact_sci(), rounded_shift))
            ok = False
    if remark_err:
        cbits.append(remark_err)
        ok = False
    elif remark != (r_lo, r_hi):
        cbits.append('Remark 1 stored speed (%s, %s) but the paper prints (%s, %s)'
                     % (r_lo, r_hi, remark[0], remark[1]))
        ok = False
    if not closer or rounded_shift != '2.7e-4':
        cbits.append('raw %s rounds to %s; distances to 2.6e-4, 2.7e-4, 2.8e-4 are %s, %s, %s'
                     % (raw.exact_sci(), rounded_shift,
                        dists['2.6e-4'].exact_sci(), dists['2.7e-4'].exact_sci(),
                        dists['2.8e-4'].exact_sci()))
        ok = False
    if cbits:
        lines.append('check 3: ' + ' | '.join(cbits))
    else:
        lines.append(
            'check 3: raw %s rounds to %s; distances to 2.6e-4 / 2.7e-4 / 2.8e-4 are %s / %s / %s '
            '(closer to 2.7e-4 than to either neighbor, so 2.6e-4 and 2.8e-4 are rejected)'
            % (raw.exact_sci(), rounded_shift,
               dists['2.6e-4'].exact_sci(), dists['2.7e-4'].exact_sci(), dists['2.8e-4'].exact_sci()))

    # --- check 4: negative control on a copy of the 18.5 C abstract speed ---
    if not shown or '18.5 C' not in shown:
        lines.append('check 4: no 18.5 C abstract speed to copy')
        ok = False
    else:
        original = shown['18.5 C']
        copied = bump_last_digit(original)
        if copied == original:
            lines.append('check 4: bump did not change the copy')
            ok = False
        else:
            lo, hi = stored['18.5 C']
            msg = first_disagreement(copied, lo, hi)
            if msg is None:
                lines.append('check 4: increasing the last digit of a copy still matched the shared prefix')
                ok = False
            else:
                lines.append('check 4: copy of the 18.5 C abstract speed, last digit %s->%s, fails check 1 at index %d (abstract %r, endpoints %r and %r)'
                             % (original[-1], copied[-1], msg['index'], msg['abstract'], msg['lo'], msg['hi']))

    lines.append('ALL CHECKS PASSED' if ok else 'FAIL')
    print('\n'.join(lines))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
