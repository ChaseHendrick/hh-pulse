#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Numerical pulse illustrations from the committed printed-leak shooting nodes.

Run from any directory: python3 code/make_figure.py
No proof is rerun. Each short DOP853 segment starts at a stored high-precision
node, preventing unstable forward-error accumulation across the whole pulse.
The reconstructed curves are numerical illustrations, not interval enclosures.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hhwave

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper' / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .6, 'legend.frameon': False})


def reconstruct(d, tolerance, leak_potential=10.613):
    nodes = np.asarray(d['nodes_profile_time'], float)
    starts = np.asarray([d['p0']] + d['Y'], float)
    assert starts.shape == (len(nodes) - 1, 5)
    assert np.all(np.diff(nodes) > 0)
    model = hhwave.Wave(d['T'], EL=leak_potential)  # Never use the zero-current default.
    times, curves, joins = [], [], []
    for i, start in enumerate(starts):
        t = np.linspace(nodes[i], nodes[i + 1], 401)
        sol = solve_ivp(model.f, (t[0], t[-1]), start, args=(float(d['K']),),
                        method='DOP853', t_eval=t, rtol=tolerance, atol=tolerance * .01)
        assert sol.success and np.isfinite(sol.y).all()
        if i + 1 < len(starts):
            joins.append(np.abs(sol.y[:, -1] - starts[i + 1]))
        times.extend(t[:-1] if i + 1 < len(starts) else t)
        curves.extend((sol.y[:, :-1] if i + 1 < len(starts) else sol.y).T)
    return np.asarray(times), np.asarray(curves), np.max(joins, axis=0), model


fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.5), constrained_layout=True)
checks = []
for row, temperature in enumerate(('18.5', '6.3')):
    path = ROOT / 'data' / ('hp_pulse_' + temperature + '_El10.613.json')
    d = json.loads(path.read_text())
    t, y, joins, model = reconstruct(d, 1e-11)
    tr, refined, joins_refined, _ = reconstruct(d, 1e-12)
    assert np.array_equal(t, tr)
    delta = np.max(np.abs(y - refined), axis=0)
    # Display-quality gates only, far weaker than the proof's certified bounds.
    limits = np.array([1e-6, 1e-5, 1e-8, 1e-8, 1e-8])
    assert np.all(delta < limits) and np.all(joins_refined < limits)
    assert np.min(refined[:, 2:]) >= 0 and np.max(refined[:, 2:]) <= 1
    # A historical zero-current leak must not pass as the printed-leak model.
    _, _, wrong_joins, _ = reconstruct(d, 1e-12, hhwave.el_zero())
    assert np.any(wrong_joins > limits), 'Wrong-leak failure control was not rejected'
    # The phase origin is arbitrary; align the resolved voltage maxima for readability.
    peak = int(np.argmax(refined[:, 0]))
    phase = t - t[peak]
    a, b = axes[row]
    a.plot(phase, refined[:, 0], color='#2368a2', lw=1.5)
    a.axhline(model.urest, color='#555555', lw=.65, ls=':')
    a.set_ylabel('depolarization $u$ (mV)')
    a.set_title('(%s) %s °C: pulse voltage' % ('a' if row == 0 else 'c', temperature), loc='left')
    for index, name, color, style in ((2, '$m$', '#2368a2', '-'),
                                      (3, '$n$', '#a54d00', '--'),
                                      (4, '$h$', '#39734b', '-.')):
        b.plot(phase, refined[:, index], color=color, ls=style, lw=1.35, label=name)
    b.set_ylabel('gate fraction')
    b.set_ylim(-.025, 1.025)
    b.set_title('(%s) %s °C: gating variables' % ('b' if row == 0 else 'd', temperature), loc='left')
    b.legend(loc='upper right', ncol=3, handlelength=1.5, columnspacing=.8)
    for ax in (a, b):
        ax.set_xlabel(r'profile coordinate $\xi-\xi_{\rm peak}$ (ms)')
        ax.set_xlim(phase[0], phase[-1])
        ax.grid(color='#dededb', lw=.45)
        ax.set_axisbelow(True)
    checks.append({'temperatureC': d['T'], 'source': str(path.relative_to(ROOT)),
                   'sourceSha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'leakPotentialMv': model.EL, 'KPerMs': d['K'], 'segments': len(d['Y']) + 1,
                   'samples': len(t), 'stateOrder': ['u', 'du/dxi', 'm', 'n', 'h'],
                   'maximumSegmentEndpointDisagreement': joins_refined.tolist(),
                   'maximumToleranceRefinementDifference': delta.tolist(),
                   'displayCheckLimits': limits.tolist(),
                   'wrongLeakControlEndpointDisagreement': wrong_joins.tolist(),
                   'wrongLeakControlRejected': True, 'sampledMaximumMvNumerical': float(refined[peak, 0]),
                   'minimumMvNumerical': float(np.min(refined[:, 0])),
                   'recordedCoordinateExtentMs': [float(t[0]), float(t[-1])]})
fig.savefig(FIG / 'pulse-profiles.pdf', metadata={'CreationDate': None,
            'Title': 'Numerical Hodgkin-Huxley pulse profiles at the printed leak potential',
            'Author': 'Chase Hendrick'})
plt.close(fig)
result = {'scope': 'Numerical illustration only, not proof evidence or a global error enclosure.',
          'method': 'DOP853 short segments restarted at committed high-precision shooting nodes; no extrapolated tails.',
          'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                       'matplotlib': matplotlib.__version__}, 'checks': checks}
(ROOT / 'data' / 'figure-profile-checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
