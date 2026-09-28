#!/bin/sh
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
#
# Reruns every computation of the paper from scratch, one process at a time, each under nice -n 19 and a time limit.
# Usage (from any folder):  sh code/run.sh [tests | 18.5 | 6.3 | zero | all]     (default: all)
#   tests  test_temperature.py (phi from the decimal temperature), test_field.py (the vector field against an
#          independent transcription of the 1952 equations) and test_lohner6.py (the jets and the integrator), each
#          with a negative control (about 4 minutes)
#   18.5   Theorem 1: 18.5 C, printed leak potential E_l = 10.613 mV (about 1 hour)
#   6.3    Theorem 2: 6.3 C, printed leak potential (about 2 hours)
#   zero   Remark 1: 18.5 C, the zero-current leak potential (about 1 hour)
# The temperature is passed as a decimal string, never as a float. Each proof first deletes everything a previous run
# of it wrote (certificates, summary, numerical centre, closing block, reports, checkpoints, resumable states), so no
# old file can stand in for a stage that did not run. Then it recomputes the numerical centre K*, the block, the
# configuration and every stage. A stage that must pass must exit with status 0; a negative control must run to the
# end and exit with status 1 (a written FAIL verdict), and the summary checks that it failed for its stated reason.
# Any other exit status (an error, or the time limit) stops the proof. The summary checks every certificate against
# the current configuration, closing block, programs, phi and E_l (sha256 and balls), and the summary control plants
# stale or wrong certificates in a copy and requires the summary to refuse each. Exit status 0 if and only if all of
# this holds for every proof asked for. Logs go to data/logs/ (not part of the record).
# The starting profiles data/pulse_<T>.npz were made by pulse_bvp.py (numerical; only an initial guess for hp_pulse.py)
# and are inputs here.
set -u
cd "$(dirname "$0")"
LOG=../data/logs
mkdir -p "$LOG"
status=0

step() {   # step <log name> <seconds> <command...>: run bounded, report the exit status and the last line
    name=$1; lim=$2; shift 2
    t0=$(date +%s)
    nice -n 19 timeout "$lim" "$@" > "$LOG/$name.out" 2>&1
    st=$?
    printf '%-28s exit %s  %5ss  %s\n' "$name" "$st" "$(( $(date +%s) - t0 ))" "$(tail -n 1 "$LOG/$name.out" | cut -c1-90)"
    return $st
}

centre() {   # centre <tag> <T> <t_end>: hp_pulse.py resumes after 12 iterations (exit 3), at most 4 times
    for i in 1 2 3 4; do
        step "hp_pulse_$1_$i" 10800 python3 hp_pulse.py "$2" "$3"; st=$?
        [ $st -ne 3 ] && return $st
    done
    return 3
}

proof() {   # proof <T> <t_end> <config arguments...>, with HH_EL exported or unset by the caller (in a subshell)
    T=$1; tend=$2; shift 2
    tag=$(python3 -c "import certify_rest_wave as C; print(C.tag(C.temperature('$T')))") || return 1
    D=../data
    for s in config setup interval K1 K2 neg-shift neg-model; do
        rm -f "$D/pulse_proof_${tag}_$s.json" "$D/ckpt/pulse_${tag}_$s.json" "$D/ckpt/pulse_${tag}_$s.json.tmp"
    done
    rm -f "$D/pulse_proof_${tag}_summary.txt" "$D/pulse_proof_${tag}_summary_control.txt" \
          "$D/hp_pulse_${tag}.json" "$D/hp_pulse_${tag}_tol1.json" "$D/hp_pulse_${tag}.log" \
          "$D/closing_block_${tag}.json" "$D/block0_${tag}.txt" "$D/block_check_iv_${tag}.txt" \
          "$LOG/hp_pulse_${tag}_state_"*.json
    centre "$tag" "$T" "$tend" || return 1
    if [ "$T" = "6.3" ]; then
        # at 6.3 C the error budget written for 18.5 C is too loose; a second run with it 1e-8 times smaller starts
        # from the converged state (manuscript, Section 5)
        step "hp_pulse_${tag}_tight" 10800 python3 hp_pulse.py "$T" "$tend" 1e-8 || return 1
    fi
    step "block0_$tag" 3600 python3 block0.py "$T" || return 1
    step "config_$tag" 600 python3 hh_prove_pulse.py "$T" config "$@" || return 1
    step "setup_$tag" 3600 python3 hh_prove_pulse.py "$T" setup || return 1
    for s in interval K1 K2; do
        step "${s}_$tag" 7200 python3 hh_prove_pulse.py "$T" "$s" || return 1
    done
    for s in neg-shift neg-model; do     # a negative control must end with a written FAIL verdict (status 1)
        step "${s}_$tag" 7200 python3 hh_prove_pulse.py "$T" "$s"; st=$?
        [ $st -eq 1 ] || { echo "$s: exit status $st, not a written FAIL verdict"; return 1; }
    done
    step "block_check_iv_$tag" 3600 python3 hh_block_check_iv.py "$T" || return 1
    step "summary_$tag" 600 python3 hh_prove_pulse.py "$T" summary || return 1
    step "summary-control_$tag" 600 python3 hh_prove_pulse.py "$T" summary-control
}

what=${1:-all}
case "$what" in tests|18.5|6.3|zero|all) ;; *) echo "usage: sh run.sh [tests | 18.5 | 6.3 | zero | all]"; exit 2;; esac
if [ "$what" = tests ] || [ "$what" = all ]; then
    step test_temperature 600 python3 test_temperature.py || status=1
    step test_field 600 python3 test_field.py || status=1
    step test_lohner6 1800 python3 test_lohner6.py || status=1
    cp "$LOG/test_lohner6.out" ../data/test_lohner6.txt       # the other tests write their reports themselves
fi
if [ "$what" = 18.5 ] || [ "$what" = all ]; then
    (export HH_EL=10.613; proof 18.5 10.5 1.5e-45 1e-25 13.625 1e-16 1e-70) || status=1
fi
if [ "$what" = 6.3 ] || [ "$what" = all ]; then
    (export HH_EL=10.613; proof 6.3 23.0 1.4e-61 1e-32 36.125 1e-35 1e-70) || status=1
fi
if [ "$what" = zero ] || [ "$what" = all ]; then
    (unset HH_EL; proof 18.5 10.5 1.5e-45 1e-25 13.625 1e-16 1e-70) || status=1
fi
[ $status -eq 0 ] && echo "run.sh $what: ALL AS EXPECTED" || echo "run.sh $what: SOMETHING FAILED"
exit $status
