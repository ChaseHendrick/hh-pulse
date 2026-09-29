---
title: "The Propagated Action Potential of Hodgkin and Huxley at Their 1952 Constants: A Computer-Assisted Existence Proof"
author: "Chase Hendrick, Independent Researcher (ORCID 0009-0002-9754-6087)"
status: "Release 1.0.0, 2026-09-28."
---

## Abstract

In 1952 Hodgkin and Huxley computed their propagated action potential by hand. They shot in the conduction speed on their travelling-wave equation (J. Physiol. 117, eq. (31)), and noted that the solution "goes off towards either +infinity or -infinity" on the two sides of the speed.

The existence proofs that followed do not treat that equation. Hastings (1976) and Carpenter (1977) slow or speed the gating variables by a small parameter.

We prove that the unmodified equation has a pulse, an orbit homoclinic to rest, at 18.5 C and at 6.3 C. The rate functions and constants are those Hodgkin and Huxley printed, including the leak potential 10.613 mV. The proof is computer-assisted, in ball arithmetic. The speed parameter K lies in an interval of width 3e-45 at 18.5 C and of width 2.8e-61 at 6.3 C.

For the fibre constants of their p. 528, the conduction speed begins

    18.73188824788048354046831343329624387695575077 m/s

at 18.5 C, against the 18.8 m/s they computed, and

    12.31375672016229859330140853674732362368394751453213155988245 m/s

at 6.3 C. Every digit shown is proved.

The orbit leaves rest along its one-dimensional unstable manifold. A validated Taylor integrator carries a whole interval of speeds through the spike. An isolating block with a cone condition around rest, and a Wazewski-type shooting argument, closes it. [Computer-assisted.]

## 1. Introduction

**The question.** A travelling wave V(x, t) = V(t - x/theta) of the Hodgkin-Huxley cable equation solves eq. (31) of
Hodgkin and Huxley (1952, p. 524),

    d^2V/dt^2 = K { dV/dt + (1/C_M) [ g_K n^4 (V - V_K) + g_Na m^3 h (V - V_Na) + g_l (V - V_l) ] },
    K = 2 R_2 theta^2 C_M / a,

together with their equations for m, n, h. A propagated action potential is a solution that starts and ends at rest.
Hodgkin and Huxley found it numerically, at K = 10.47 /ms and 18.5 C (their p. 528). **Is there a proof that it
exists, at their own constants?**

**What was known** (details in Section 6). Hastings (1976) proved existence for a class of Hodgkin-Huxley-type systems in
which n and h are slowed by a factor epsilon, "for epsilon sufficiently small", and wrote that "it is not clear that
our results apply to the original HODGKIN-HUXLEY system" (p. 230). Carpenter (1977) proved existence for a generalized
Hodgkin-Huxley system under abstract hypotheses, with n and h slowed by epsilon and m sped up by 1/delta, "for small
epsilon" (Theorem 3.4) and "all small delta" (Theorem 4.2), and showed that the pulse is lost when either parameter is
too large (Theorem 5.1(B)). Neither treats epsilon = delta = 1 at the 1952 functions. Computer-assisted proofs of
travelling pulses exist for the FitzHugh-Nagumo equation (Arioli and Koch 2015, and others), not, as far as we found,
for Hodgkin-Huxley.

**What we prove.** Theorems 1 and 2 (Section 3): at 18.5 C and at 6.3 C, with the 1952 rate functions and constants
and the printed leak potential, the travelling-wave system has a pulse, with the speed parameter K in an explicit
interval of width 3e-45 at 18.5 C and 2.8e-61 at 6.3 C. [Computer-assisted.] The same holds at 18.5 C with the leak
potential that makes the resting current exactly zero (Remark 1). [Computer-assisted.]

**Earlier work, as far as the search reached.** The searches summarized in Appendix C (2026-09-25 to 2026-09-27)
are recorded there. They are not a claim of priority. Two papers could not be obtained in full: Hastings (1976), of
which we read pp. 229-230 of its 29 pages, and Foote and Chen, "Traveling wave properties of the Hodgkin-Huxley
equations", Chinese J. Math. 9 (1981) 1-23, which we could not read at all. zbMATH Open has no review of either
(Zbl 0374.35004, Zbl 0472.35048), and MathSciNet was not reachable. Carpenter (1977) was read in full.

**What the proof rests on.** Besides the computations, the proof uses only Hodgkin and Huxley's equations and constants
(1952, pp. 519-528 and Table 3, read in a scan of the paper) and standard facts about ordinary differential equations,
cited from one textbook that was read for the purpose (Teschl 2012: extension of solutions, Corollary 2.15; the flow is
continuous on its open domain, Theorem 6.1; orbits in a compact set are complete and have nonempty, compact, invariant
limit sets, Lemmas 6.3, 6.5 and 6.6; the local unstable manifold of a hyperbolic equilibrium, Theorems 9.4 and 9.5).
Every other step is proved here (Section 4 and Appendices A and B). No step of the proof depends on Hastings (1976),
Carpenter (1977), Foote and Chen (1981) or Arioli and Koch (2015); they bear only on the paragraph on earlier work and
on the comparison of methods. The computations rest on the correctness of Arb (Johansson 2017), in FLINT, through
python-flint 0.9.0, and, for the independent re-check of the closing block, of mpmath 1.3.0.

**Methods and their sources.** The integrator is a Lohner-type method (Lohner 1988): Taylor enclosures of the flow
with the wrapping effect controlled by moving coordinates. The closing argument is a topological shooting argument in
the tradition of Wazewski (1947) and of Conley's isolating blocks (Conley 1975, 1978), the approach Carpenter (1977)
used for nerve impulse equations with small parameters. The blocks are checked through cone conditions, a
quadratic form that increases along the flow, as in Zgliczynski (2009). Every lemma the proof uses is proved here from
these ideas, so the citations credit the methods and are not premises.

## 2. The equations

We use the modern sign convention u = -V (depolarization, mV), t in ms, C_M = 1 uF/cm^2. With u = -V, the reversal
potentials rewritten in the new sign (V_Na = -115, V_K = 12 and V_l = -10.613 mV become 115, -12 and E_l = 10.613 mV)
and the rate functions written in u, eq. (31) keeps its form:

    u'' = K (u' + I(u, m, n, h)),   I = 120 m^3 h (u - 115) + 36 n^4 (u + 12) + 0.3 (u - E_l),
    x'  = phi (alpha_x(u) (1 - x) - beta_x(u) x),   x = m, n, h,   phi = 3^((T - 6.3)/10),

with alpha_m = Psi((25 - u)/10), beta_m = 4 e^(-u/18), alpha_n = Psi((10 - u)/10)/10, beta_n = e^(-u/80)/8,
alpha_h = 0.07 e^(-u/20), beta_h = 1/(e^((30 - u)/10) + 1), Psi(x) = x/(e^x - 1) (Psi(0) = 1), and
**E_l = 10.613 mV**, Hodgkin and Huxley's V_l = -10.613 mV (Table 3) in our convention. The state is
y = (u, u', m, n, h) in R^5, and we write the system as y' = f(y, K); f is real analytic in (y, K). The speed is
theta = sqrt(K a / (2 R_2 C_M)), with a = 238 um and R_2 = 35.4 ohm cm for the fibre of their p. 528.

With the printed E_l the resting current is not exactly zero at u = 0 (Table 3's footnote says the value was chosen to
make it zero; with the printed rate functions the exact value is 10.5989...). Rest is then the equilibrium
y* = (u*, 0, m_inf(u*), n_inf(u*), h_inf(u*)) with

    u* = 0.0036206688079425688368876905420... mV     [computer-assisted: Lemma B.1; the only zero of the
                                                       resting current within 1e-3 mV of this value]

and it depends neither on T nor on K. A pulse is a non-constant solution with y(t) -> y* as t -> +-infinity.

**The programs integrate these equations.** `test_field.py` transcribes eq. (31) and Table 3 in the 1952 sign
convention, with V = -u, V_Na = -115, V_K = 12 and V_l = -E_l, and compares that transcription with `hhjet6.vfield`.
At 64 states, at both temperatures and both leak potentials, including the two values where Psi is singular, the
largest relative difference is 5.22e-60. Changing V_K to -12 in the transcription alone makes the two disagree, so
the comparison is not vacuous. The neural-field reduction is a different paper and is not checked here.

## 3. Results

**Theorem 1 (18.5 C).** [Computer-assisted.] Let T = 18.5 C. For some K* in (K1, K2), with K1 and K2 the exact binary
fractions of `data/pulse_proof_18.5_El10.613_config.json`,

    K1 = 10.4380510601011236922764862383185791218586697704783...,   K2 - K1 = 3.000e-45 (to 4 digits),

the system of Section 2 has a pulse, and max u > 90.57 mV along it. The corresponding speed lies in

    (18.73188824788048354046831343329624387695575077276, 18.73188824788048354046831343329624387695575077548) m/s.

**Theorem 2 (6.3 C).** [Computer-assisted.] Let T = 6.3 C. For some K* in (K1, K2), with K1 and K2 the exact binary
fractions of `data/pulse_proof_6.3_El10.613_config.json`,

    K1 = 4.51063243827085108340326514573481715152474867970496...,   K2 - K1 = 2.800e-61 (to 4 digits),

the system of Section 2 has a pulse, and max u > 102.98 mV along it. The corresponding speed lies in

    (12.313756720162298593301408536747323623683947514532131559882457804,
     12.313756720162298593301408536747323623683947514532131559882458189) m/s.

In both theorems the pulse leaves rest along a branch of the one-dimensional unstable manifold of y*, the branch
through the exit set E of Lemma 1, on which u > u* where it leaves the box B (the programs check u - u* > 0 on all of
E), and returns to y* inside the block B0 of Section 4.

**Remark 1 (the zero-current leak potential).** [Computer-assisted.] With E_l = 10.5989209693916785... (the value
that makes the resting current zero at u = 0, so that u* = 0) Theorem 1 holds with K2 - K1 = 3.000e-45 (to 4 digits) and the speed in
(18.73216081438890211377538515402816936801773373586, 18.73216081438890211377538515402816936801773373858) m/s.
The printed E_l moves the speed by 2.7e-4 m/s (to 2 digits).

**Remark 2 (numerical, not proved).** High-precision multiple shooting, at the printed leak potential, gives

    K*(18.5 C) = 10.43805106010112369227648623831857912185866977197832623694577177829239

    K*(6.3 C) = 4.510632438270851083403265145734817151524748679704964534920325310959260

each inside the interval [K1, K2] of its theorem. Hodgkin and Huxley's K = 10.47 /ms is 0.3 per
cent higher; with the same fibre constants it gives 18.76 m/s, which they report as 18.8 m/s. The measured speed in
that fibre was 21.2 m/s. As rounding intervals of those printed values, [10.465, 10.475] /ms, [18.75, 18.85] m/s and
[21.15, 21.25] m/s are each disjoint from the interval of Theorem 1.

**Not claimed.** Uniqueness of the pulse or of K*; stability; other temperatures; the slow pulse that Huxley (1959)
and later authors found numerically.

## 4. The proof

The proof has computed hypotheses (H2) to (H5) and an argument that uses only them and the facts from Teschl (2012)
listed in Section 1. All computations are in ball arithmetic (FLINT/Arb through python-flint 0.9.0) at 256 bits (128
bits for enclosures that are only widened); each program stops on a failed check. Appendix A proves that the
integrator's enclosures are enclosures, and Appendix B the interval lemmas the checks use.

For an invertible matrix P we use the coordinates z = P (y - y*). Since f(y*, K) = 0, for y in a convex set Q that
contains y*,

    z' = A-bar z,   A-bar = integral_0^1 P Df(y* + s (y - y*), K) P^-1 ds,                      (4.1)

an average of the matrices P Df(x, K) P^-1 over x in Q. We write sym S = (S + S^T)/2.

**Lemma 0 (a cone condition gives the inertia).** Let A be a real n x n matrix and D = diag(1, -1, ..., -1). If
D A + A^T D is positive definite, then A has one simple real eigenvalue lambda_u > 0, with an eigenvector v such that
v^T D v > 0, and n - 1 eigenvalues with negative real part.

*Proof.* If A v = lambda v with v complex and nonzero, then v^* (D A + A^T D) v = 2 Re(lambda) v^* D v > 0, so no
eigenvalue is on the imaginary axis. Let E+ and E- be the real invariant subspaces of A belonging to the eigenvalues
with positive and with negative real part; R^n is their direct sum. For x in E- \ {0}, x(t) = e^(tA) x tends to 0 as
t -> +infinity and d/dt (x(t)^T D x(t)) = x(t)^T (D A + A^T D) x(t) > 0, so x^T D x < 0; likewise x^T D x > 0 on
E+ \ {0}, with t -> -infinity. Every 2-dimensional subspace contains a nonzero x with x_1 = 0, where x^T D x <= 0, so
dim E+ <= 1; and E- does not contain e_1, so dim E- <= n - 1. Hence dim E+ = 1: E+ is spanned by a real eigenvector v
of a real eigenvalue lambda_u > 0 of algebraic multiplicity one, and v^T D v > 0. QED

**(H2) Where the unstable manifold leaves a small box** (`certify_rest_wave.lemma_B`). In z = T_B (y - y*), with T_B
an exact binary matrix that approximately diagonalizes Df(y*) (unstable, fast real, complex pair, slow real), let

    B = {|z1| <= r_B, |z2| <= s2, z3^2 + z4^2 <= s3^2, |z5| <= s5},

with r_B = 1e-25 (18.5 C) or 1e-32 (6.3 C) and s_j of order r_B^2 (the values are in the configuration files).
Checked in ball arithmetic over B x [K1, K2]: (i) D A + A^T D is positive definite for every A in an interval matrix
enclosing T_B Df(y, K) T_B^-1, D = diag(1, -1, -1, -1, -1) (Gershgorin's bound, Lemma B.5); (ii) the stable faces
are strictly inflowing:
z2 z2' < 0 where |z2| = s2, z3 z3' + z4 z4' < 0 where z3^2 + z4^2 = s3^2, and z5 z5' < 0 where |z5| = s5, at every
point of B; (iii) z1' > 0 on the exit face E = B intersected with {z1 = r_B}.

Write L = z1^2 - (z2^2 + ... + z5^2). By (i), (4.1) with P = T_B and Q = B, and the concavity in A of the smallest
eigenvalue of D A + A^T D, dL/dt = z^T (D A-bar + A-bar^T D) z > 0 at every point of B other than y*.

**(H1) Rest is hyperbolic, with a one-dimensional unstable manifold.** By Lemma 0 applied to T_B Df(y*, K) T_B^-1 (y*
lies in B), for every K in [K1, K2]. As an independent check, `certify_rest_wave.lemma_A` encloses the characteristic
polynomial of Df(y*, K) over the K interval, shows that it has one simple real root lambda_u in an interval, and checks
the Routh-Hurwitz inequalities for the quotient (Lemma B.4); the proof does not use this check.

**Lemma 1.** Assume (H2). For every K in [K1, K2]:

(a) the unstable manifold of y* has a branch, the orbit Gamma_K of a solution x_K with x_K(t) -> y* as t -> -infinity
and z1 > 0 on it near y*, that stays in B until a first time t_e, at which it leaves B through a point p(K) of E;

(b) p(K) is the only point of E whose backward orbit stays in B, and K -> p(K) is continuous on [K1, K2].

*Proof.* (a) By Lemma 0, y* is hyperbolic with a one-dimensional unstable space spanned by v, with v^T D v > 0 in z
coordinates, so v_1 is not 0. By Teschl's Theorems 9.4 and 9.5 the points near y* whose backward orbits stay near y*
form a C^1 curve through y*, tangent to v, that consists of y* and two orbits; near y*, z1 has opposite signs on them,
and L > 0 on them (tangency and v^T D v > 0). Let Gamma_K be the orbit with z1 > 0 and x_K a solution on it. The point
z = 0 is interior to B, so x_K(t) lies in B for all t below some t_0. While x_K is in B, L increases, so L > 0 and z1
cannot vanish: z1 > 0. Let t_e = sup {t : x_K(s) in B for all s <= t}. If t_e were infinite, the forward orbit would
lie in the compact set B; by Teschl's Lemmas 6.3, 6.5 and 6.6 its omega-limit set would be nonempty, compact,
invariant and inside B, and L, increasing and bounded along the orbit, would be constant on it; since dL/dt > 0 on
B \ {y*}, the omega-limit set would be {y*}, and L(x_K(t)) would increase to L(y*) = 0 from positive values, which is
impossible. So t_e is finite, x_K(t_e) is on the boundary of B, and there are times t > t_e arbitrarily close to t_e
at which x_K(t) is outside B. One of the constraints g_j <= 0 that define B (g = z2^2 - s2^2, and so on) is violated
at such times while g_j(x_K(t_e)) = 0, so d g_j(x_K(t))/dt >= 0 at t_e. By (ii) this is impossible for the stable
faces, so x_K(t_e) is on a face |z1| = r_B, and since z1 > 0, p(K) = x_K(t_e) is in E.

(b) Let q be in E, with its backward orbit in B. By Teschl's Lemma 6.3, q is backward complete; by Lemmas 6.5 and 6.6
its alpha-limit set is nonempty, compact, invariant and in B, and L is constant on it, so as in (a) it is {y*}. So the
backward orbit tends to y*: otherwise there are times t_n -> -infinity at which it stays at distance at least some
epsilon > 0 from y*, and since B is compact a subsequence converges to a point of the alpha-limit set other than y*.
Then L decreases to 0 along it in backward time, hence L > 0 and z1 > 0 on it, and by Teschl's
Theorem 9.5 it eventually lies on the local unstable manifold, on the branch with z1 > 0: q is on Gamma_K. A point of
Gamma_K after p(K) has a backward orbit through points just after p(K), where z1 > r_B by (iii), so outside B. A point
of Gamma_K in E before p(K) would, by (iii), be followed at once by points with z1 > r_B, outside B, before t_e,
contradicting the definition of t_e. So q = p(K).

Continuity: let K_n -> K in [K1, K2]. Since E is compact it suffices to show that every limit q of a subsequence of
p(K_n) is p(K). Consider the system with K as a sixth variable, K' = 0, and its flow Phi, continuous on its open
domain (Teschl, Theorem 6.1). For every t <= 0 in the maximal interval of the solution through (q, K), the points
Phi(t, (p(K_n), K_n)) are defined for large n (the domain is open) and lie in B, so their limit Phi(t, (q, K)) lies in
B (B is closed). By Teschl's Lemma 6.3 the solution through q is then defined for all t <= 0, and its backward orbit
lies in B. By (b), q = p(K). QED

**(H3) The closing block** (`block0.py`). In zeta = M (y - y*), M = diag(10, 7, 1, 1, 40) T (T an exact binary
approximate inverse eigenbasis), let B0 = {|zeta_1| <= r, |zeta_s|_2 <= rho}, zeta_s = (zeta_2, ..., zeta_5), with
rho = 0.8, r = 0.84 (18.5 C) and rho = 0.6, r = 0.63 (6.3 C), as binary floating-point numbers that the programs use
exactly: rho = 0.8000000000000000444..., r = 0.8 x 1.05 rounded to 0.8400000000000000799..., and rho =
0.5999999999999999777..., r = 0.6300000000000000044.... Checked on a cover of B0 by cells (Lemma B.3), with
interval Cholesky factorizations (Lemma B.2), for every K in the interval: (C) D A + A^T D is positive definite for
every A in an interval matrix enclosing M Df(x) M^-1 over the cell; (E) on the cells that meet B0 and
{|zeta_1| <= rho}, -sym(A_ss) - mu I is positive definite, where mu is an upper bound of the Euclidean norm of the
column A_s1 over the cell; so lambda_max(sym A_ss) + |A_s1|_2 < 0 there.

**Lemma 2.** Assume (H3). While an orbit is in B0 and not at y*, L = zeta_1^2 - |zeta_s|^2 increases strictly; every
boundary point of B0 with L <= 0 is a point of strict entrance; the cones K+ = {L > 0, zeta_1 > 0} and
K- = {L > 0, zeta_1 < 0} cannot be left while the orbit stays in B0; and an orbit that stays in B0 for all t >= t_0
tends to y*.

*Proof.* Apply (4.1) with P = M and Q = B0, or Q = B0 intersected with {|zeta_1| <= rho}; both are convex and contain
y*. The smallest eigenvalue of D A + A^T D is a concave function of A and lambda_max(sym A_ss) + |A_s1|_2 is a convex
one, so the bounds (C) and (E), which hold for every M Df(x) M^-1 with x in the region, hold for the averages A-bar.
Then dL/dt = zeta^T (D A-bar + A-bar^T D) zeta > 0 for zeta != 0. At a boundary point with L <= 0 we have
|zeta_s| = rho and |zeta_1| <= rho (because r > rho), and (1/2) d|zeta_s|^2/dt = zeta_s^T (A-bar_ss zeta_s +
A-bar_s1 zeta_1) <= (lambda_max(sym A-bar_ss) + |A-bar_s1|_2) rho^2 < 0, so the orbit is outside B0 just before and
inside just after. While the orbit is in B0, L > 0 persists and zeta_1 cannot vanish, so it keeps its sign. An orbit
that stays in B0 for t >= t_0 has, by Teschl's Lemmas 6.3, 6.5 and 6.6, a nonempty invariant omega-limit set in B0 on
which L is constant, hence {y*}; and the orbit tends to y*, since otherwise times t_n -> infinity at which it stays
at distance at least some epsilon > 0 from y* would have, B0 being compact, a subsequence converging to a point of
the omega-limit set other than y*. The face |zeta_1| = r is not used. QED

**(H4) The interval run** (`hh_prove_pulse.py interval`). A Lohner-type integrator in the six variables (y, K), K' = 0
(`lohner6.py`; Taylor jets of order 40 with derivatives in the initial point, `hhjet6.py`), carries a set containing
E x [K1, K2] from t = 0 to t = T_enter and encloses it, at T_enter, in the interior of B0. By Appendix A, for every
(q, K) in E x [K1, K2] the solution through (q, K) exists on [0, T_enter] and its value at T_enter lies in int B0.

**(H5) The endpoint runs** (`hh_prove_pulse.py K1`, `K2`). For K = K1 (resp. K2) the set E is carried to T_enter, lies
in int B0 there, and is carried further in steps of 2^-7 ms, with the whole path of every step enclosed (Lemma A.4)
and inside int B0, until the set lies in K- (resp. K+).

**Proof of Theorems 1 and 2 from (H2) to (H5).** For K in [K1, K2] let x_K be the solution with x_K(0) = p(K). It
lies on the unstable manifold, so x_K(t) -> y* as t -> -infinity. By Lemma 1(b) and the continuity of the flow of the
six-variable system on its open domain (Teschl, Theorem 6.1), K -> x_K(t) is continuous, uniformly for t in compact
intervals on which the solutions exist. Let S+ (resp. S-) be the set of K in [K1, K2] for which there is t >= T_enter
with x_K([T_enter, t]) in int B0 and x_K(t) in K+ (resp. K-). Both are open in [K1, K2]: the conditions are open and
concern a compact time interval. They are disjoint (Lemma 2: a cone cannot be left while in B0, and an orbit in K+
never meets K-). K2 is in S+ and K1 in S- (H5, since p(K1) and p(K2) are in E). Since [K1, K2] is connected, some K*
is in neither. By (H4), x_{K*}(T_enter) is in int B0. If x_{K*} left B0, then at the first time t_e at which it
reaches the boundary, either L <= 0, and then by Lemma 2 the orbit enters B0 strictly there, so it was outside B0 just before
t_e, which it was not; or L > 0, and then it was in K+ or K- just before t_e while in int B0, so K* would be in S+ or
S-. Hence x_{K*}(t) stays in B0 for t >= T_enter (and exists for all such t, by Teschl's Lemma 6.3) and tends to y*
(Lemma 2). It is not constant. So it is a pulse. The speed bounds follow from theta = sqrt(K a / (2 R_2 C_M)) in ball
arithmetic, with the decimal ends rounded outward. The lower bound on max u is the lower end of an enclosure of u at a
step end of the interval run. QED

**What is not part of the proof.** The numerical centre K* (Remark 2), used only to place [K1, K2]; the choice of the
weights, the radii, the matrices T_B, T and M, and T_enter; floating-point step-size heuristics (they choose step
lengths; every enclosure is checked); the characteristic-polynomial check of (H1).

## 5. Computations, controls and checks

Table: stages at 18.5 C (printed E_l), one process at a time, 256 bits, order 40. The times are the wall-clock seconds
the certificates record (`secs`), not processor time.

| stage | result | wall time |
|---|---|---|
| setup (H2, H3, the check of H1) | lambda_u = 10.89208...; Lemma B at r_B = 1e-25; z1' > 0 and u > u* on the exit set; B0 certified on 1232 + 5916 cells | seconds |
| interval (H4) | at T_enter = 13.625 ms: zeta_1 in [-0.341, 0.341], abs(zeta_s) <= 0.63603 < 0.8 | 376 s |
| K1 (H5) | enters K- at 13.6953125 ms, path in int B0 | 376 s |
| K2 (H5) | enters K+ at 13.6875 ms, path in int B0 | 375 s |
| negative control: K interval shifted by 40 half-widths | zeta_1 about 13 at T_enter, outside B0: fails, as it must | 373 s |
| negative control: alpha_m times (1 + 1e-12 (u - u*)^2) | the whole set escapes below u = -60 mV at 6.72 ms: fails, as it must | 390 s |
| negative controls in setup | a bracket above lambda_u; Lemma B faces 100 times thinner; B0 with radius x 1.5: all rejected | seconds |

Table: the same at 6.3 C (printed E_l).

| stage | result | wall time |
|---|---|---|
| setup (H2, H3, the check of H1) | lambda_u = 4.974030...; Lemma B at r_B = 1e-32; z1' > 0 and u > u* on the exit set; B0 certified on 3590 + 1374 cells | seconds |
| interval (H4) | at T_enter = 36.125 ms: zeta_1 in [-0.273, 0.273], abs(zeta_s) <= 0.45631 < 0.6 | 852 s |
| K1 (H5) | enters K- at 36.2578125 ms, path in int B0 | 849 s |
| K2 (H5) | enters K+ at 36.234375 ms, path in int B0 | 852 s |
| negative control: K interval shifted by 40 half-widths | zeta_1 about 10 at T_enter, outside B0: fails, as it must | 856 s |
| negative control: alpha_m times (1 + 1e-12 (u - u*)^2) | the whole set escapes below u = -60 mV at 17.20 ms: fails, as it must | 860 s |
| negative controls in setup | as at 18.5 C: all rejected | seconds |

The model control multiplies alpha_m by 1 + 1e-12 (u - u*)^2, with u* the rest-voltage ball rather than its midpoint, so the factor is 1 only at a single point of that ball. It re-runs the cone and inflow test. It does not compare linearizations, and it does not re-check z1' > 0. (H3) is
not checked again; on B0, where |u - u*| is of order 1 mV, the factor changes the field by about 1e-12. The control only
has to fail, and the perturbation moves the pulse speed by far more than the width of the K interval. At 6.3 C the numerical centre has to be computed with a local error budget 1e-8 times tighter than at
18.5 C: the budget is written for the growth rate at 18.5 C, and the looser centre differs from the tighter one by about 5e-59, far more than the width of the K interval. The proof uses the tighter centre.

**Independent re-check of (H3).** `hh_block_check_iv.py` is a separate program: mpmath interval arithmetic at 113 bits,
the Jacobian from hand-derived formulas, M^-1 in exact rational arithmetic, its own cover and Cholesky test, phi from
the decimal temperature, and its own enclosure of the rest state by bisection, which it checks to lie within the
window where Lemma B.1 shows that the zero is unique. It confirms (C) and (E) at both temperatures and rejects the
enlarged block with the same depth limit as the check itself.

**Tests.** `test_field.py` compares the vector field of the programs with an independent transcription of the 1952
equations in Hodgkin and Huxley's own sign convention, at 64 states at both temperatures and both leak potentials,
including the removable singularities at u = 10 and u = 25 and states next to them; the largest relative difference
is 5.2e-60, and the same transcription with V_K = -12 is caught. `test_lohner6.py`: the six-variable jets agree with
an independent Picard implementation and with central differences in K; Lohner enclosures through the upstroke, at
orders 30 and 8, contain a high-precision reference solution computed with the same jets and a different step
sequence (so this tests the enclosures, not the field); at order 8 with the remainder term dropped the enclosure
misses it, as it must. `test_temperature.py` checks that phi is computed from the decimal temperature, so that phi = 1
at 6.3 C; the binary number nearest 6.3, 6.29999999999999982 C, gives phi = 1 - 1.95e-17, and the test rejects it.

**The harness.** Every certificate records the sha256 of the configuration, of the closing block and of the eight
programs the proof runs, the python-flint version, and the phi and E_l balls. The summary recomputes all of these, and
each verdict from the certificate's own fields, and refuses a certificate that does not match, whose K is not the
configuration's, or whose negative control failed for a reason other than its stated one. A summary control plants
thirteen kinds of stale, foreign or self-contradictory certificate in a copy of the data and requires the summary to
refuse each, and to accept the unaltered copy.

**Consistency.** At T_enter = 13.625 ms (18.5 C, printed E_l) the certificates enclose zeta_1(K1) in [-0.3032 +/- 1.03e-5] and zeta_1(K2) in [0.33917 +/- 8.72e-6].

## 6. Comparison with Carpenter (1977) and Hastings (1976)

Carpenter's Theorem 3.4 (p. 353) proves, under the abstract Hypotheses (3.1, CUBIC, H) and (3.3, HOM, H), that the
reduced system (3.1, H), with m = m_inf(V) and n, h multiplied by epsilon, has a homoclinic solution "for small
epsilon > 0"; Theorem 4.2 (p. 357) restores m as a fast variable "for all small delta > 0"; and Theorem 5.1(B)
(pp. 357-358) shows that the pulse is lost when epsilon or delta is too large. Her method, isolating blocks and a
Wazewski-type shooting in the speed around a singular orbit, is the same kind of topological argument as ours, applied
where the small parameters make the orbit computable by hand; ours applies it to a validated numerical orbit at
epsilon = delta = 1. zbMATH lists her lecture notes "Nerve impulse equations" (Carpenter 1976) next to the 1977 paper;
we have not read them. Hastings's theorem (pp. 229-230, read) needs n and h slowed by a small epsilon and hypotheses he
did not verify for the 1952 functions.

## 7. Reproducibility

The companion repository is [ChaseHendrick/hh-pulse](https://github.com/ChaseHendrick/hh-pulse).
Checking release 1.0.1 archives the exact existence-proof programs, certificates and supplementary checking
scripts at [doi:10.5281/zenodo.23028512](https://doi.org/10.5281/zenodo.23028512).

The programs are in `code/` (Apache-2.0) and need python-flint 0.9.0, mpmath, numpy and scipy
(`code/requirements.txt`). From the folder of this paper:

    python3 -m pip install -r code/requirements.txt
    sh code/run.sh all          # or: tests, 18.5, 6.3, zero

`run.sh` runs `test_lohner6.py` and then, for each proof, from scratch and one process at a time: the numerical centre
(`hp_pulse.py`), the block (`block0.py`), the configuration and the stages of `hh_prove_pulse.py` (setup, interval, K1, K2
and the two negative controls), the independent block check (`hh_block_check_iv.py`) and the summary, whose exit status is
0 if and only if every check passed and every negative control failed. The certificates are
`data/pulse_proof_<T>_El10.613_*.json` (printed E_l) and `data/pulse_proof_18.5_*.json` (zero-current E_l), with the
summaries `*_summary.txt`. The starting profiles `data/pulse_<T>.npz` are numerical initial guesses made by
`pulse_bvp.py`.

## Appendix A. Validated integration

The integrator works with sets X = xbar + C r0 + B r, r0 in R0, r in R (xbar a point of R^6, C and B matrices, R0 and R
boxes), in the variables (y, K) with K' = 0. Write [X] for the interval hull of X and F(W) for an interval enclosure of
f over a box W. The Taylor coefficients x_k(x0) of the solution through x0 (x(t) = sum x_k(x0) t^k) are computed by
the standard recursion on truncated power series (automatic differentiation of the field) in ball arithmetic, so that
for a box Q the computed ball [x_k](Q) contains x_k(x0) for every x0 in Q. The only function beyond the arithmetic
operations and exp is Psi; near 0 it is 1/G with G(x) = (e^x - 1)/x = sum_n x^n/(n + 1)!, whose Taylor coefficients at
a ball x0 with |x0| <= 1/2 are summed to n = 400, with the tail bounded by twice the first omitted term (for
n >= 2k + 2 the ratio of consecutive terms of the k-th coefficient is at most 2|x0|/(n + 2) <= 1/2).

**Lemma A.1 (a priori enclosure).** Since K' = 0, the K component of a solution is constant, and the statement is
about the five moving components y. Let W = W_y x W_K be a box in R^5 x R whose K component W_K contains the K
component of [X], let F_y(W) be an interval enclosure of f(y, K) over W, and suppose that [X]_y + [0, h] F_y(W) is
contained in the interior of W_y, where [X]_y is the hull of the y components of X. Then for every (y0, K) in X the
solution of y' = f(y, K), y(0) = y0, exists on [0, h] and lies in W_y there.

*Proof.* Fix (y0, K) in X; then K is in W_K. Let tau be the supremum of the t in [0, h] such that the solution exists on
[0, t] and lies in W_y there. For t < tau, y(t) = y0 + integral_0^t f(y(s), K) ds lies in y0 + t F_y(W) (F_y(W) is a
box, hence convex), a subset of the compact set [X]_y + [0, h] F_y(W), which lies in int W_y. By Teschl's Corollary
2.15 the solution extends beyond tau, and by continuity it stays in int W_y a little longer; so tau = h, and the
solution lies in W_y on [0, h]. QED

In the six variables the solution (y(t), K) then lies in W on [0, h]; the program checks the inclusion for the five
moving components only (for the endpoint runs, where K is a point, a six-dimensional interior would be empty).

**Lemma A.2 (Lagrange remainder).** Under Lemma A.1, for t in [0, h] and each component i,
x_i(t) - sum_{k <= p} x_{i,k}(x0) t^k lies in t^(p+1) [x_{i,p+1}](W') for any box W' that contains the solution on
[0, h]; if boxes W'_1, ..., W'_m contain it on subintervals that cover [0, h], it lies in the hull of the
t^(p+1) [x_{i,p+1}](W'_j).

*Proof.* Taylor's theorem with the Lagrange remainder for the real function x_i gives the remainder
x_i^(p+1)(xi) t^(p+1)/(p+1)! with xi in (0, t), and x_i^(p+1)(xi)/(p+1)! = x_{i,p+1}(x(xi)) because the system is
autonomous; x(xi) lies in W' (or in the W'_j of a subinterval containing xi). QED

Lemma A.2 with a lower order in place of p and t in [0, h] gives tighter enclosures of the path on [0, h] or on
subintervals, which are then used as W'.

**Lemma A.3 (mean-value form and the new set).** Let Phi(x0) = sum_{k <= p} x_k(x0) h^k, [J] = sum_{k <= p} h^k
[D x_k]([X]), and let Rem be the remainder box of Lemma A.2 at t = h. With y = Phi(xbar) + Rem, xbar' = mid(y),
C' = mid([J] C), B' an invertible point matrix, [B'^-1] an enclosure of its inverse, and

    R' = [B'^-1](y - xbar' + ([J] C - C') R0) + ([B'^-1] [J] B) R,

the solution at time h from every x0 in X lies in X' = xbar' + C' r0 + B' r', r0 in R0, r' in R'.

*Proof.* The point xbar lies in X, and so the segment from xbar to x0 lies in the convex hull of X and in [X]: the
boxes R0 and R contain 0 at every step. R0 is symmetric about 0 by construction and does not change; the initial R
contains 0 by construction; and if R contains 0, so does R', since y - xbar' contains 0 (xbar' = mid(y)),
([J] C - C') R0 contains 0 (R0 does), and ([B'^-1] [J] B) R contains 0 (R does). For each component, the mean value
theorem on that segment gives Phi_i(x0) = Phi_i(xbar) + grad Phi_i(xi_i) (x0 - xbar) with grad Phi_i(xi_i) in the i-th row of
[J]. So x(h) = Phi(x0) + rem with rem in Rem, and x(h) - xbar' - C' r0 = (Phi(xbar) + rem - xbar') + (J C - C') r0
+ J B r for a matrix J in [J]. Multiplying by B'^-1 and enclosing each term gives r' = B'^-1 (x(h) - xbar' - C' r0)
in R'. QED

In the program B' is the orthogonal factor of a QR factorization of mid([J] B), and `arb_mat.inv` encloses its
inverse; Phi(xbar) and the linear algebra are at 256 bits, [J], W and Rem at 128 bits (wider balls, still
enclosures).

**Lemma A.4 (the path of a step).** Under Lemma A.1, for every t in [0, h] the solution lies in
sum_{k <= p} [x_k]([X]) [0, h]^k + [0, h]^(p+1) [x_{p+1}](W). *Proof.* Lemma A.2 with t in [0, h] and the interval
extension of each term. QED

In (H5) this box, mapped to zeta coordinates, is checked to lie in int B0 for every step.

## Appendix B. Interval lemmas

**Lemma B.1 (interval Newton in one variable).** Let g be C^1 on X = [a, b], m in X, G' a closed interval that
contains g'(X) and not 0, and N = m - g(m)/G' (interval arithmetic) contained in the interior of X. Then g has exactly
one zero in X, and it lies in N.

*Proof.* g' has constant sign on X, so g has at most one zero there, and a zero x satisfies x = m - g(m)/g'(xi), in
N, by the mean value theorem. Existence: say G' = [c, d] with c > 0 and g(m) > 0 (the other cases are symmetric, and
g(m) = 0 is trivial). Then N = [m - g(m)/c, m - g(m)/d], and N in int X gives m - g(m)/c > a, so
g(a) <= g(m) - c (m - a) < 0 < g(m), and g has a zero in (a, m). QED

`certify_rest_wave.rest_state` applies Lemma B.1 to the resting current g(u) = I(u, m_inf(u), n_inf(u), h_inf(u)) on
the interval of radius 1e-3 mV around a floating-point zero, and then intersects further Newton steps down to a radius
of about 1e-73.

**Lemma B.2 (interval Cholesky).** Let H be an interval matrix whose lower triangle contains the lower triangle of
every symmetric matrix S in a set. If the Cholesky recursion, carried out in interval arithmetic on the lower triangle
of H, produces pivots whose intervals are positive, then every such S is positive definite.

*Proof.* Run the same recursion on S in exact arithmetic. By induction on the steps, each quantity it computes lies
in the corresponding interval (inclusion isotonicity), so each pivot is positive and the recursion does not break
down. It produces a real lower-triangular L with positive diagonal and S = L L^T, so S is positive definite. QED

In (H3)(C), S = D A + A^T D for A in the interval matrix of the region, and the interval entries of H are evaluated
from those of A, so they contain the entries of S. (H2)(i) uses Lemma B.5 instead.

**Lemma B.3 (covers).** `block0.cover_check` starts from the box {|zeta_1| <= r, |zeta_j| <= rho, j = 2..5}, which
contains B0 (for the entrance check, from the same box with |zeta_1| <= rho), and bisects coordinates in a fixed cycle.
A cell is discarded only when a rigorous lower bound of |zeta_s| over it exceeds rho, so that it misses B0, and it is
accepted only when its check passes; a cell that still fails at the maximal depth ends the program with a failure. So
the accepted cells cover B0 (resp. B0 intersected with {|zeta_1| <= rho}).

**Lemma B.4 (the check of (H1)).** With lambda_u enclosed in [a, b] (P(a) < 0 < P(b), P' > 0 on [a, b]), synthetic
division of the enclosed coefficients of the characteristic polynomial P by x - [a, b] encloses the coefficients of
the quotient Q for every K and every root in [a, b], and the remainder encloses 0. The Routh-Hurwitz inequalities for
the quartic Q/q_4 = x^4 + a3 x^3 + a2 x^2 + a1 x + a0, namely a3, a2, a1, a0 > 0, a3 a2 - a1 > 0 and
a3 a2 a1 - a1^2 - a3^2 a0 > 0 (Teschl 2012, p. 72, eq. (3.45)), are checked on those enclosures. The proof does not use this check; Lemma 0 gives (H1).

**Lemma B.5 (Gershgorin's bound).** Let H be an interval matrix whose entries contain those of every symmetric matrix
S in a set. If, for every i, the lower end of H_ii exceeds the sum over j != i of the upper bounds of |H_ij|, then every
such S is positive definite.

*Proof.* Let S v = lambda v with v real and nonzero (S is symmetric), and let i be an index with |v_i| maximal. Then
(lambda - S_ii) v_i = sum_{j != i} S_ij v_j, so lambda >= S_ii - sum_{j != i} |S_ij| |v_j| / |v_i| >=
S_ii - sum_{j != i} |S_ij| > 0. QED

`certify_rest_wave.lemma_B` applies Lemma B.5 to H = D A + A^T D, with the entries of H evaluated from the interval
matrix A that encloses T_B Df(y, K) T_B^-1 over B x [K1, K2] (for (H2)(i)).

## Appendix C. The prior-article search

Dates: 2026-09-25 to 2026-09-27. Sources: arXiv (abstract and all-field search; the API refused requests from our
machine), zbMATH Open (web and API), PubMed, Crossref, Semantic Scholar (search, and the citing papers of Hastings
1976, Carpenter 1977 and Arioli and Koch 2015), OpenAlex (its free budget was exhausted), a general web search, and the
book of abstracts of Dynamics, Topology and Computations 2025. Query families: "Hodgkin-Huxley" with travelling or
traveling wave or pulse, propagated or propagating action potential, homoclinic, existence or cable, and with
computer-assisted, rigorous numerics, interval arithmetic or validated; and "computer-assisted" with travelling wave,
homoclinic, nerve, excitable or conductance-based. Positive controls: the computer-assisted FitzHugh-Nagumo results
(Arioli and Koch 2015; Czechowski and Zgliczynski 2016) were found each time.

Result: no proof, with or without a computer, of the existence of the pulse of the unmodified 1952 equations, and no
computer-assisted travelling-wave result for Hodgkin-Huxley or any conductance-based model. Every existence proof
found (Hastings 1976; Carpenter 1977; Ikeda, Mimura and Tsujikawa 1987 and 1989, from their abstracts) uses artificial
small parameters. Not read: Hastings (1976) beyond pp. 229-230, Foote and Chen (1981), Huxley (1959), and the
MathSciNet reviews (not reachable); Google Scholar was not reachable. Hastings (1976) beyond pp. 229-230 and Foote and
Chen (1981) could not be obtained. These are the limits of the statement on earlier work in Section 1: we found no
earlier proof, within them.

## References

- Arioli, G., Koch, H. Existence and stability of traveling pulse solutions of the FitzHugh-Nagumo equation. Nonlinear
  Anal. 113 (2015) 51-70. doi:10.1016/j.na.2014.09.023. (Read in the parts cited; not used by the proof.)
- Carpenter, G. A. Nerve impulse equations. In: Structural Stability, the Theory of Catastrophes, and Applications in
  the Sciences, Lecture Notes in Math. 525, Springer, 1976, 58-76. doi:10.1007/BFb0077843. Zbl 0364.92015. (Listed by
  zbMATH next to the 1977 paper; not read; not used by the proof.)
- Carpenter, G. A. A geometric approach to singular perturbation problems with applications to nerve impulse
  equations. J. Differential Equations 23 (1977) 335-367. doi:10.1016/0022-0396(77)90116-4. (Read in full; not used
  by the proof.)
- Conley, C. On traveling wave solutions of nonlinear diffusion equations. In: Dynamical Systems, Theory and
  Applications (J. Moser, ed.), Lecture Notes in Physics 38, Springer, 1975, 498-510. doi:10.1007/3-540-07171-7_13.
  (Credited for the method; not read; not used by the proof.)
- Conley, C. Isolated Invariant Sets and the Morse Index. CBMS Regional Conference Series in Mathematics 38, American
  Mathematical Society, 1978. Zbl 0397.34056. (Credited for the method; not read; not used by the proof.)
- Czechowski, A., Zgliczynski, P. Existence of periodic solutions of the FitzHugh-Nagumo equations for an explicit
  range of the small parameter. SIAM J. Appl. Dyn. Syst. 15 (2016) 1615-1655. doi:10.1137/15M1007707;
  arXiv:1502.02451. (Found in the search; not read.)
- FLINT team. FLINT: Fast Library for Number Theory, version 3.6.0 (which contains Arb), https://flintlib.org; used
  through python-flint 0.9.0, https://github.com/flintlib/python-flint. (The software of the computations.)
- Foote, J. R., Chen, K.-H. Traveling wave properties of the Hodgkin-Huxley equations. Chinese J. Math. 9 (1981)
  1-23. Zbl 0472.35048. (Could not be obtained; bears only on the paragraph on earlier work.)
- Hastings, S. P. On travelling wave solutions of the Hodgkin-Huxley equations. Arch. Rational Mech. Anal. 60 (1976)
  229-257. doi:10.1007/BF01789258. (Read: pp. 229-230; the rest could not be obtained; bears only on the paragraph on
  earlier work.)
- Hodgkin, A. L., Huxley, A. F. A quantitative description of membrane current and its application to conduction and
  excitation in nerve. J. Physiol. 117 (1952) 500-544. (Read: pp. 519-528 and Table 3.)
- Huxley, A. F. Ion movements during nerve activity. Ann. N.Y. Acad. Sci. 81 (1959) 221-246.
  doi:10.1111/j.1749-6632.1959.tb49311.x. (Not read; cited for the slow pulse, which is not claimed.)
- Ikeda, H., Mimura, M., Tsujikawa, T. Slow traveling wave solutions to the Hodgkin-Huxley equations. In: Recent
  Topics in Nonlinear PDE III, Lecture Notes Numer. Appl. Anal. 9 (1987) 1-73; and Japan J. Appl. Math. 6 (1989)
  1-66, doi:10.1007/BF03167914. (Abstracts read; not used by the proof.)
- Johansson, F. Arb: efficient arbitrary-precision midpoint-radius interval arithmetic. IEEE Trans. Comput. 66 (2017)
  1281-1292. doi:10.1109/TC.2017.2690633. (The ball arithmetic the computations use.)
- Lohner, R. J. Einschliessung der Loesung gewoehnlicher Anfangs- und Randwertaufgaben und Anwendungen. Dissertation,
  Universitaet Karlsruhe, 1988. Zbl 0663.65074. (Credited for the method; not read; not used by the proof.)
- mpmath development team. mpmath: a Python library for arbitrary-precision floating-point arithmetic, version 1.3.0,
  https://mpmath.org. (The interval arithmetic of the independent block check.)
- Teschl, G. Ordinary Differential Equations and Dynamical Systems. Graduate Studies in Mathematics 140, American
  Mathematical Society, 2012. (Read in the author's freely available preliminary version, whose page numbers are
  given: Corollary 2.15, p. 52; the Routh-Hurwitz criterion, p. 72; Theorem 6.1, p. 189; Lemmas 6.3 and 6.5, p. 193;
  Lemma 6.6, p. 194; Theorems 9.4 and 9.5, p. 259.)
- Wazewski, T. Sur un principe topologique de l'examen de l'allure asymptotique des integrales des equations
  differentielles ordinaires. Ann. Soc. Polon. Math. 20 (1947) 279-313 (zbMATH gives 1948: Zbl 0032.35001). (Credited
  for the method; not read; not used by the proof.)
- Zgliczynski, P. Covering relations, cone conditions and the stable manifold theorem. J. Differential Equations 246
  (2009) 1774-1819. doi:10.1016/j.jde.2008.12.019. (Credited for the method; not read; not used by the proof.)
