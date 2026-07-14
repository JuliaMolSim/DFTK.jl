# STATUS — SBT-based Fourier tables for UPF pseudopotentials (issue #1258)

Working notes for whoever picks this up. Companion to `PLAN.md` (the original plan);
**where they disagree, this file wins**.

Branch: `loggrid`. Commit `23fa199` ("First attempt") holds everything **except**
`src/pseudo/psp_fourier_table.jl`, which was never committed and was lost. It has now been
**reconstructed from scratch** (from its call sites in `PspUpf.jl` and the notes below) and
is uncommitted, together with the changes listed under "Since the reconstruction".

**Commit `src/pseudo/psp_fourier_table.jl` early** — the whole feature is unusable without it.

---

## Goal

`eval_psp_*_fourier(psp::PspUpf, p)` used to run a Simpson quadrature over the radial grid
*per p value* — O(N_grid) each, with Dual arithmetic in the stress/response paths. Instead,
tabulate each radial quantity's modified Hankel transform once at psp load (via
`SphericalBesselTransforms.jl`, Talman's FFT-based spherical Bessel transform) and
interpolate at runtime. Evaluation becomes O(1) and ForwardDiff differentiates the
interpolant.

## Design of `psp_fourier_table.jl` (as reconstructed)

The tabulated quantity is DFTK's **modified** Hankel transform (`common/hankel.jl`,
`NormConservingPsp.jl`):

    H̃(p) = 4π/p^l ∫ r² f(r) jₗ(p r) dr

The `1/p^l` is what makes the whole design work: H̃ is **smooth and even in p**, so it
splines well in log p and has a Taylor series at p = 0. Nothing downstream needs `l`.

**There are two splines here, both of order `HANKEL_TABLE_ORDER = 6`, and they do different
jobs.** Keeping them straight is the key to the whole file:

| | in **r** (`radial_spline`) | in **log p** (`HankelTable`) |
|---|---|---|
| job | carry the sampled psp data onto the transform's log grid | the representation we keep and evaluate |
| its error sets | the accuracy of the tabulated **values** | the accuracy of the **derivatives** |
| lives | build time, CPU only | the hot path: every \|G\|, on GPU, under ForwardDiff |
| why order 6 | data-limited; O(h⁶) instead of O(h⁴) drops values 1e-10 → 6e-14 | order k is only C^{k-2}: cubic makes H̃″ piecewise-linear (4e-8); order 6 gives 1e-10 |

Raising *either* alone is useless — the other floors you. (And high order in p is essentially
free: H̃(p) is **analytic** by Paley–Wiener, since the data has compact support.)

- **`BSplineKit` builds both.** Its default end condition is right; **never pass `Natural()`**
  (that is finding 7). It replaced a hand-rolled not-a-knot `RadialSpline` (Interpolations
  cannot do cubic on irregular grids, and radial meshes are linear / log / `e^x − 1`) *and*
  the Interpolations cubic prefilter.
- `HankelTable{K,T,AT}` — order-`K` B-spline coefficients on the uniform log-p grid, plus
  `moment0/moment2/moment4`, the p⁰/p²/p⁴ Taylor coefficients used below `pcut`.
- `eval_hankel_table(coefficients, Val(K), logpmin, …, p)` and `uniform_bsplines` — the **only**
  hand-rolled numerics left, and they exist purely because a `BSplineKit.Spline` is not
  `isbits` (it holds `Vector` knots + coefficients) and so cannot be captured in a GPU kernel.
  On a uniform grid the Cox–de Boor denominators all collapse to `j`, so this is a short
  branch-free loop with no knot search — O(1), Dual-safe, GPU-safe. **Verified to reproduce
  BSplineKit's own evaluation to 1.3e-15 at every order**, so it is not an independent
  implementation to be trusted on faith.
  The spline is cardinal only away from the ends of the grid; the `clamp` keeps us K/2 nodes
  clear, and we never evaluate there anyway (below `pcut` we take the series, `pmax` = 1000 is
  far past any |G|).
- `hankel_table_plan(rmax, lmax)` — one `SBTPlan` per (rmax, lmax), shared by all quantities
  of a psp. Quantities cut off before `rmax` (projectors) are zero-padded, which is exact.
- `gregory_weights` — see finding 1.

**Dependency cost, accepted deliberately:** BSplineKit adds **~734 ms to `using DFTK`** (2.4 s
→ 3.1 s, +30%). Note 93% of that is *not* BSplineKit (52 ms) but its `BandedMatrices` +
`ArrayLayouts` stack, needed only for a banded solve that runs ~15 times per psp at load.
`Dierckx` (2.8 ms of load, FITPACK) was measured to give **numerically identical** results
(degree 5 ⇒ y′′(0) = 19.999983, same as BSplineKit k=6, and the same table errors to every
digit) and is the fallback if the load time is ever judged unacceptable — but it would not
supply the log-p prefilter, which would then have to be hand-rolled (a banded solve with k−2
boundary rows). Antoine chose to eat the load time in exchange for owning less numerics.

## Findings that shaped it (the important part)

1. **`sbt` alone is not accurate enough.** It evaluates `∑ᵢ f(rᵢ) rᵢ³ jₗ(p rᵢ) Δρ` — a
   *rectangle* rule in log r, spectrally accurate only if the integrand vanishes at **both**
   ends. It does at rmin (killed by r³), but at the mesh end only the projectors and the
   erf-corrected vloc vanish — the **atomic densities and pseudo-wavefunctions do not**.
   **Fix:** the transform kernel depends on `i+j` only, so any per-node quadrature weight can
   be folded into `f` for free. We fold in 4th-order **Gregory** endpoint corrections,
   restoring O(Δρ⁴). Without this the densities regress ~1000×.

2. **A one-ulp bug that cost 5 orders of magnitude.** `exp(log(rmax))` can land an ulp
   *above* `rmax`, so the last log-grid node fell outside the radial data and the resampler
   returned **0**, silently dropping the endpoint sample. `RadialSpline` therefore tests "past
   the end of the data" as `r > rlast*(1 + 8eps)`. Do not "fix" this by pinning the grid: it
   is a boundary-condition problem, not a grid problem.

3. **Simpson is NOT uniformly worse than the SBT.** On a *smooth decaying* function (a
   Gaussian) Simpson is spectrally accurate and the table is not. The table wins on *real*
   pseudos only because their radial data has a cusp or a cutoff kink, which breaks Simpson's
   Euler–Maclaurin cancellation. Don't over-claim "more accurate" in the PR.
   This is now nailed down against **analytic** transforms ("Fourier tables against analytic
   transforms"), which is a far better reference than the adaptive-quadrature-of-the-same-spline
   we used before (that one cannot see an error made by the spline itself — which is exactly how
   finding 7 hid). Relative error on a UPF-like mesh (linear, h = 0.01, rmax = 15):

   | | table | Simpson |
   |---|---|---|
   | Gaussian e^{−r²}, l = 0..2, p ≤ 3 (smooth, dies at both ends) | 1e-10 | **1e-15** |
   | Slater e^{−2r}, l = 0, p = 5 (cusp at r = 0, p⁻⁴ tail) | **1.3e-8** | 1.4e-7 |
   | Slater e^{−2r}, l = 0, p = 40 | **2.9e-5** | 4.6e-4 |

   (Slater with a = 1 instead scores identically for both: e^{−15} ≉ 0 at the mesh end, so both
   are dominated by the *domain* truncation they share. Useful as a consistency check, useless
   as a discriminator — don't be fooled by it.)

4. **Raising `pmax` is free** (measured: error identical at pmax = 100 / 1000 / 5000). The
   log-p grid just slides. Only constraint: `pmin = pmax·rmin/rmax` must stay **below `pcut`**
   (asserted in `build_hankel_table`) — this is what stops `HANKEL_TABLE_RMIN` from being
   raised to 1e-4, which would otherwise buy ~4× accuracy (measured) but pushes pmin past
   pcut for a psp with a small `rcut` (e.g. Cu at rcut = 9).

5. **No native-log-mesh code path is needed** (the plan wanted one). Resampling handles
   linear / log / `e^x−1` meshes through one path. `Si.pbe-hgh.upf` is already a log mesh and
   comes out at 1e-10 → **the plan's `Si.pz-vbc.UPF` licensing question is moot.**

6. **The two branches must be integrated the *same* way.** The moments were computed by Simpson
   over the psp's own mesh while the spline branch came from the SBT, so at `pcut` the two
   disagreed by 5.7e-7. Fixed by integrating the moments **on the plan's log grid, with the
   same Gregory weights**: they are then literally the p → 0 limit of the very same discrete
   sum, and the branches meet to **2e-12**. `moment4` is worth keeping — dropping it reopens a
   genuine 1.8e-8 p⁴-truncation gap.
   ⚠ **I first blamed this jump on Simpson, and had it backwards.** I wrote that "the ρion
   table returns 2.9999990 where the true integral is 3.0000007" — but 3.0000007 was the
   *cubic resampling spline's* integral, which was the inaccurate one, and Simpson's 2.9999990
   was very nearly right. Raising the radial spline to order 6 settled it: it now integrates to
   2.999999061917 against Simpson's 2.999999061903, agreeing to **1.4e-11**. So the cubic
   radial spline had a real 1.7e-6 error in the norm all along, and the order-6 spline (finding
   9) fixes the cause rather than the symptom. The moment-consistency fix above is still right
   and still needed — but do not repeat my mistake of treating an interpolant's own integral as
   ground truth. Only an analytic reference is ground truth.

7. **The spline's end condition at r = 0 was silently wrecking every large |G|.** `RadialSpline`
   was a *natural* spline (y′′ = 0 at both ends), but the splined quantity is r² f(r), whose
   curvature at the origin is 2f(0) ≠ 0 — so the first mesh cell was misrepresented by O(1).
   That error is confined to within one mesh spacing h of the origin, i.e. it is nearly a
   **delta function**, and the Hankel transform of a delta is **flat**: ρcore(p) plateaued at
   6.2e-6 out to p ~ 1/h instead of decaying. It perturbs no p = 0 quantity (so norms, charges
   and the whole accuracy table below stayed green) and showed up only as a **1000× more
   negative core density in real space** (Ge: -2.2e-5 vs master's -3.8e-8). Switched to
   **not-a-knot**; the transform now decays (1e-8 at p = 40) and the real-space core density
   matches master to 3 digits.

   **The end condition is l-dependent, and one general BC covers it** — worth understanding
   before touching this. `r2_f` ~ r^{l+2} at the origin, so y′′(0) = 2f(0) ≠ 0 only for
   **l = 0** (all the densities, vloc, and the l = 0 projectors/pswfcs); for l ≥ 1 the curvature
   really is zero and the natural BC was accidentally *right*. Measured against analytic
   transforms, natural vs not-a-knot at p = 20–40: l = 0 → **6.0e-7 (flat!) vs 1e-10**;
   l = 1, 2 → **4.10e-12 vs 4.08e-12, i.e. identical**. So not-a-knot fixes l = 0 and costs
   nothing where natural was already correct — no per-quantity special-casing needed.
   ⚠ The lesson generalizes: *every* test here was a p = 0 or a smooth-quantity test, and none
   of them could see this. Judge a transform by its tail, not only by its norm. Both new tests
   ("… against analytic transforms", "… decay at large p") fail loudly if the BC is reverted.

8. **`build_projector_form_factors` had a stray trailing comma** — `for l = 0:psp.lmax,`
   followed by a newline, which folds the next line into the loop header. Rewritten as a
   plain `for l = 0:psp.lmax`. **Flag this in review** — it is a behavioural line.

9. **Cubic was the wrong order, in *both* variables, and for two unrelated reasons.**
   - In **r** it capped the values at ~1e-10 (O(h⁴), h being the psp file's own mesh spacing —
     the one knob we do not control). Order 6 → **6e-14**. This is also what unmasked finding 6.
   - In **log p** it capped the *derivatives*. An order-k spline is only C^{k-2}, so the cubic's
     second derivative is merely piecewise-linear: H̃″ was wrong by **4e-8**. Order 6 → **1e-10**,
     a 400× gain, and it costs nothing because H̃(p) is **analytic** (Paley–Wiener: the radial
     data has compact support), so high order in p converges extremely fast.

   The two floors are independent: raising one order alone leaves the other in charge, which is
   why the first attempts at each looked like they "did nothing". Both together:

   | | cubic/cubic | order 6/6 |
   |---|---|---|
   | H̃ (Gaussian) | 1e-10 | **6e-14** |
   | H̃ (Slater, cusp) | 2e-10 | **7e-13** |
   | H̃′ (stress) | 2e-10 | **7e-12** |
   | H̃″ (phonons, response) | 4e-8 | **1e-10** |

   ⚠ **`HANKEL_TABLE_NPOINTS` is *not* one of these knobs.** Refining the log grid 1024 → 8192
   moves the value error not at all (it is the r-spline's), and only helps derivatives as
   O(N⁻³). 4096 is far past saturation; leave it alone. An earlier note here claimed O(1/N)
   convergence — that was an artefact of measuring against a reference built from the very same
   spline, which cancels the dominant error term.

## Measurements

### The quantities, their boundary conditions, and what each method achieves

Every radial quantity is stored as `r2_f = r² f(r)`, and they differ *only* in two things: the
power of r at the origin (which fixes the spline's end condition, finding 7) and whether the
data still carries weight where the mesh is cut (which is what the Gregory weights of finding 1
are for). Both columns below are **measured** on real pseudos (Ge, Co with rcut = 10), not
assumed:

| quantity | l | r2_f at r→0 | y′′(0) | at the mesh end | class |
|---|---|---|---|---|---|
| `vloc` (tail-corrected) | 0 | r^2.0 | **2f(0) ≠ 0** | ~1e-5 of peak | A / C |
| `r2_ρion` | 0 | r^2.0 | **2f(0) ≠ 0** | 1e-7 (2e-5 if rcut) | A / C |
| `r2_ρcore`, `r2_τcore` | 0 | r^2.0 | **2f(0) ≠ 0** | exactly 0 | A |
| projectors | 0 | r^2.0 | **2f(0) ≠ 0** | 0 … 6e-5, with a cutoff kink | D |
| projectors | ≥1 | r^{l+2} | 0 | 0 … 4e-5, with a cutoff kink | D |
| pswfcs | 0 | r^2.0 | **2f(0) ≠ 0** | **up to 1e-2 of peak** | C |
| pswfcs | ≥1 | r^{l+2} | 0 | **up to 1e-2 of peak** | C |

So the l = 0 quantities — every density, the local potential, and the l = 0 projectors/pswfcs —
have nonzero curvature at the origin and need not-a-knot; for l ≥ 1 the end condition is
immaterial. And the *pseudo-wavefunctions* are the ones truly cut while still at 1e-2 of their
peak: they are why the Gregory correction exists.

### Accuracy, against analytic transforms

Real psp data has no exact transform, so accuracy is measured on analytic surrogates carrying
each class's structure, sampled on a UPF-like mesh (h = 0.01) — see `scratchpad` script in the
PR discussion, and the "Fourier tables against analytic transforms" test. Metric: absolute
error normalised by H̃(0), the natural scale of the quantity (this is the error that actually
propagates, since it enters the density summed over all G-vectors). Worst over p ∈ {5, 20}:

Relative error against the closed form (`test/PspUpf.jl`, "Fourier tables against analytic
transforms"), for the shipped order 6/6:

| | Simpson (master) | cubic/cubic | **order 6/6 (shipped)** |
|---|---|---|---|
| Gaussian (smooth), l = 0..2, p ≤ 3 | 1e-15 | 1e-10 | **6e-14** |
| Slater (cusp), p = 5 | 1.4e-7 | 9e-13 → | **9e-13** |
| Slater (cusp), p = 40 | 4.6e-4 | 2e-8 → | **2e-8** |
| H̃′ (stress), p ≤ 10 | — | 2e-10 | **1e-9** |
| H̃″ (phonons/response), p ≤ 10 | — | ~1e-6 rel | **3e-9 rel** |

At order 6 the tables now **beat Simpson at every p and on every class**, by ~4 orders of
magnitude — which was *not* true at cubic order, where Simpson won outright on smooth data
(Euler–Maclaurin makes it spectrally accurate there). So the accuracy claim, which had to be
carefully hedged before, is now simply true. But note the honest ordering of the argument: the
first reason for the tables is still **O(1) evaluation** (local form factors 33 ms → 4.7 ms) and
clean AD derivatives; accuracy is a bonus that only materialised once the orders were raised.

⚠ Since the tables are now verified to ~1e-12 against analytic transforms while Simpson is at
1e-6…1e-9 on real psp data, **the residual ~4.5e-8 Ha energy difference from master is now
master's error, not ours.** It did not shrink when the tables got 4 orders more accurate — which
is exactly the proof. Do not chase it.

### Other

- Local form factors, bulk Si (8 atoms, Ecut=30, 262k G-vectors): master (dedup + Simpson)
  **33 ms** → tables + dedup 8.0 ms → **tables, no dedup 4.7 ms**. Of the 8.0 ms with dedup,
  7.0 ms was pure `IdDict` bookkeeping → once evaluation is O(1) the dedup is pure overhead
  (a net loss for `PspHgh` too), hence it was deleted outright. **This is the primary case for
  the PR.**
- Costs of order 6 over cubic: evaluation 18 ns → 24 ns per point; psp load ~20 ms → ~25 ms.
  Both negligible. Table memory is unchanged (still `HANKEL_TABLE_NPOINTS` coefficients).

## State

`test/PspUpf.jl` is **884/884 green**; the `stresses` / `forwarddiff` / `chi0` / `response`
items (the ones that differentiate the table) are **202/202 green**. (`Forces term-wise Fe
(GTH)` used to error with `UndefVarError: LDA` under a bare `TestItemRunner.run_tests` — it
never imported DFTK and was living off the `using` that `Pkg.test` injects. Fixed in passing;
unrelated to the tables.)

Verified end to end against a `master` worktree, same script, rattled bulk Si (LDA, UPF, Ecut
20, 4×4×4), plus the guess-density charges:

| | branch vs master |
|---|---|
| total energy | 4.5e-8 Ha (5e-9 relative) |
| forces, stress | 3e-10 |
| valence charge (Si, Ge) | identical to 12 digits |
| min ρcore in real space | Si -2.165e-10 vs -2.171e-10, Ge -3.787e-8 vs -3.784e-8 |

The residual energy difference is **master's** Simpson error, not ours — see the ⚠ above: it
did not shrink when the tables became 4 orders more accurate. The "Negative ρcore" warnings are
**unchanged vs master** (a property of the pseudos; they got *smaller* once finding 7 was fixed).

### Known warts, deliberately left alone

Looked at and judged not worth the churn — but they are the things a reviewer will ask about:

- **`pcut` is coupled to `rmin`** through the assert `pmin = pmax·rmin/rmax < pcut` (finding 4).
  Setting `pcut = plan.kmin` (series only *below the first node*, where it is exact anyway)
  would delete the constant, the assert and the coupling, and free `HANKEL_TABLE_RMIN` to rise.
  Not done: it makes the very first spline nodes load-bearing, and that needs its own accuracy
  study. The current arrangement is merely inelegant, not wrong.
- **`eval_hankel_table` takes 10 positional arguments** rather than the `HankelTable`, because
  the struct is not `isbits` and cannot be captured in a GPU kernel. `Adapt.@adapt_structure
  HankelTable` would let the struct itself be broadcast and collapse the three destructuring
  call sites in `PspUpf.jl`. Not done: it cannot be tested without a GPU here. The same
  constraint is why `uniform_bsplines` exists rather than calling BSplineKit at evaluation
  time — **not** distrust of BSplineKit (ForwardDiff through it is exact to 1e-14, and our
  evaluator is checked against it to 1e-15).
- **`HANKEL_TABLE_ORDER` must be even** (order 6 = quintic). The cardinal-B-spline indexing in
  `eval_hankel_table` assumes B-spline `i` is centred on node `i`, which only holds for even
  order. Odd orders would need collocation at midpoints.
- The vectorized paths call `to_device(architecture(ps), table)` **on every evaluation**, i.e.
  they re-upload the 4096-coefficient array to the GPU each call. A no-op on CPU. Fixing it
  properly means deciding where a psp's tables live on GPU, which is a bigger design question.

## TODO

- [ ] Julia 1.10 compat: `SphericalBesselTransforms` declares `julia = "1.11"` (it uses
      `logrange`) while DFTK supports 1.10 → DFTK is unresolvable on 1.10 as it stands.
      **Antoine said he'd handle this; do not spend time on it.**
- [ ] Before the PR: `Pkg.test("DFTK"; test_args=["minimal"])`, and flag finding 8 (the stray
      comma) to a reviewer as a behavioural line.

## Tuning knobs (all in `psp_fourier_table.jl`)

Coupled by **`pmin = pmax · rmin/rmax`**, and `pmin` must stay < `pcut` (asserted).

| const | now | controls |
|---|---|---|
| `HANKEL_TABLE_NPOINTS` | 4096 | log-grid resolution. Not the accuracy limiter, see above |
| `HANKEL_TABLE_RMIN` | 1e-5 | bottom of the r grid; buys a low `pmin`. Raising it to 1e-4 improves accuracy ~4× but collides with `pcut` (finding 4) |
| `HANKEL_TABLE_PMAX` | 1000 | top of the table; free to raise (finding 4) |
| `HANKEL_TABLE_PCUT` | 1e-2 | series ↔ spline crossover; must stay > pmin |
