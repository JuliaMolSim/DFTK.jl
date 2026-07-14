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

- `HankelTable` — cubic **B-spline coefficients** on a log-p grid (ghost-padded: node `i` ↦
  index `i+1`), plus `moment0/moment2/moment4`, the p⁰/p²/p⁴ Taylor coefficients used below
  `pcut`. The p = 0 branch is exact (`moment0` = Zion for ρion, checked).
- `eval_hankel_table(coefficients, logpmin, Δlogp, n_nodes, pcut, moment0, moment2, moment4, p)`
  — takes plain scalars + one array, not the struct, so it can run in a GPU kernel. We
  evaluate the 4-term B-spline basis by hand for the same reason (an `Interpolations` object
  cannot be captured by a kernel); `Interpolations` is used only to *prefilter* the values
  into coefficients.
- `hankel_table_plan(rmax, lmax)` — one `SBTPlan` per (rmax, lmax), shared by all quantities
  of a psp. Quantities cut off before `rmax` (projectors) are zero-padded, which is exact.
- `RadialSpline` / `resample_radial` — cubic spline with **not-a-knot** end conditions,
  hand-rolled because **Interpolations does not support cubic on irregular grids**
  (`Gridded(Cubic)` errors, and radial meshes are linear / log / `e^x − 1`). Used to move the
  psp data onto the plan's log grid. The end condition is not a detail — see finding 7.
- `gregory_weights` — see finding 1.

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

6. **The two branches must be integrated the *same* way** (this supersedes the old finding 6,
   which blamed the 5.7e-7 jump at `pcut` on truncating the series after p² — wrong: adding
   `moment4` did not close it). The moments were computed by **Simpson over the psp's own
   radial mesh** while the spline branch came from the SBT, so the series inherited Simpson's
   ~1e-6 error: at p = 0 the ρion table returned 2.9999990 where the true integral is
   3.0000007. Fixed by integrating the moments **on the plan's log grid, with the same Gregory
   weights** — they are then literally the p → 0 limit of the same discrete sum, and the two
   branches meet to **2e-12**. Both now agree with an adaptive quadrature to ~6e-12.
   `moment4` is still worth keeping: dropping it reopens a (genuine, p⁴-truncation) 1.8e-8 gap.

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

| class | surrogate | OLD (Simpson) | NEW (tables) | order in mesh h |
|---|---|---|---|---|
| A. smooth, vanishing (l=0) | e^{−r²} | **7e-18** | 1.3e-10 | ≈ 4 |
| B. smooth, vanishing (l=1, 2) | r^l e^{−r²} | **1e-17** | 6.7e-11 | ≈ 4 |
| C. cut while nonzero | e^{−r²} on [0, 2.5] | 2.5e-10 | 4.3e-10 | ≈ 4 |
| D. cutoff kink (projectors) | (1−r/rc)² e^{−r²} | 1.8e-9 | **1.6e-10** | ≈ 4 |
| E. cusp at origin | e^{−2r} | 2.7e-9 | **2.4e-10** | ≈ 4 |

**Read this honestly.** On *smooth* data that dies inside the mesh (class A/B) Simpson is not
merely competitive, it is exact for all practical purposes (1e-17): Euler–Maclaurin makes it
spectrally accurate there, and the tables — capped by the O(h⁴) resampling spline — lose by
seven orders of magnitude. The tables win only where the data is **not** smooth: the cutoff kink
of the projectors and a cusp at the origin (classes D/E), by 10–30×. On class C they tie.
Real pseudos are a blend of all of these, which is why end-to-end the two agree to ~5e-8 Ha in
the total energy. **The case for this PR is O(1) evaluation and clean AD derivatives, not
accuracy** — accuracy is a wash-to-modest-win, and any PR text claiming otherwise is wrong.

### Convergence order

- **In the psp mesh spacing h: ≈ 4**, i.e. the cubic resampling spline, for every class
  including the kinked ones (fits come out 3.8–5.4; the ones above 4 are pre-asymptotic, not
  real). This is the only knob that matters and *we do not control it* — it is the psp file's.
- **In the table grid `HANKEL_TABLE_NPOINTS`: none — the error is flat.** Refining 1024 → 8192
  moves nothing (e.g. class A: 9e-11, 8e-11, 1e-10, 1e-10), because the error is dominated by
  the spline through the psp mesh, not by the log grid. 4096 is already far past saturation and
  is only about the p-interpolation, so it is a fine place to stop — but for the *right* reason.
  (An earlier note here claimed O(1/N); that was an artefact of measuring against a reference
  built from the very same spline, which cancels the dominant error term.)

### Other

- **C² confirmed** (this was the point of the B-spline, and had never been demonstrated):
  across a table node, |ΔH| = 1e-15, |ΔH′| = 1.6e-13, |ΔH″| = **8.1e-12** relative. With the
  old 4-point Lagrange stencil H″ jumped.
- Local form factors, bulk Si (8 atoms, Ecut=30, 262k G-vectors): master (dedup + Simpson)
  **33 ms** → tables + dedup 8.0 ms → **tables, no dedup 4.7 ms**. Of the 8.0 ms with dedup,
  7.0 ms was pure `IdDict` bookkeeping → once evaluation is O(1) the dedup is pure overhead
  (a net loss for `PspHgh` too), hence it was deleted outright. **This is the actual case for
  the PR.**

## State

All of the previous TODO list is done; `test/PspUpf.jl` is **876/876 green**, and the
stress/forces items pass too. (`Forces term-wise Fe (GTH)` used to error with
`UndefVarError: LDA` under a bare `TestItemRunner.run_tests` — it never imported DFTK and was
living off the `using` that `Pkg.test` injects. Fixed in passing; unrelated to the tables.)

Verified end to end against a `master` worktree, same script, rattled bulk Si (LDA, UPF, Ecut
20, 4×4×4), plus the guess-density charges:

| | branch vs master |
|---|---|
| total energy | 5e-8 Ha (6e-9 relative) |
| forces, stress | 3e-10 |
| valence charge (Si, Ge) | identical to 12 digits |
| min ρcore in real space | Si -2.165e-10 vs -2.171e-10, Ge -3.787e-8 vs -3.784e-8 |

The residual energy difference is the tables vs master's Simpson, and the tables are the more
accurate side. The "Negative ρcore" warnings are **unchanged vs master** (they are a property
of the pseudos, and they got *smaller* once finding 7 was fixed).

### Known warts, deliberately left alone

Looked at and judged not worth the churn — but they are the things a reviewer will ask about:

- **`pcut` is coupled to `rmin`** through the assert `pmin = pmax·rmin/rmax < pcut` (finding 4).
  Setting `pcut = plan.kmin` (series only *below the first node*, where it is exact anyway)
  would delete the constant, the assert and the coupling, and free `HANKEL_TABLE_RMIN` to rise.
  Not done: it makes the very first spline nodes load-bearing, and that needs its own accuracy
  study. The current arrangement is merely inelegant, not wrong.
- **`eval_hankel_table` takes 9 positional scalars** rather than the `HankelTable`, because the
  struct is not `isbits` and cannot be captured in a GPU kernel. `Adapt.@adapt_structure
  HankelTable` would let the struct itself be broadcast and collapse the three destructuring
  call sites in `PspUpf.jl`. Not done: it cannot be tested without a GPU here.
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
