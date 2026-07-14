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
- `RadialSpline` / `resample_radial` — natural cubic spline, **hand-rolled because
  Interpolations does not support cubic on irregular grids** (`Gridded(Cubic)` errors, and
  radial meshes are linear / log / `e^x − 1`). Used to move the psp data onto the plan's
  log grid.
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
   pseudos (20–50×) only because their radial data has a cutoff kink, which breaks Simpson's
   Euler–Maclaurin cancellation. Don't over-claim "more accurate" in the PR.
   ⚠ Any accuracy claim needs an *adaptive* quadrature of the **same radial spline** as
   reference (`DFTK.RadialSpline` + QuadGK). Simpson is not a valid reference, and neither is
   QuadGK over a *different* interpolant.

4. **Raising `pmax` is free** (measured: error identical at pmax = 100 / 1000 / 5000). The
   log-p grid just slides. Only constraint: `pmin = pmax·rmin/rmax` must stay **below `pcut`**
   (asserted in `build_hankel_table`) — this is what stops `HANKEL_TABLE_RMIN` from being
   raised to 1e-4, which would otherwise buy ~4× accuracy (measured) but pushes pmin past
   pcut for a psp with a small `rcut` (e.g. Cu at rcut = 9).

5. **No native-log-mesh code path is needed** (the plan wanted one). Resampling handles
   linear / log / `e^x−1` meshes through one path. `Si.pbe-hgh.upf` is already a log mesh and
   comes out at 1e-10 → **the plan's `Si.pz-vbc.UPF` licensing question is moot.**

6. **The p⁴ term of the series is needed.** With only `moment0 + moment2 p²`, the series and
   the spline disagree by **5.7e-7** at `pcut = 1e-2` (a visible discontinuity). Adding
   `moment4` brings the two branches together. In practice only p = 0 lands in the series
   branch (the smallest |G| of a real cell is ≫ pcut), but the jump was real.

7. **`build_projector_form_factors` had a stray trailing comma** — `for l = 0:psp.lmax,`
   followed by a newline, which folds the next line into the loop header. Rewritten as a
   plain `for l = 0:psp.lmax`. **Flag this in review** — it is a behavioural line.

## Measurements (all reproduced after the reconstruction)

Accuracy vs an adaptive quadrature of the same radial spline, at p ∈ {0.3, 1.1, 2.7, 6.5,
15, 40}, over Si (log mesh) / Li / Co (NLCC + rcut):

| | |
|---|---|
| densities, pswfcs, most projectors | **1e-9 – 1e-11** |
| worst case (Li projectors, kink-limited) | **4.5e-7** |
| the Simpson quadrature it replaces | 1e-5 – 1e-6 |

- **C² confirmed** (this was the point of the B-spline, and had never been demonstrated):
  across a table node, |ΔH| = 1e-15, |ΔH′| = 1.6e-13, |ΔH″| = **8.1e-12** relative. With the
  old 4-point Lagrange stencil H″ jumped.
- **Grid resolution is not the limiter.** Worst-case projector error vs `HANKEL_TABLE_NPOINTS`:
  2048 → 8.5e-7, 4096 → 4.5e-7, 8192 → 2.6e-7, 16384 → 9.8e-8. That is **O(1/N), not O(Δρ⁴)**
  — the error is dominated by the projector's cutoff kink, so spending points is a poor trade.
  4096 (≈ 3 ms/psp of table build) is the right place to stop.
- Local form factors, bulk Si (8 atoms, Ecut=30, 262k G-vectors): master (dedup + Simpson)
  **33 ms** → tables + dedup 8.0 ms → **tables, no dedup 4.7 ms**. Of the 8.0 ms with dedup,
  7.0 ms was pure `IdDict` bookkeeping → once evaluation is O(1) the dedup is pure overhead
  (a net loss for `PspHgh` too), hence it was deleted outright.

## Since the reconstruction (all uncommitted, **not yet all verified**)

- `src/pseudo/psp_fourier_table.jl` — new (rewritten).
- `moment4` added: `HankelTable` field, `eval_hankel_table` argument, and the three
  vectorized `eval_psp_*_fourier` call sites in `PspUpf.jl` destructure/pass it.
- **Out-of-range p is now an error, as asked.** `eval_hankel_table` clamps (a GPU kernel
  cannot raise, and a `maximum(ps)` check in the vectorized path would reintroduce a GPU
  sync), so the check lives where it is cheap and exact: `max_momentum_fourier(psp)`
  (`NormConservingPsp.jl`: `Inf`; `PspUpf.jl`: `psp.vloc_table.pmax`) is compared against the
  largest |G + q| of the FFT cube once, in the `PlaneWaveBasis` inner constructor.
  With pmax = 1000 this is unreachable in practice (Ecut ≈ 5·10⁵ Ha). **Untested.**
- `test/PspUpf.jl`: the two new items were repaired — the old one called the (now gone)
  `interpolate_radial`, and the "falls back to quadrature past their range" item is obsolete
  (there is no fallback any more) and was replaced by one covering the small-p series, the C²
  crossover and the vectorized paths.
- `Project.toml` / `Manifest.toml`: `SphericalBesselTransforms` v0.1.0 (+ `Bessels`) resolved.

## TODO

- [ ] **Run the PspUpf test items to green.** Last complete run (before `moment4`) was
      688 pass / 1 fail / 1 error: the fail was finding 6 (fixed by `moment4`, needs a rerun);
      the error was an interrupted psp parse, an artifact of the run, not a code bug.
      `TestItemRunner.run_tests("test/"; filter=ti -> occursin("PspUpf.jl", string(ti.filename)))`
      Note the accuracy item is the expensive one — keep its p list and pseudo list short, and
      build the reference `RadialSpline` **once per quantity**, never inside the integrand.
- [ ] Sanity-check the physics end to end (Si LDA, forces, stresses); the pre-tables numbers
      to compare against are energy `-34.068186777787`, forces ~1e-16, stress 3.49781e-5, with
      per-call results shifting ~1e-6 relative (not bit-identical). Also confirm the
      pre-existing "Negative ρcore" warnings are unchanged vs master.
- [ ] Exercise the new `PlaneWaveBasis` momentum check at least once (e.g. temporarily lower
      `HANKEL_TABLE_PMAX`) — it has never been triggered.
- [ ] Julia 1.10 compat: `SphericalBesselTransforms` declares `julia = "1.11"` (it uses
      `logrange`) while DFTK supports 1.10 → DFTK is unresolvable on 1.10 as it stands.
      **Antoine said he'd handle this; do not spend time on it.**

## Tuning knobs (all in `psp_fourier_table.jl`)

Coupled by **`pmin = pmax · rmin/rmax`**, and `pmin` must stay < `pcut` (asserted).

| const | now | controls |
|---|---|---|
| `HANKEL_TABLE_NPOINTS` | 4096 | log-grid resolution. Not the accuracy limiter, see above |
| `HANKEL_TABLE_RMIN` | 1e-5 | bottom of the r grid; buys a low `pmin`. Raising it to 1e-4 improves accuracy ~4× but collides with `pcut` (finding 4) |
| `HANKEL_TABLE_PMAX` | 1000 | top of the table; free to raise (finding 4) |
| `HANKEL_TABLE_PCUT` | 1e-2 | series ↔ spline crossover; must stay > pmin |
