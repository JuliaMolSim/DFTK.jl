# Plan: SBT-based Fourier evaluation for UPF pseudopotentials (#1258)

> **Note to the implementer:** read this file *and* the discussion on issue #1258, but
> where they differ, this plan is authoritative — it deliberately deviates from some
> early suggestions in the thread (e.g. it interpolates directly on the SBT's log-q
> output grid, and defers deleting the unique-|p| dedup caching to a follow-up PR).

## Problem

Every `eval_psp_*_fourier(psp::PspUpf, p)` call does a full Simpson quadrature over the
radial grid (`hankel` in `src/common/hankel.jl`), i.e. O(N_grid) work *per p value*. The
unique-|G+k| `IdDict` dedup in `terms/nonlocal.jl` / `terms/local.jl` /
`density_methods.jl` only removes repeats within one call — each unique norm still pays a
quadrature, with Dual arithmetic in stress/response paths (e.g. the core-density
evaluation in `Xc` term instantiation), where this is a major bottleneck.
Prerequisite PR #1318 (solid harmonics / modified Hankel transform) is merged.

## Design

Two independent changes, and it takes both to "do better":

1. **Tabulate once, interpolate at runtime** (structural): at psp load, compute each
   quantity's modified transform `H̃_l(q) = 4π/q^l ∫ f j_l(qr) r² dr` on a fixed q-grid;
   `eval_psp_*_fourier` becomes an O(1) interpolation, Duals differentiate the
   interpolant. This is what QE and ABINIT both do (see "Prior art") and is what kills
   the per-p cost, enabling deletion of the dedup caching (antoine's hope in #1258).
2. **Fill the table with a better algorithm** (the point of #1258): Talman's FFT-based
   spherical Bessel transform via
   [SphericalBesselTransforms.jl](https://github.com/azadoks/SphericalBesselTransforms.jl)
   (registered, v0.1.0) — O(N log N) for the whole table instead of O(nq · N_grid) of
   quadrature-per-node, and accurate at *all* q, where composite Simpson degrades once
   q·Δr ≳ 1 (relevant for log meshes whose outer spacing reaches ~1 Bohr; also the reason
   QE/ABINIT need their dq/qmax calibrated against quadrature error). Construction cost
   becomes negligible (ms), so there is no load-time regression to manage.

Conventions: DFTK's `hankel` = `4π·sbt(l, f; normalize=false)/p^l` (verified against
`examples/psp.jl` in the SBT repo, which compares to DFTK-style HGH analytics).

### Prior art (QE / ABINIT), for validation of the runtime side

- **QE** (`upflib/beta_mod.f90`, `vloc_mod.f90`, …): uniform q-grid, `dq = 0.01` Bohr⁻¹,
  filled by Simpson quadrature on the native mesh; 4-point Lagrange cubic interpolation;
  stress derivatives from differentiated Lagrange weights; tables interpolated on-GPU;
  vloc handled with the same `Z·erf(r)/r` subtraction DFTK uses, q = 0 limit stored
  separately.
- **ABINIT** (`m_pspini.F90`, `m_psp8.F90`): linear q-grid (mqgrid default 3001, qmax
  from ecut + `dilatmx` margin), corrected-trapezoid fill, cubic-spline evaluation;
  stores `q²·Vloc(q)` to regularize the Coulomb tail.

Neither uses a fast transform — their fill is the part we improve on. What they validate:
fixed-grid tabulation + cubic interpolation, interpolant derivatives for stress,
device-side interpolation, and the erf-tail treatment.

## Implementation steps

### 1. Dependency — ⚠ blocker to resolve first
Add `SphericalBesselTransforms` to `Project.toml` `[deps]`, `[compat] = "0.1"`.
Transitive deps (Bessels, SpecialFunctions, FFTW) are light / already present.

**However**: the package declares `julia = "1.11"` (it uses `logrange`, added in Julia
1.11) while DFTK supports and CI-tests Julia 1.10 — adding the dep as-is makes DFTK
unresolvable on 1.10. Fix upstream first: PR against SphericalBesselTransforms.jl
replacing `logrange(a, b, n)` with `exp.(range(log(a), log(b), n))` (or a `Compat.jl`
dep) and lowering compat to `julia = "1.10"`, then wait for the release. Local
development can proceed on Julia ≥ 1.11 meanwhile, but the DFTK PR cannot merge before
the upstream release is out.

### 2. Transform + table helper (new `src/pseudo/psp_fourier_table.jl`)

Per psp: one log r-grid + one `SBTPlan(rlog, lmax + 1, kmax)` shared by all quantities
(plan discarded after construction — it holds mutable FFTW caches). Per quantity
`(rgrid, r2_f, l)`:

1. **f on the log grid.** UPF prescribes no mesh form; in practice:
   - native log mesh `r_i = e^{x_min+(i−1)dx}/z_mesh` (ld1.x output: legacy QE-website NC
     files — verified on `Si.pz-vbc.UPF` —, pslibrary NC, FHI conversions; the norm for
     US/PAW should DFTK ever support them): use directly, `f = r2_f ./ rgrid.^2`,
     zero-padding truncated projectors (exact). Detect log-ness ourselves and hand
     `SBTPlan` the exactly reconstructed `logrange` (its `≈` check has rtol √eps, file
     values are rounded).
   - linear mesh starting at 0 (ONCVPSP output: PseudoDojo/SG15/SPMS — the overwhelmingly
     common case for DFTK) or ld1.x's log-like `r_i ∝ e^{(i−1)dx} − 1` variant: resample
     with **cubic** interpolation (meshes are uniform → `BSpline(Cubic)` directly;
     resampling error O(h⁴) ≈ 1e-8; linear interpolation would cap accuracy at ~1e-5–1e-6
     — not good enough). Log grid: `rmin ≈ 1e-4`, `rmax = last(rgrid)`, `N ≈ 4096`
     (Δρ ≈ 3.5e-3 — calibrate in step 5).
2. **Transform:** `g = sbt(l, f, plan)`; `H̃` on `plan.k` is `4π g ./ plan.k.^l`.
3. **Runtime interpolant, directly on the SBT output grid.** `plan.k` is uniform in
   `κ = log k`, so this *is* a uniform-grid table — no regridding:

   ```
   struct HankelTable{T}   # callable
       values::Vector{T}   # H̃_l on plan.k
       κmin::T, Δκ::T      # log-k grid parameters
       l::Int
       pcut::T             # below this, use the series
       moment::T           # H̃_l(0) = 4π ∫ f r^{l+2} dr / (2l+1)!!
   end
   ```

   Evaluation `(tab::HankelTable)(p)`:
   - `p ≤ pcut` (includes p = 0, present in every basis): series value from `moment`
     (computed at build time by the existing small-p branch of `hankel`; extend with the
     next moment if the spline/series mismatch at `pcut` demands it) — this also
     sidesteps trusting `4π g/k^l` (tiny/tiny, l ≥ 1) near the grid's kmin;
   - `pcut < p ≤ kmax`: 4-point Lagrange cubic at index `(log(p) − κmin)/Δκ` — ~10
     lines, generic in `p` so Duals flow through, `isbits`-friendly so GPU kernels can
     capture it (values as device array + scalars), at the cost of one `log` per
     evaluation;
   - `p > kmax`: fall back to direct `hankel` quadrature (correct, just slow — do NOT
     silently return 0). `kmax ≈ 100` Bohr⁻¹ default, generous for density cube grids.

### 3. Wire into `PspUpf` (`src/pseudo/PspUpf.jl`)

Tables built at construction for:

| quantity | l | note |
|---|---|---|
| `vloc + Zion·erf(r)/r` | 0 | tail-corrected part, as today; `eval_psp_local_fourier` returns `H̃₀(p) − 4π·Zion/p²·exp(−p²/4)`, keeps `p == 0 → 0` |
| each `r2_projs[l+1][i]` | l | respect per-projector cutoff as today |
| each `r2_pswfcs[l+1][i]` | l | full mesh (no rcut), as today |
| `r2_ρion`, `r2_ρcore`, `r2_τcore` | 0 | |

Replace the scalar `eval_psp_*_fourier` bodies with table lookups; vectorized/GPU methods
become a `map` with the table's array moved `to_device` (deletes the
`default_psp_quadrature` plumbing). `eval_psp_energy_correction` keeps its quadrature
(single evaluation). Keep the quadrature path for `PspUpf{T}, T != Float64` (SBTPlan
internals hardcode Float64; arbitrary-precision users keep today's behavior). `hankel` +
its ForwardDiff rule stay (small-q moments, fallback, tests). `PspHgh` (analytic) and
`PspLinComb` (delegates) untouched.

### 4. AD correctness

Duals differentiate the interpolant (QE-validated practice for stress). Verify:
- `test/stresses.jl` (UPF cases) and forwarddiff/response test items;
- spot-check `d/dp` of the table against the exact rule
  `H̃'_l(p) = −4π/p^l ∫ f r³ j_{l+1}(pr) dr` (= `−4π·sbt(l+1, r·f)/p^l`). Escape hatch if
  cubic-derivative accuracy disappoints: a derivative table at `l+1` + a one-line Dual
  method.

### 5. Tests & calibration

- New `@testitem` in `test/PspUpf.jl`: for a linear-mesh UPF (PseudoDojo Li/Si), a psp8,
  and a log-mesh NC UPF (add legacy `Si.pz-vbc.UPF` to `test/pseudos/`), compare every
  `eval_psp_*_fourier` against direct `hankel` quadrature on a dense p-range (0, near-0,
  mid, near-kmax). Discrepancies at large q need judgment — the SBT is *more* accurate
  than Simpson there; arbitrate a few points with QuadGK. Calibrate `rmin`, `N`, `pcut`;
  target ~1e-7 agreement in the well-resolved region.
- Existing consistency tests (real↔Fourier, vs-HGH; tolerances 1e-2–1e-5) pass unchanged.
- Full check: `Pkg.test("DFTK"; test_args=["minimal"])`, then psp/stress/AD/gpu tags.

### 6. Benchmark (goes in the PR description)

Before/after: (a) `PlaneWaveBasis` + term instantiation for a medium UPF system,
(b) `compute_stresses_cart(scfres)`, (c) a response/ForwardDiff call — the Dual paths are
where the win is expected. Also psp load time (should stay ~ms thanks to the SBT fill).

### 7. Follow-up (separate PR, benchmark-gated)

Delete the unique-|p| `IdDict` dedup caching (`build_projector_form_factors`,
`atomic_local_form_factors`, `atomic_density_form_factors`) — with O(1) evaluation it is
likely pure complexity, but it also serves `PspHgh` and reduces GPU work, so benchmark
first, delete second.

## Implementation gotchas

- **`sbt`'s `np_in` argument** sets the small-r extrapolation power (`f ∝ r^{np_in+ℓ}`
  below rmin). The default `np_in = 1` matches projectors/pswfcs (`β_l ~ r^{l+1}`);
  densities and the erf-corrected vloc behave like `r⁰` near 0, so pass `np_in = 0` for
  the `l = 0` quantities. The r < rmin contribution is r²-suppressed and tiny either
  way, but be deliberate.
- **Use `sbt`, not `sbt_large_k`** — `sbt` internally stitches the large-k and small-k
  algorithms (picking the crossover automatically); the pieces alone are wrong at the
  other end.
- **`SBTPlan` is not thread-safe** (shared internal work buffers) — do not
  `Threads.@threads` the table fill over quantities sharing one plan.
- **`eval_psp_local_fourier(psp, 0)` must keep returning 0** (compensating charge
  background) — the table's p → 0 limit of the erf-corrected part is finite and nonzero;
  the zero is an API contract applied on top, not a table value.
- **Below `pcut` the constant series value has zero derivative.** That is exact at
  p = 0 (`H̃` is even in p) but approximate on (0, pcut]. Smallest nonzero |G+k| in
  practice is ~2π/L ≈ 0.1–0.5 Bohr⁻¹, so keep `pcut` ≲ 1e-2 and this never matters; if
  calibration wants a larger `pcut`, add the p² term to the series instead.
- **Add explicit Dual-evaluation tests** for the table path: the existing `hankel`
  ForwardDiff rule keeps passing its own tests while silently no longer being exercised
  on the fast path — don't let AD coverage regress by accident.
- **Check provenance/licensing before committing `Si.pz-vbc.UPF`** to `test/pseudos/`
  (QE-website pseudos are GPL-adjacent); follow whatever `test/pseudos/README.md` does
  for the existing files.

## Open questions / risks

1. **Small-k accuracy for l ≥ 1** (`4π g/k^l` near the log-grid's kmin): handled by the
   `p ≤ pcut` series branch; `pcut` is a calibration output of step 5.
2. **Resampling error for linear input meshes**: cubic keeps it at O(h⁴) ≈ 1e-8, further
   suppressed near r = 0 by the r² weight; the SBT's small-r power-law extension
   (`np_in + l`) covers r < rmin. Confirm in step 5.
3. **kmax = 100 default**: generous for the density cube grid at realistic Ecut; beyond
   it the quadrature fallback is correct-but-slow. Acceptable. (Note kmax also fixes
   kmin = kmax·rmin/rmax and thus the grid's low end; with pcut this is a non-issue.)
4. **Interpolation vs today's exact quadrature**: results shift at the ~1e-8 level; watch
   for tolerance trips in tight regression tests (SCF energies, forces vs finite
   differences).
5. **Serialization**: new fields are plain arrays — JLD2 scfres round-trips unaffected
   (test item exists).
6. Upstream is v0.1.0 with a rough interface (author's own words); if we need e.g. a
   generic-eltype plan or a relaxed grid check, file issues/PRs against
   SphericalBesselTransforms.jl rather than working around it here.
