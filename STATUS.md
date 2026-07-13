# STATUS — SBT-based Fourier tables for UPF pseudopotentials (issue #1258)

Working notes for whoever picks this up. Companion to `PLAN.md` (the original plan);
**where they disagree, this file wins** — several of the plan's assumptions turned out to be
wrong, see "Findings that changed the plan".

Branch: `master`, **uncommitted working tree**. Nothing has been committed or pushed.

---

## Goal

`eval_psp_*_fourier(psp::PspUpf, p)` used to run a full Simpson quadrature over the radial
grid *per p value* — O(N_grid) each, with Dual arithmetic in the stress/response paths.
Replace it with: tabulate each radial quantity's modified Hankel transform once at psp load
(via `SphericalBesselTransforms.jl`, Talman's FFT-based spherical Bessel transform), then
interpolate at runtime. Evaluation becomes O(1) and ForwardDiff differentiates the
interpolant.

## Current state: WORKS, except an in-progress refactor (see "IN PROGRESS")

### Done and verified

- **`src/pseudo/psp_fourier_table.jl`** (new) — `HankelTable` + builder.
- **`src/pseudo/PspUpf.jl`** — tables built at construction (6–10 ms/psp), all
  `eval_psp_*_fourier` are table lookups.
- **`src/DFTK.jl`** — includes the new file before `PspUpf.jl`.
- **`Project.toml`** — added `SphericalBesselTransforms` (`[deps]` + `[compat] = "0.1"`).
- **`test/PspUpf.jl`** — two new `@testitem`s.
- **Dedup deleted** (this was the follow-up the plan deferred; benchmark said do it now):
  removed the unique-|p| `IdDict` caching from `terms/local.jl`, `density_methods.jl`,
  `terms/nonlocal.jl` (both `build_projector_form_factors` *and* the `radials` cache in
  `build_form_factors`), and `terms/xc.jl` (the plan missed this 4th consumer).
  Form factors are now indexed directly by G; `iG2ifnorm` is gone everywhere
  (`grep -rn iG2ifnorm src/` → nothing).

Antoine approved deleting the dedup and asked for the two changes below.

### IN PROGRESS — half-applied, does not currently run

Switching both interpolations from a 4-point Lagrange stencil to **C² splines**, and
deleting the out-of-range quadrature fallback ("the cliff"). Edits to
`psp_fourier_table.jl` + `PspUpf.jl` are **written but not yet verified**:

1. `HankelTable` now stores **B-spline coefficients**, not values (field renamed
   `values` → `coefficients`, new field `n_nodes`). Prefilter via `Interpolations.jl`
   (already a dep — `interpolate(values, BSpline(Cubic(Line(OnGrid()))))`, take
   `parent(itp.coefs)`, which is ghost-padded so node `i` ↦ index `i+1`). We evaluate the
   4-term B-spline basis ourselves in `eval_hankel_table` because an `Interpolations`
   object cannot be captured by a GPU kernel.
2. `interpolate_radial` (C⁰ Lagrange) replaced by `RadialSpline` + `resample_radial`
   (natural cubic spline, **hand-rolled because Interpolations does *not* support cubic on
   irregular grids** — verified: `Gridded(Cubic)` errors, and radial meshes are log /
   `e^x−1`, i.e. non-uniform).
3. `HANKEL_TABLE_PMAX` 100 → **1000**, and every quadrature-fallback branch deleted from
   `PspUpf.jl` (no more `_eval_psp_projector_fourier_quadrature`, no elementwise
   table-vs-quadrature branch, no device copies of the radial data in the vectorized
   methods).

**Last thing seen:** `MethodError: Cannot convert ... HankelTable to @world(...)HankelTable`
— that is a *stale-REPL world-age artifact* of having changed the struct's fields, **not
necessarily a code bug**. Restart the REPL (`~/.claude/tools/jlrepl.sh restart`) and re-run
before debugging anything.

### TODO to finish

- [ ] Restart REPL, confirm the spline version builds and is accurate (script below).
- [ ] **Re-verify the C² claim**: differentiate `tab(p)` twice with ForwardDiff either side
      of a node; the second derivative should now be continuous (with Lagrange it jumped).
      This was the whole point of the spline change and is **not yet demonstrated**.
- [ ] Check whether the radial spline lifts the ~1e-6 projector error plateau (hypothesis:
      the C⁰ Lagrange kinks at every mesh node were blunting the transform's accuracy).
- [ ] **Out-of-range `p` is currently unhandled**: `eval_hankel_table` clamps its node index,
      so `p > pmax` silently *extrapolates* instead of erroring. With `pmax = 1000`
      (≈ Ecut 4·10⁴ Ha) this is unreachable in practice, but it is a silent-wrong-answer
      path and Antoine explicitly asked for an **error**. Cheapest correct place is a check
      at `PlaneWaveBasis` construction (max |G| vs each psp's `pmax`) — a per-call check in
      the vectorized path would reintroduce a `maximum(ps)` GPU sync, which was deliberately
      removed.
- [ ] Update `test/PspUpf.jl`: the new test item calls `DFTK.interpolate_radial`, which no
      longer exists → use `DFTK.resample_radial(rgrid, r2_f, [r])[1]`. Also the
      "falls back to quadrature past their range" test item is now **obsolete** (there is no
      fallback) — delete or repurpose it.
- [ ] Re-run focused tests (below). **Do not run the full suite — it is too big for this
      machine** (Antoine killed it twice).
- [ ] Julia 1.10 compat: `SphericalBesselTransforms` declares `julia = "1.11"` (it uses
      `logrange`) while DFTK supports 1.10 → DFTK is unresolvable on 1.10 as it stands.
      **Antoine said he'd handle this; do not spend time on it.**

---

## Findings that changed the plan (the important part)

1. **The plan's `sbt` call is wrong as written.** `sbt` evaluates
   `∑ᵢ f(rᵢ) rᵢ³ jₗ(p rᵢ) Δρ` — a *rectangle* rule in log r. That is spectrally accurate only
   if the integrand vanishes at **both** ends of the grid. It does at `rmin` (killed by r³),
   but at the mesh end only the projectors (exactly 0 past their cutoff) and the
   erf-corrected vloc vanish — the **atomic densities and pseudo-wavefunctions do not**
   (Li's `r2_ρion` is still 8% of peak at the last mesh point). There the rule is only
   O(Δρ) and loses ~3 orders of accuracy.
   **Fix:** the transform kernel depends on `i+j` only, so *any* per-node quadrature weight
   can be folded into `f` for free. We fold in **4th-order Gregory endpoint corrections**
   (`gregory_weights`), restoring O(Δρ⁴). Without this the densities regress ~1000×.

2. **A one-ulp bug that cost 5 orders of magnitude.** `exp(log(rmax))` can land an ulp
   *above* `rmax`, so the last log-grid node fell outside the radial data and the resampler
   returned **0** — silently dropping the endpoint sample. Symptom: error O(Δρ) (halving
   with N) instead of O(Δρ⁴), worst on pswfcs (1.4e-5). Fixed by making the
   "past the end of the data" test tolerant to rounding (`r > rlast*(1+8eps)`), **not** by
   pinning the grid (Antoine's call: it's a boundary-condition problem, not a grid problem).
   Worst case went 1.4e-5 → **1.4e-11**.

3. **Simpson is NOT uniformly worse than the SBT** — on a *smooth decaying* function
   (Gaussian) Simpson is spectrally accurate (1e-15) and the table is 1e-9. The table wins on
   *real* pseudos (20–50×) only because their radial data has a cutoff kink, which breaks
   Simpson's Euler–Maclaurin cancellation. Don't over-claim "more accurate" in the PR.
   ⚠ My first "table is 40× more accurate" measurement was an **artifact of a bad reference**
   (QuadGK over a C⁰ interpolant). It only became trustworthy once two *independent*
   interpolants (C⁰ Lagrange and a C² spline) agreed with each other to 1e-9 and both
   disagreed with Simpson by 9e-6. Any future accuracy claim needs the same treatment.

4. **Raising `pmax` is free** (measured): error identical at pmax = 100 / 1000 / 5000. The
   log-p grid just slides; resolution per decade is set by `n_points` and the r log-range.
   Only constraint: `pmin = pmax·rmin/rmax` must stay below `pcut`. Hence pmax=1000 and no
   need for the "auto-size pmax from the basis" machinery.

5. **No native-log-mesh code path is needed** (the plan wanted one). Resampling handles
   linear / log / `e^x−1` meshes through one path. `Si.pbe-hgh.upf` in the test set is
   already a log mesh (r₁=6.5e-5, rmax=100.6) and comes out at 1e-10 → **the plan's
   `Si.pz-vbc.UPF` licensing question is moot, no new test pseudo needed.**

6. **`build_projector_form_factors` had a stray trailing comma** — `for l = 0:psp.lmax,`
   followed by a newline, which folds the next line into the loop header. I rewrote it as a
   plain `for l = 0:psp.lmax`. **Flag this in review** — it's a behavioural line.

## Tuning knobs (all in `psp_fourier_table.jl`)

Coupled by **`pmin = pmax · rmin/rmax`**, and `pmin` must stay < `pcut`.

| const | now | controls | cost to raise |
|---|---|---|---|
| `HANKEL_TABLE_NPOINTS` | 4096 | log-grid resolution — sets **both** the SBT quadrature error O(Δρ⁴) and the interpolation error | ~2 ms/psp per doubling. **The dial to spend on.** |
| `HANKEL_TABLE_RMIN` | 1e-5 | bottom of r-grid; buys a low `pmin` | free — but **~half the 4096 points currently sit below r=0.01**, where the integrand is r³-suppressed. Raising to 1e-4 keeps pmin=7e-4 « pcut and buys ~20% finer spacing for free. **Probably the easiest win.** |
| `HANKEL_TABLE_PMAX` | 1000 | top of table | free (finding 4) |
| `HANKEL_TABLE_PCUT` | 1e-2 | series ↔ spline crossover | must stay > pmin |

## Numbers (measured, pre-spline)

Accuracy vs QuadGK, worst case over all test pseudos & quantities: **1.3e-6** relative
(most 1e-9–1e-11). The Simpson it replaces: 1e-5–1e-6.

Local form factors, bulk Si (8 atoms, Ecut=30, 262k G-vectors, 4184 unique |G| = 62.7×
redundancy):

| | |
|---|---|
| master (dedup + Simpson quadrature) | **33 ms** |
| tables + dedup | 8.0 ms |
| **tables, no dedup** | **4.7 ms** |

Of the 8.0 ms with dedup, **7.0 ms was pure IdDict bookkeeping** and 0.10 ms the actual psp
evaluation → once evaluation is O(1) the dedup is pure overhead. It was a net loss for
**HGH too** (1.81×), so it was deleted outright rather than kept for `PspHgh`'s sake.

Physics unchanged: Si SCF `-34.068186777787`, forces ~1e-16, stress 3.49781e-5.
Per-call results shift ~1e-6 relative (not bit-identical). The pre-existing
"Negative ρcore" warnings are unchanged vs master except in the 6th digit (verified by
stashing).

## How to work on this

**Warm REPL** (see `~/.claude/CLAUDE.md`) — never `julia file.jl`:
```bash
~/.claude/tools/jlrepl.sh eval 'using TestEnv; TestEnv.activate(); using Revise, DFTK'
```
`TestEnv.activate()` is needed for QuadGK / AtomsBuilder / TestItemRunner (test-only deps)
and is **not idempotent** — call it once per fresh REPL, before `using DFTK`.

**Focused tests only** (full suite is too big for this machine):
```julia
using TestItemRunner
TestItemRunner.run_tests("test/"; filter=ti -> occursin("PspUpf.jl", string(ti.filename)))
```
That was 1091/1091 green before the spline refactor. `test/forces.jl` items error with
`UndefVarError: LDA` when run this way — a harness scoping artifact, not our bug.

**Accuracy check** — the reference must be an *adaptive* quadrature of the same radial
function (Simpson is not a valid reference, see finding 3), and evaluate it at a **handful**
of p values, never across the whole grid (that's what made earlier runs hang):
```julia
ref(p) = 4π/p^l * quadgk(r -> r^2*DFTK.resample_radial(rg, r2f, [r])[1] *
                              DFTK.sphericalbesselj_fast(l, p*r),
                         rg[1], rg[end]; rtol=1e-10)[1]
```
Compare the reference **at a table node's own p**, not at a nearby target p — otherwise you
measure |H′|·δp (the node offset) instead of the transform error. This bit me once.
