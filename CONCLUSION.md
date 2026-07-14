# SBT-based Fourier tables for UPF pseudopotentials — conclusion

What the branch does: `eval_psp_*_fourier(psp::PspUpf, p)` used to run a Simpson quadrature
over the radial mesh *per p value*. Instead, each radial quantity's modified Hankel transform

    H̃(p) = 4π/p^l ∫ r² f(r) jₗ(p r) dr

is tabulated once at psp load (via `SphericalBesselTransforms`, Talman's FFT-based spherical
Bessel transform) and interpolated at runtime. Evaluation becomes O(1) and ForwardDiff
differentiates the interpolant.

**The case for the change is speed, not accuracy.** Accuracy is a bonus that only materialised
once the spline orders were raised (see below) — and at the cubic order this was first written
with, derivatives were *worse* than master's.

---

## The headline: timings

Bulk Si (2 atoms, Ecut = 30, 64k G-vectors, 2×2×2 k-grid), measured on this machine, master vs
this branch:

| | master | this branch | |
|---|---|---|---|
| `eval_psp_local_fourier`, one p | 21 000 ns | **59 ns** | **355×** |
| `eval_psp_projector_fourier`, one p | 2 010 ns | **59 ns** | **34×** |
| local form factors | 48.9 ms | **4.2 ms** | **12×** |
| density form factors | 42.7 ms | **3.8 ms** | **11×** |
| `PlaneWaveBasis` construction | 117 ms | **28.7 ms** | **4×** |
| `compute_forces_cart` | 133 ms | **42.8 ms** | **3×** |
| **`compute_stresses_cart`** | **9 686 ms** | **1 198 ms** | **8×** |

Stresses are the big one: they run the whole form-factor machinery in `Dual` arithmetic, where
an O(n_radial) quadrature per p hurts most.

Paid for with:

| | master | this branch |
|---|---|---|
| `load_psp` (per pseudopotential) | 4.5 ms | 24.6 ms |
| `using DFTK` | 2.4 s | 3.1 s (**+30%**, all of it BSplineKit's deps) |

The load-time regression is a deliberate trade, and it is **not** BSplineKit's own code (52 ms)
but its `BandedMatrices` + `ArrayLayouts` stack (677 ms), pulled in for a banded solve that runs
~15 times per psp. `Dierckx` (2.8 ms of load, FITPACK) was measured to give **numerically
identical** results and is the fallback if the load time is judged unacceptable — but it would
not supply the log-p prefilter, which would then have to be hand-rolled.

---

## The quantities being integrated, and their boundary conditions

Everything is stored as `r2_f = r² f(r)`. The quantities differ in exactly two ways, and both
matter:

| quantity | l | r2_f as r→0 | y′′(0) | at the mesh end | class |
|---|---|---|---|---|---|
| `vloc` (erf tail-corrected) | 0 | r² | **2f(0) ≠ 0** | ~1e-5 of peak | smooth |
| `r2_ρion` | 0 | r² | **2f(0) ≠ 0** | 1e-7 (2e-5 if rcut) | smooth |
| `r2_ρcore`, `r2_τcore` | 0 | r² | **2f(0) ≠ 0** | exactly 0 | smooth, peaked |
| projectors | 0 | r² | **2f(0) ≠ 0** | 0…6e-5, cutoff kink | kinked |
| projectors | ≥1 | r^{l+2} | 0 | 0…4e-5, cutoff kink | kinked |
| pswfcs | 0 | r² | **2f(0) ≠ 0** | **up to 1e-2 of peak** | cut while nonzero |
| pswfcs | ≥1 | r^{l+2} | 0 | **up to 1e-2 of peak** | cut while nonzero |

Two consequences, both of which caused real bugs:

1. **Every l = 0 quantity has nonzero curvature at the origin.** A *natural* spline (y′′ = 0)
   is therefore O(1) wrong in the first mesh cell. For l ≥ 1 it happens to be right.
2. **The pseudo-wavefunctions are genuinely cut while still at 1e-2 of their peak**, which is
   why the transform needs Gregory endpoint corrections.

---

## Accuracy, default settings

Relative error against *closed-form* transforms (`test/PspUpf.jl`, "Fourier tables against
analytic transforms"), on a typical UPF mesh (linear, h = 0.01, rmax = 15). A Gaussian
`r^l e^{-r²}` stands for the smooth quantities; a Slater `e^{-2r}` (cusp at the origin, p⁻⁴
tail, so there is real signal at high p) for the hard case.

### Values

| | master (Simpson) | **this branch** |
|---|---|---|
| Gaussian, l = 0..2, p ≤ 3 | 1e-15 | **6e-14** |
| Slater, p = 1 | 4.2e-9 | **7.0e-13** |
| Slater, p = 5 | 1.4e-7 | **9.3e-13** |
| Slater, p = 10 | 1.8e-6 | **6.6e-11** |
| Slater, p = 20 | 2.8e-5 | **1.3e-9** |
| Slater, p = 40 | 4.6e-4 | **2.0e-8** |

The tables beat Simpson by ~4 orders on realistic (cusped/kinked) data. On a perfectly smooth
Gaussian, Simpson is spectrally accurate (Euler–Maclaurin) and still wins — do not over-claim.

### Derivatives (what stresses, phonons and response differentiate)

| Slater | master H′ | **branch H′** | master H″ | **branch H″** |
|---|---|---|---|---|
| p = 1 | 7.7e-11 | 7.7e-11 | 4.4e-10 | 1.9e-11 |
| p = 5 | 8.2e-11 | 1.6e-10 | 3.3e-9 | 3.4e-9 |
| p = 10 | 2.5e-9 | 1.2e-9 | 2.7e-8 | 2.6e-8 |

**Derivatives are a wash.** Master differentiates its quadrature *through the integrand*
(ForwardDiff hits `jₗ` analytically), so its derivatives are accurate despite its values not
being. The tables' win on derivatives is **speed (8× on stresses), not accuracy.**

⚠ This is exactly why the spline order had to go up. At the cubic order this was first written
with, the branch's H″ was ~1e-6 relative — a **regression** against master. Order 6 was the
price of not regressing, not a luxury.

---

## The knobs, and which ones actually do anything

All in `psp_fourier_table.jl`.

| knob | now | what it controls | worth turning? |
|---|---|---|---|
| `HANKEL_TABLE_ORDER` | 6 | B-spline order in **r** *and* in **log p** | **This is the knob.** See below. |
| `HANKEL_TABLE_NPOINTS` | 4096 | log-grid resolution | **No.** Error is flat in it. |
| `HANKEL_TABLE_PMAX` | 1e3 | top of the table | free to raise; unreachable in practice |
| `HANKEL_TABLE_RMIN` | 1e-5 | bottom of the r grid | coupled to `pcut`, see below |
| `HANKEL_TABLE_PCUT` | 1e-2 | series ↔ spline crossover | must stay > `pmin = pmax·rmin/rmax` |

**`HANKEL_TABLE_ORDER` is really two knobs that happen to share a value**, and they floor
different things — which is why raising either alone looks like it does nothing:

- the spline **in r** carries the psp data onto the transform's log grid, and caps the
  **values** (order 4: 1e-10; order 6: 6e-14). Its convergence is O(h⁴)/O(h⁶) in *the psp
  file's own mesh spacing*, which we do not control.
- the spline **in log p** is what we evaluate and differentiate, and caps the **derivatives**.
  An order-k spline is only C^{k-2}, so a cubic has a piecewise-linear second derivative
  (H″ error 4e-8, vs 1e-10 at order 6). High order is cheap here because H̃(p) is *analytic*
  (Paley–Wiener: the radial data has compact support).

Measured, at p = 5, absolute error vs the closed form:

| k_r | k_p | H̃ | H̃′ | H̃″ |
|---|---|---|---|---|
| 4 | 4 | 8e-10 | 3e-11 | **4e-08** |
| 4 | 6 | 7e-10 | 2e-10 | 1e-10 |
| **6** | 4 | **1e-12** | 2e-10 | **4e-08** |
| **6** | **6** | **6e-14** | **7e-12** | **1e-10** |

Order 8 buys nothing at N = 4096. Orders must be **even** (the cardinal-B-spline indexing in
`eval_hankel_table` assumes B-spline i is centred on node i).

**`HANKEL_TABLE_NPOINTS` is a trap.** Refining 1024 → 8192 moves the value error not at all
(it belongs to the r-spline) and helps derivatives only as O(N⁻³). 4096 is far past saturation.
An earlier note claimed O(1/N) convergence; that was an artefact of measuring against a
reference built from the very same spline, which cancels the dominant error term.

---

## Bugs found along the way (all fixed, all now covered by tests)

1. **A natural spline put a delta function at the origin.** `r² f(r)` has curvature 2f(0) ≠ 0
   for l = 0, so pinning y′′ = 0 is O(1) wrong in the first cell. Confined to one mesh spacing
   of r = 0, that error is nearly a delta — and the Hankel transform of a delta is **flat**, so
   it left every p = 0 quantity (norms, charges) *exactly right* while putting a 6e-7 floor
   under the entire tail of H̃. It surfaced only as a core density 1000× more negative in real
   space than master's. **Every existing test was a p = 0 or smooth-quantity test, and none
   could see it.** Judge a transform by its tail, not only by its norm.
2. **The small-p series and the spline were integrated differently**, leaving a 5.7e-7 step at
   the crossover. The moments are now integrated on the same log grid, with the same Gregory
   weights, as the transform itself — so they are literally the p → 0 limit of the same
   discrete sum, and the branches meet to 2e-12.
3. **Out-of-range p extrapolated silently.** The evaluation is a GPU kernel and cannot raise, so
   `PlaneWaveBasis` now checks the largest |G + q| of the FFT cube against
   `max_momentum_fourier(psp)` once, at construction.

Two of my own diagnoses were **wrong and are recorded as such** in `STATUS.md`:

- I blamed the crossover step on Simpson, taking the *cubic spline's own integral* as ground
  truth. It was the spline that was wrong, by 1.7e-6 in the norm. Only an analytic reference is
  ground truth.
- I flagged a stray trailing comma in `build_projector_form_factors` as a behavioural line. It
  is not: Julia iterates a `Number` once, yielding it, so the loop was equivalent. Cosmetic.

---

## State

- `test/PspUpf.jl`: 884/884. The differentiating suites (`stresses`, `forwarddiff`, `chi0`,
  `response`): 202/202.
- End-to-end vs a `master` worktree (rattled bulk Si, LDA, UPF, Ecut 20, 4×4×4): energy agrees
  to 4.5e-8 Ha, forces and stress to 3e-10, guess-density charges to 12 digits. **That residual
  is master's Simpson error, not ours** — it did not shrink when the tables became 4 orders more
  accurate, which is the proof.

### Left to do

- **Julia 1.10 compat:** `SphericalBesselTransforms` declares `julia = "1.11"` (it uses
  `logrange`) while DFTK supports 1.10, so DFTK is currently unresolvable on 1.10. Blocks CI.
- **Drop `PLAN.md` / `STATUS.md` / this file before the PR** — they are working notes, not
  library content.

### Known warts, deliberately left

- `pcut` is coupled to `rmin` by an assert. Setting `pcut = plan.kmin` would delete the
  constant, the assert and the coupling, but makes the first spline nodes load-bearing.
- `eval_hankel_table` takes 8 positional arguments rather than the struct, and the vectorized
  path re-uploads the coefficients to the GPU on every call. Both are because a `HankelTable`
  is not `isbits`; `Adapt.@adapt_structure` would fix them, but cannot be tested without a GPU.
- `uniform_bsplines` is the only hand-rolled numerics left, and exists solely because a
  `BSplineKit.Spline` cannot enter a GPU kernel — **not** from distrust: ForwardDiff through
  BSplineKit is exact to 1e-14, and our evaluator is checked against it to 1.3e-15.
