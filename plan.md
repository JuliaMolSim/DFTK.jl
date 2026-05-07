# Plan: time-reversal symmetry for k-point reduction

Tracking: [#224](https://github.com/JuliaMolSim/DFTK.jl/issues/224). Related: [#203](https://github.com/JuliaMolSim/DFTK.jl/issues/203). Depends on [#1316](https://github.com/JuliaMolSim/DFTK.jl/pull/1316) (the `breaks_TRS` predicate).

Goal: halve the irreducible k-point count in inversion-asymmetric crystals (GaAs,
h-BN, ZnO…) and unlock a separate 2× speedup for collinear antiferromagnets, by
treating time-reversal as a regular antiunitary symmetry of the single-particle
Hamiltonian.

Out of scope: exploiting TRS at high-symmetry points (e.g. Γ) to reduce real DOFs in ψ
itself; spin-orbit / `:full` polarisation; phonons under broken TRS (Dal Corso 2019).

---

## Two free wins to start from

The codebase is much closer to this feature than it looks:

1. **`bzmesh.jl:61` already has the TODO**:
   ```julia
   # TODO implement time-reversal symmetry and turn the flag below to true
   is_shift = ...
   spg_mesh = Spglib.get_stabilized_reciprocal_mesh(rotations, kgrid.kgrid_size, qpoints;
                                                    is_shift, is_time_reversal=false)
   ```
   Spglib's k-mesh reducer handles TRS natively. Once SymOp carries θ, this flag is
   just `is_time_reversal = !any(breaks_TRS, basis.model.term_types)` (gated on spin
   sector — see below). No `−W` workaround, no manual k-orbit code.

2. **`symmetry.jl:93–96` already calls the right spglib API and throws away the
   answer**:
   ```julia
   rotations, translations, spin_flips = Spglib.get_symmetry_with_site_tensors(cell, tol_symmetry)
   rotations[spin_flips.==1], translations[spin_flips.==1]    # discards spin_flips==-1!
   ```
   The comment immediately below explicitly notes the discarded rows would give a
   2× speedup for AFM order. Spglib's `spin_flips ∈ {±1}` is *exactly* our θ. Stop
   dropping them; tag the `−1` rows as `θ=−1` SymOps.

So the actual project is: extend SymOp with a θ field, stop discarding spglib's
output, flip the bzmesh flag, and audit the wavefunction-touching code that now
needs to apply complex conjugation when consuming a θ=−1 partner.

---

## Design

Extend `SymOp` from `(W, w)` to `(W, w, θ)` with `θ ∈ {+1, −1}`. The action of an
antiunitary (`θ = −1`) on a wavefunction is

```
(Uu)(x) = conj(u(W·x + w))             (real space)
(Uu)(G) = e^{+iG·τ} conj(u(−S^{-1}·G))  (reciprocal space)
```

with `S = Wᵀ`, `τ = −W^{-1}·w` as today. The action on a k-point is `θ · S · k`.
`model.symmetries` continues to be the full group; `basis.symmetries` continues to
be the subgroup preserving the kgrid.

### Source of θ=−1 partners by spin sector

| `spin_polarization` | Where partners come from | Action on ρ |
|---|---|---|
| `:none`, `:spinless` | synthesise: duplicate every `(W,w,+1)` to `(W,w,−1)` | identity (ρ is real) |
| `:collinear` | keep spglib's `spin_flips==−1` rows as `θ=−1` | swaps `ρ_↑ ↔ ρ_↓` *and* applies `(W,w)` |
| `:full` (SOC) | out of scope | `iσ_y K` on the spinor |

In all cases, gate on `!any(breaks_TRS, terms)`. For `:collinear` with a net moment,
spglib won't return spin-flip rows in the first place (the magnetic point group
doesn't contain them), so no extra gating needed.

### Group composition

A bit of case analysis: `(U₁ U₂ u)(x)` works out to `(W₁W₂, w₁ + W₁w₂, θ₁·θ₂)` in
all four sign combinations. So composition is "spatial parts compose normally, θ
multiplies". Inverse: `(W^{-1}, −W^{-1}·w, θ)` (θ unchanged — `inv` of antiunitary
is antiunitary). Verify both with unit tests on small fabricated groups.

---

## File-by-file changes

### `src/SymOp.jl`

- Add `θ::Int` field. Default `θ=+1` in the convenience constructor `SymOp(W, w)`.
- `==`, `isapprox`, `one`, `isone`, `inv`: include θ.
- `*`: `(W₁W₂, w₁ + W₁w₂, θ₁·θ₂)` — case analysis above.
- `complete_symop_group` / `check_group`: should just work once `*`/`inv`/`isapprox`
  carry θ. Add a regression test on a group containing antiunitaries.
- Helper `apply_to_kpoint(symop, k) = symop.θ * symop.S * k`.

### `src/symmetry.jl`

- Lines 90–103 (the `spin_polarization` switch in `symmetry_operations`):
  - `:none` / `:spinless`: keep current `Spglib.get_symmetry` call, then duplicate
    each returned `(W, w, +1)` to `(W, w, −1)` *iff* `!any(breaks_TRS, terms)`.
    (Caller will pass `terms` through.)
  - `:collinear`: replace the discard
    `rotations[spin_flips.==1], translations[spin_flips.==1]`
    with returning *all* rows along with the `spin_flips` array, and build
    `SymOp(W, w; θ=spin_flip)` for each. Keep the existing AFM-comment intent;
    delete the "this would cut runtime by 2×" sentence (we're delivering it).
- `_check_symmetries`: spatial check `W·a + w ≡ a' (mod 1)` is unchanged for θ=±1.
  Spglib has already verified the moment-flipping consistency for `:collinear`. No
  code change expected — but verify with a test that the assertions still pass for
  AFM systems.
- `symmetries_preserving_kgrid` (lines 162, 175, 184): `symop.S * k` →
  `symop.θ * symop.S * k`.
- `symmetrize_ρ`, `accumulate_over_symmetries!`, `lowpass_for_symmetry!`:
  - `n_spin == 1` (`:none`/`:spinless`): θ=−1 is a no-op on real ρ — can skip
    those iterations as a fast path. Without the optimisation, the loop is just
    redundant work but still correct.
  - `n_spin == 2` (`:collinear`): θ=−1 partners must swap the spin index of the
    source density before accumulating into the target. Branch on θ inside the
    accumulate loop.
- `unfold_bz` (line 459): rebuilds the full k-mesh from the irreducible one.
  Needs to also unfold via θ=−1 partners — i.e., for each θ=−1 symop, the
  unfolded k is `−Sk` and the corresponding ψ at that k is `conj` of ψ at the
  irreducible k (with G-flip in reciprocal space).

### `src/bzmesh.jl`

- Line 69: change `is_time_reversal=false` to a value derived from
  `!any(breaks_TRS, model.term_types)` and the spin sector. This is the
  one-line core of the speedup.
- Line 294 (`symop.S * k` equivalence test inside `irreducible_kcoords`): use
  `symop.θ * symop.S * k`.
- Remove the `# TODO implement time-reversal symmetry...` comment once done.

### `src/transfer.jl` *(highest-risk file)*

- `find_equivalent_kpt` (line 180) currently looks for `kcoord` modulo a
  reciprocal lattice vector only. After TRS reduction, callers asking for `k+q`
  may find only `−(k+q)` is in `basis.kpoints`. Extend the signature to also try
  `−kcoord` and return a `θ` (or `needs_conj::Bool`) alongside `index, ΔG`.
  Update both callers (line 115 of `PlaneWaveBasis.jl`, and
  `k_to_kpq_permutation` on line 200 of `transfer.jl`).
- `transfer_blochwave_kpt`, `transfer_blochwave_equivalent_to_actual`: when the
  source/target k-mapping is via an antiunitary, the coefficient transfer needs
  `conj` and a sign flip on G-indices. Audit all uses; in the phonon flow
  (`compute_δρ` at `densities.jl:83`), this is what determines whether
  `δψ_plus_k` is correctly assembled when k+q is only equivalent to −(k+q) in
  the irreducible mesh.

### `src/response/chi0.jl`, `src/response/hessian.jl`

- The Sternheimer / response solvers iterate over irreducible k. Any orbit-completion
  step that reconstructs full-BZ ψ-level quantities from irreducible ones must apply
  `conj` for θ=−1 partners. Density-level accumulation is fine because it goes
  through `compute_δρ` → `symmetrize_ρ` which we've already fixed.

### `src/postprocess/current.jl`

Currents are TRS-odd: `j(x) → −j(x)` under θ=−1. Symmetrising over a
TRS-augmented group must include this sign flip — otherwise equilibrium currents
in non-magnetic systems get symmetrised to zero (correct by accident) while
field-induced response currents get incorrectly killed. Add a θ-aware sign in
the symmetrisation loop.

### `src/terms/hubbard.jl`

Hubbard occupation matrices `n^{Iσ}_{mm'}` transform as `n → S·n·S†` under unitary
symmetries (in the spherical-harmonic basis). For θ=−1 add a `conj` on the
matrix and (for collinear) swap the σ index. Audit + add a regression test on
a Hubbard system without inversion.

### `src/postprocess/phonon.jl`

Already guarded by the `!breaks_TRS` assertion landed in #1316. No change. The
TRS k-reduction *helps* phonons in the unbroken case for free.

### `src/external/spglib.jl`, `src/external/DFTKCalculator.jl`, `src/workarounds/forwarddiff_rules.jl`

Grep for `SymOp(`. Anywhere a `SymOp` is constructed by hand passes `θ=+1` by
default. No semantic change.

### Extensions: `ext/DFTKJLD2Ext`, `ext/DFTKJSON3Ext`, `ext/DFTKWannier90Ext`, `ext/DFTKPlotsExt`

- JLD2 / JSON3: serialised SymOp gains a field. Bump the on-disk version tag and
  default-fill `θ=+1` when reading old files.
- Wannier90: it cares about k-point lists and full-BZ unfolding; check the
  interface still produces a sensible reducible k-set after TRS reduction.
- Plots: band-structure k-paths are user-specified, not affected. But if any
  helper auto-generates a path through the irreducible BZ, double-check.

### GPU kernel: `src/symmetry.jl:301`

`accumulate_over_symmetries!` pushes `symop.S` and `symop.τ` to device arrays.
Add `symop.θ` to the device array and special-case in the kernel (no-op for
θ=+1, swap-spin / sign-flip / conj as appropriate for θ=−1).

---

## Spin gotchas (read carefully)

- **`:collinear`, θ=−1**: action is "spatial `(W, w)` *plus* spin swap `↑ ↔ ↓`".
  The current `symmetrize_ρ` iterates over the spin axis independently per symop,
  which is wrong for θ=−1. Branch on θ inside the accumulate loop.
- **AFM with paired moments** is the headline win here, not just an edge case:
  spglib already detects the spin-flipping symmetries; we just stopped throwing
  them away. The existing comment at `symmetry.jl:97–102` documents this exact
  optimisation as future work.
- **Net magnetic moment**: spglib's magnetic point group already excludes the
  spin-flip rows. No extra check needed; `breaks_TRS` (on `Magnetic` term, not
  on initial moments) is the user-side gate.
- **`:full` (SOC)**: TRS acts as `iσ_y K` on the spinor. Skip — DFTK has limited
  SOC support anyway. Hard-assert `θ=+1` when `spin_polarization == :full`.

---

## Other gotchas

- **k-weights**: irreducible k-weights must reflect the orbit size *under the
  full augmented group*. Easy off-by-2× if you flip the spglib flag but forget
  that `basis.symmetries` doubled. Spglib returns the correct mapping table
  (`ir_mapping_table`); use it. Sanity: `sum(basis.kweights) ≈ 1`.
- **`find_equivalent_kpt` is the single most error-prone site** because it
  silently returns a wrong index instead of failing if `−k` is what it needs.
  Extend the signature; don't add a `try`.
- **Test data & ψ comparisons**: irreducible meshes shrink, so per-k indices
  don't line up across symmetry modes. Tests must compare derived quantities
  (ρ, E, forces, eigenvalues at fixed external k-paths), never raw ψ.
- **Performance regression for non-TRS cases**: don't accidentally make
  symmetrise_ρ slower when no antiunitaries are present. Benchmark Si (no TRS
  benefit) before/after.
- **MPI**: halving k-points may leave ranks idle on small jobs. Not a correctness
  issue; document it.
- **Consistency with the phonon δρ derivation** (in `compute_δρ` comment block,
  if PR #1316 has landed): both that derivation and this k-reduction *use* TRS.
  No double-counting, but verify by running phonon tests with TRS-reduced k-grids.
- **`SYMMETRY_CHECK = true`** (`SymOp.jl:24`): assertions on group closure must
  accept antiunitaries. Should be automatic once `*` and `isapprox` carry θ.

---

## Test strategy

The existing pattern in `test/bzmesh_symmetry.jl` — SCF with `symmetries=false`
versus default symmetries on the same system, asserting ρ and E agree — is
exactly what we extend. **Add the new TRS tests in that file.**

Required new tests:

1. **SymOp group-theory unit tests**. Small fabricated group containing
   antiunitaries: identity, closure under `*`, inverse, associativity. No SCF.

2. **Triple-comparison equivalence** on inversion-asymmetric systems:
   `(symmetries=false)` vs `(symmetries on, no TRS)` vs `(symmetries on, TRS on)`
   — all three must agree on ρ, E, forces, stresses to SCF tolerance, and the
   third should have ~half the k-count of the second.
   - **GaAs** (zincblende, no inversion) — primary case.
   - **h-BN monolayer** (small, polar, non-centrosymmetric) — secondary.

3. **Collinear AFM equivalence** — exercises spglib's `spin_flips==−1` path that
   DFTK currently discards. A 2-atom AFM toy with paired up/down moments suffices
   (e.g. simple-cubic Cr or extend an existing collinear `testcases.jl` setup).
   Confirm post-TRS symmetry count is exactly 2× the pre-TRS count and ρ unchanged.

4. **k-weight sanity**: `sum(basis.kweights) ≈ 1` for every variant above.

5. **Currents** (TRS-odd, easy to silently break):
   - On a TRS-symmetric ground state with TRS-augmented symmetries, current
     symmetrises to ~zero. (It would anyway in equilibrium, but this catches sign
     errors.)
   - On a state with a `Magnetic` term (so `breaks_TRS` is true and TRS partners
     are absent), current is *not* symmetrised to zero.

6. **Phonon non-regression**: existing phonon tests must keep passing. Phonons
   under unbroken TRS should silently benefit from k-reduction.

7. **Forces / stresses** on GaAs via the existing
   `test/run_scf_and_compare.jl`-style helpers.

8. **Serialisation round-trip**: `save_scfres → load_scfres` preserves θ. One-line
   check in the JLD2 / JSON3 test files.

Tags: group-theory units + k-weight sanity + serialisation → `:minimal`. GaAs /
h-BN / AFM / currents → default. Heavy combinations → `:slow`.

What *not* to test: raw ψ at a specific k-index across symmetry modes — irreducible
meshes differ.

---

## Implementation order

1. **SymOp + group-theory unit tests** (½–1 day): θ field, composition, inverse,
   equality. No behaviour change beyond the field existing.
2. **`default_symmetries` wiring** (½ day): stop discarding `spin_flips==−1`;
   synthesise `:none`/`:spinless` partners. Gate on `breaks_TRS`. Full test suite
   should still pass — k-reduction isn't using TRS yet, so this is a no-op for
   actual k-counts but exercises the new SymOp paths in symmetrise_ρ etc.
3. **Flip the spglib flag** (½ day): `is_time_reversal=true` in `bzmesh.jl`
   when allowed. *This is the moment k-counts halve* on GaAs. Smell test:
   `length(basis.kpoints)` for GaAs at `kgrid=[4,4,4]` drops by ~2×.
4. **`transfer.jl` audit** (2–3 days): the long tail. Extend
   `find_equivalent_kpt`, fix `transfer_blochwave_*` callers, run forces/stresses
   regression on GaAs.
5. **Currents + Hubbard + GPU kernel + extensions** (1–2 days): polish.
6. **Performance pass + θ=+1 fast path in symmetrise_ρ** (½ day).

Total: ~1.5 weeks for someone familiar with the symmetry code. Step 4 is the
unknown — could expand if there are subtler ψ-level uses than the obvious ones.

**Smell-test trip-wires** along the way:
- After step 2: full test suite green; `length(basis.symmetries)` for GaAs is
  exactly 2× what it was.
- After step 3: GaAs at `kgrid=[4,4,4]` has roughly half the k-points; SCF energy
  matches the no-symmetry run to 1e-10.
- After step 4: forces and stresses on GaAs match the no-symmetry run to SCF
  tolerance.

---

## Hand-off context

**Read first**: `docs/src/developer/symmetries.md`, `src/SymOp.jl`,
`src/symmetry.jl`, `src/bzmesh.jl`, `src/transfer.jl`, this plan, the discussion
in #224 and #1316.

**Assumptions**:
- PR #1316 has landed (or is cherry-picked). It provides `breaks_TRS(::Any) = false`
  and overloads on `Magnetic`, `Anyonic`, plus the `compute_dynmat` assertion that
  no term breaks TRS.
- The phonon `δρ` comment block in `src/densities.jl` (also from #1316) describes
  the *other* TRS-dependent trick. This project is consistent with it; both rely on
  the same `breaks_TRS` predicate.

**Conventions**:
- `SymOp` is constructed in many places via `SymOp(W, w)`. Default `θ=+1` in that
  constructor; don't reorder fields.
- Spglib magnetic / dataset APIs are accessed through the existing `Spglib` import
  in `src/external/spglib.jl`. Don't `using Spglib` from new files.
- GPU/Wannier/JLD2 etc. live in `ext/`. Don't import them from `src/`.

**What does *not* change**: SCF driver, mixing, eigensolvers, Hamiltonian assembly,
real-space density representation, `breaks_symmetries` predicate, model construction.

---

## References

- DFTK developer docs: `docs/src/developer/symmetries.md`.
- Issue threads: [#224](https://github.com/JuliaMolSim/DFTK.jl/issues/224), [#203](https://github.com/JuliaMolSim/DFTK.jl/issues/203), [#1316](https://github.com/JuliaMolSim/DFTK.jl/pull/1316).
- Quantum ESPRESSO `Modules/symm_base.f90` — sanity-check on conventions (their
  `t_rev` is our θ).
- Dal Corso, *Density-functional perturbation theory within the projector
  augmented wave method*, https://arxiv.org/abs/1906.11673 — for the
  broken-TRS phonon case (out of scope here).
