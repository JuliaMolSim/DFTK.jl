# TRS k-point reduction: implementation progress

Read `plan.md` first for full context and design decisions.

---

## Status: Steps 1–5 complete. Next: tests (step 6), then performance (step 7).

Run tests with: `julia test_trs.jl` (no `--project` flag — there is a pre-existing PseudoPotentialIO/Psp8File version mismatch that breaks compilation under `--project`; running without it uses the system-installed DFTK environment which works).

---

## What is done

### Files modified

**`src/SymOp.jl`**
- Added `θ::Int` field to `SymOp{T}` struct (after `τ`). θ ∈ {+1, -1}: +1 unitary, -1 antiunitary.
- `SymOp(W, w; θ=1)` constructor: keyword arg, accepts `Cint` from spglib, stores `Int(θ)`.
- `convert`: includes θ.
- `==`, `isapprox`: include θ comparison.
- `isone`: requires θ == 1.
- `*`: θ multiplies (θ₁·θ₂).
- `inv`: θ preserved (antiunitary inverse is antiunitary).

**`src/terms/terms.jl`**
- Added `breaks_TRS(::Any) = false`.
- Added `breaks_TRS(::Magnetic) = true`.
- Added `breaks_TRS(::Anyonic) = true`.

**`src/symmetry.jl`**
- `symmetry_operations`: new `time_reversal::Bool=true` keyword. For `spin_polarization == :none`, when set, augments the returned list with θ=−1 antiunitary partners.
- `symmetry_operations` for `:collinear`: returns ALL rows from spglib's `get_symmetry_with_site_tensors`, including `spin_flips == -1` as θ=−1 SymOps (was previously discarded).
- `symmetries_preserving_kgrid` (both overloads): uses `symop.θ * symop.S * k`.
- `unfold_kcoords`, `unfold_mapping`: use `symop.θ * symop.S * k`.
- `accumulate_over_symmetries!`: now operates on the full 4D ρ (Nx, Ny, Nz, n_spin) and handles θ-dependent spin source: for θ=−1 with n_spin==2 the source spin is swapped (↑↔↓); for n_spin==1 (real ρ) θ acts trivially.
- `symmetrize_ρ`: simplified — single call into `accumulate_over_symmetries!` with the whole ρ; lowpass loops over spin.
- `apply_symop` for wavefunctions: unified loop using θ — index by `θ·invS·G`, phase `cis2pi(-θ·G·τ)`, conjugate when θ=−1.

**`src/Model.jl`**
- `default_symmetries`: passes `time_reversal = spin_polarization ∈ (:none, :spinless) && !any(breaks_TRS, terms)` to `symmetry_operations`. The augmentation itself lives in `symmetry_operations`.

**`src/bzmesh.jl`**
- `irreducible_kcoords`: `is_time_reversal = any(s -> s.θ == -1, symmetries)` — flips the spglib flag when the symmetry group contains antiunitary operations. Removes old TODO comment.
- `_check_kpoint_reduction`: uses `symop.θ * (symop.S * k)`.

### Test result (from `test_trs.jl`)
- Si at `kgrid=[4,4,4]`: nosym=64 k-points, TRS=8 k-points (was ~28 with spatial symmetry only, now ~4× reduction from full TRS exploitation).
- k-weights sum to 1.0.
- Energy matches no-symmetry run to 0.0 difference.

---

## What is NOT done (remaining steps from plan)

### Step 4: verify forces/stresses ✓ DONE

Forces use `q=0`. `transfer_blochwave_equivalent_to_actual` short-circuits immediately
for `iszero(q)` — `find_equivalent_kpt` is never reached. Forces work correctly.

Tested (`test_trs.jl`):
- GaAs zinc-blende [4,4,4]: 23 spatial (Td, no inversion) + 24 TRS → 8 k-points.
  ΔE=3.6e-15, max|Δf|=4e-12, max|Δρ|=4.4e-12. Forces zero (equilibrium).
- Rattled GaAs [4,4,4]: 0 spatial + 1 TRS → 36 k-points (64→36; TRIM points
  are TRS-invariant so not halved: (64+8)/2=36). Forces non-zero (0.033 Ha/Bohr).
  ΔE=1.8e-15, max|Δf|=2.6e-11, max|Δρ|=6.2e-12. All machine precision.

Note on TRIM counting: for a [N,N,N] grid there are 2³=8 time-reversal-invariant
k-points (where 2k≡0 mod lattice). The irreducible count with TRS only is (N³+8)/2.

### Step 4b: phonons/chi0 with q≠0 — NOT NEEDED

Phonon tests always call with `symmetries=false` (full BZ). The phonon derivation
requires TRS implicitly and is only correct on unfolded systems anyway. No change
needed — the existing convention of passing `symmetries=false` for phonons is sufficient.

### Step 5: Polish items ✓ DONE

**`src/postprocess/current.jl`**: assertion left as `length(basis.symmetries) == 1` (per
review). Currents still require full BZ; a proper TRS-aware symmetrization (with θ-odd
sign flip) is a future TODO.

**`src/symmetry.jl` — `symmetrize_hubbard_n`**: restricted to θ=+1 symops only.
For n_spin=1 (real n) this is equivalent to before. For n_spin=2 with TRS it now
gives correct results (previously would silently average wrong spin components).
Full antiunitary treatment (conj(WigD) + spin swap) left as a future TODO.

**GPU kernel** (`accumulate_over_symmetries!`): no change needed. The kernel formula
`e^{-iG·τ} ρ(S⁻¹G)` is correct for θ=−1 on real densities; spin swap for collinear
is already handled outside the kernel in `symmetrize_ρ`.

**Serialization**: no change needed — extensions store flags, not SymOp objects.

**Wannier90 ext**: not checked, low priority.

### Step 6: Tests (TODO)

Move `test_trs.jl` content into `test/bzmesh_symmetry.jl` as proper `@testitem` entries:
1. SymOp group-theory unit tests (antiunitaries): `*`, `inv`, `isone`. Tag `:minimal`.
2. GaAs equilibrium (Td+TRS): E/ρ/f agreement, k-weight sanity. Tag default.
3. Rattled GaAs (TRS-only, non-zero forces): E/ρ/f agreement, k-count=36. Tag default.
4. Collinear AFM equivalence (spglib spin_flips path): symmetry count doubles,
   ρ unchanged. Tag default.

### Step 7: Performance (TODO)

- `symmetrize_ρ` for n_spin=1: iterates over 2N symops but θ=−1 is redundant for
  real ρ. Can pass only θ=+1 symops (N) and divide by N. Saves ~2× work.

---

## Key invariants to preserve

- `sum(basis.kweights) ≈ 1` always.
- Energies with TRS must match `symmetries=false` to SCF tolerance.
- `check_group` passes for the augmented symmetry group (it currently does — `*` and `isapprox` carry θ).
- The `SYMMETRY_CHECK = true` path: `_check_symmetries` checks spatial atom mapping (W, w) only, unaffected by θ. `_check_kpoint_reduction` already updated to use θ·S·k.
