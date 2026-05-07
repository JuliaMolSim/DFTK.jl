# TRS k-point reduction: implementation progress

Read `plan.md` first for full context and design decisions.

---

## Status: Steps 1–4 complete. Next: phonons/chi0 with q≠0 (step 4b), then polish.

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
- `symmetry_operations` for `:collinear`: now returns ALL rows from spglib's `get_symmetry_with_site_tensors`, including `spin_flips == -1` as θ=−1 SymOps (was previously discarded).
- `symmetries_preserving_kgrid` (both overloads): uses `symop.θ * symop.S * k` instead of `symop.S * k`.
- `unfold_kcoords`: uses `symop.θ * symop.S * kcoord`.
- `unfold_mapping`: uses `symop.θ * symop.S * kpt_irred.coordinate`.
- `symmetrize_ρ`: split into two paths:
  - `n_spin == 1` or no antiunitary symops: uniform loop over all symmetries (θ=−1 on real ρ is a no-op, just redundant work).
  - `n_spin == 2` with antiunitary symops: θ=+1 symops accumulate same spin; θ=−1 symops accumulate from opposite spin (↑↔↓ swap).
- `apply_symop` for wavefunctions: handles θ=−1 — uses `-invS * G_full` and `conj(ψk[igired])` with phase `e^{+iGτ}`.

**`src/Model.jl`**
- `default_symmetries`: after calling `symmetry_operations`, if `spin_polarization ∈ (:none, :spinless)` and `!any(breaks_TRS, terms)`, appends θ=−1 duplicates of all θ=+1 symops.

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

### Step 4b: phonons/chi0 with q≠0

After TRS k-reduction, `k+q` may only be present in `basis.kpoints` as `−(k+q)`.
`find_equivalent_kpt` will crash in that case (returning `nothing` then `nothing + int`).

**Rejected approach**: teaching `find_equivalent_kpt` to try `−kcoord` as a fallback.
This makes a low-level lookup function aware of physics it shouldn't know about, and
silently changes the return contract of `get_kpoint` and `k_to_kpq_permutation`.

**Correct approach**: treat TRS like any other symmetry. Rewrite the inner loop of
`transfer_blochwave_equivalent_to_actual` to search `basis.symmetries` (including
θ=−1 entries) for a symop mapping some irreducible k-point to `k+q`, then call
`apply_symop` to get ψ at `k+q`. This is the same pattern used in `unfold_bz`.

### Step 5: Polish items

**`src/postprocess/current.jl`**:
- Currents are TRS-odd: `j(x) → -j(x)` under θ=−1.
- The symmetrisation loop needs a sign flip for θ=−1 symops. Without this, equilibrium currents symmetrize to zero accidentally (correct), but response currents get killed incorrectly.
- Find the accumulation loop and add: `factor = symop.θ` (or just `-1`) when applying θ=−1 operations.

**`src/terms/hubbard.jl`** — `symmetrize_hubbard_n`:
- For θ=−1: add `conj` on the Wigner-D rotation and (for collinear) swap the σ index.
- Low priority unless Hubbard+TRS is tested.

**GPU kernel** (`src/symmetry.jl:301`, `accumulate_over_symmetries!`):
- Currently passes `symm_invS` and `symm_τ` to GPU kernel.
- For collinear with θ=−1, the kernel needs `symm_θ` and spin-swap logic.
- The n_spin=1 path is already correct (θ=−1 is no-op for real ρ on GPU too).
- Only needed for GPU collinear AFM use case.

**Serialization** (`ext/DFTKJLD2Ext.jl`, `ext/DFTKJSON3Ext.jl`):
- Grep for anywhere that reconstructs a `SymOp` from saved data. If SymOp is stored field-by-field, add `θ` (default 1 for backward compat).
- Confirmed: current JLD2 ext does NOT serialize SymOp objects directly — it stores `symmetries_respect_rgrid` and `use_symmetries_for_kpoint_reduction` flags. So likely no change needed.

**Wannier90 ext** (`ext/DFTKWannier90Ext.jl`):
- Check if it generates a k-point list that needs TRS-aware unfolding. Low priority.

### Step 6: Tests

Add to `test/bzmesh_symmetry.jl`:
1. SymOp group-theory unit test with antiunitaries.
2. Triple comparison (nosym / sym-no-TRS / sym-TRS) for inversion-asymmetric system — all agree on ρ, E, forces.
3. Collinear AFM equivalence — confirm symmetry count doubles and ρ unchanged.
4. k-weight sanity: `sum(basis.kweights) ≈ 1` for each variant.

### Step 7: Performance

- In `symmetrize_ρ` for n_spin=1: currently iterates over ALL 2N symops (θ=−1 is redundant for real ρ). Can filter to θ=+1 only and divide by N instead of 2N. Negligible correctness impact, saves ~2× work in symmetrize_ρ.

---

## Key invariants to preserve

- `sum(basis.kweights) ≈ 1` always.
- Energies with TRS must match `symmetries=false` to SCF tolerance.
- `check_group` passes for the augmented symmetry group (it currently does — `*` and `isapprox` carry θ).
- The `SYMMETRY_CHECK = true` path: `_check_symmetries` checks spatial atom mapping (W, w) only, unaffected by θ. `_check_kpoint_reduction` already updated to use θ·S·k.
