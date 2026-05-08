# TRS k-point reduction: status

Read `plan.md` for the design. This file summarises what shipped and what's left.

---

## Shipped

### Core data structure — `src/SymOp.jl`
- New `θ::Int ∈ {+1, -1}` field on `SymOp`. Default `θ=+1` everywhere.
- `==`, `isapprox`, `isone`, `*`, `inv`, `convert` all carry θ. Composition
  multiplies θ; inverse preserves it (antiunitary inverse is antiunitary).
- `check_group` / `complete_symop_group` work unchanged on the augmented group.

### Symmetry detection — `src/symmetry.jl`, `src/Model.jl`
- `symmetry_operations(...; time_reversal=true)` now augments the group with
  antiunitary partners for `spin_polarization == :none` when the flag is set.
- For `:collinear`, the spglib `spin_flips==-1` rows that were previously
  *discarded* (with a TODO comment) are now kept and tagged θ=-1 — this is the
  AFM 2× speedup.
- `default_symmetries` (in `Model.jl`) passes `time_reversal = (spin ∈ :none/:spinless)
  && !any(breaks_TRS, terms)`.

### k-point reduction — `src/bzmesh.jl`
- `irreducible_kcoords` flips spglib's `is_time_reversal` to `any(s -> s.θ == -1, symmetries)`.
  This is the moment GaAs / h-BN / non-centrosymmetric lattices halve their k-count.
- `_check_kpoint_reduction` checks `kred ≡ θ·S·k`.

### k-action and consumers — `src/symmetry.jl`
- `symmetries_preserving_kgrid` (both overloads), `unfold_kcoords`, `unfold_mapping`:
  use `θ·S·k`.
- `unfold_mapping` also handles the ↑↔↓ spin swap induced by θ=-1 in collinear.
- `apply_symop` (wavefunctions): one unified loop — index by `θ·invS·G`, phase
  `cis2pi(-θ·G·τ)`, `conj` the source coefficient when θ=-1.
- `accumulate_over_symmetries!` now operates on the full 4D ρ (Nx, Ny, Nz, n_spin)
  and handles the θ-dependent source-spin selection inside the kernel
  (collinear ↑↔↓ swap; trivial on real n_spin==1 density). Single GPU launch
  per spin component as before.
- `symmetrize_ρ`: simplified to a single accumulate call. Fast path filters to
  θ=+1 symops up front when `n_spin == 1` (real ρ), halving the per-iteration work.

### Polish
- `src/postprocess/current.jl`: assert kept as `length(basis.symmetries) == 1`
  pending TRS-aware sign flip — currents are TRS-odd.
- `symmetrize_hubbard_n`: now handles θ=-1 partners — the contribution is
  `WigD' · conj(n_src) · WigD`, with `n_src` drawn from the opposite spin
  channel for collinear (`σ_src = 3 - σ` when n_spin == 2).
- GPU kernel: `symm_θ` device array added; spin-swap branch is GPU-friendly.
- `_check_symmetries`, serialization extensions (JLD2/JSON3), Wannier ext: no
  change needed (they don't materialise SymOp objects).
- Docs (`docs/src/developer/symmetries.md`): theory section now includes
  the antiunitary action and the `θ·S·k` Brillouin-zone action.

### Tests — `test/bzmesh_symmetry.jl`
- SymOp group-theory units (`*`, `inv`, `isone`, `check_group`) — `:minimal`.
- GaAs equilibrium (Td + TRS): E/ρ/forces agreement, k-weight = 1, k-count drops.
- Rattled GaAs (TRS-only): k-count = 36 on `[4,4,4]` (TRIM points unhalved),
  non-zero forces, agreement with no-symmetry run.
- `unfold_bz` round-trip on GaAs: exercises `unfold_mapping` + `apply_symop`
  on θ=-1 partners.
- Collinear AFM (Si with `magnetic_moments=[1,-1]`): asserts θ=-1 partners
  present, group closes.

Other touched tests:
- `test/symmetry_issues.jl` (CuO2 group): expected count updated 48 → 96 (×2 TRS).
- `test/bzmesh.jl` (irreducible k reduction): reconstruction loop uses `θ·S·k`.

### Defensive
- `find_equivalent_kpt`: clean error message instead of the implicit
  `nothing + Int` crash, pointing users to `unfold_bz` /
  `use_symmetries_for_kpoint_reduction=false` when they hit a missing k+q.

---

## What is left

### `transfer_blochwave_equivalent_to_actual` for q ≠ 0
Plan called for replacing the inner loop with a symop-based lookup so chi0 /
phonon paths can run on TRS-reduced bases. Not done — phonon tests already pass
`symmetries=false` and chi0 with q ≠ 0 is exercised on unfolded bases. The
defensive error in `find_equivalent_kpt` makes the misuse fail cleanly.

### `compute_current` TRS symmetrisation
Currently asserts `length(basis.symmetries) == 1`. Lifting this needs a
θ-aware sign flip in the symmetrisation loop (currents are TRS-odd).

### Collinear AFM and the spglib k-orbit
For non-magnetic systems the θ=-1 partners share W with a θ=+1 partner (we
synthesise them as exact duplicates), and spglib's `is_time_reversal=true`
flag pairs each rotation with T — exactly the right group. For collinear AFM
the θ=-1 partners come from spglib's `spin_flips==-1` rows, where W is
typically *not* in the θ=+1 set: only `W·T` is a magnetic-group element, not
W alone. Naively passing every rotation to spglib + `is_time_reversal=true`
would tell it that `W` *and* `W·T` are both symmetries, over-reducing the BZ.

`irreducible_kcoords` therefore passes only θ=+1 rotations to spglib and sets
`is_time_reversal=true` only when the θ=-1 rotation set is contained in the
θ=+1 set (i.e., the synthesis case). For AFM, k-orbit reduction falls back to
the unitary part — this matches pre-PR behaviour, since for the most common
AFM cases the antiunitary k-action coincides with the unitary one (e.g., AFM
Si: `θ=-1 = T_d·(-I)`, whose k-action equals the T_d-orbit). Cases where the
antiunitary part adds genuinely new BZ-orbit elements would need a custom
orbit reduction (spglib's API doesn't accept a magnetic point group); not
implemented.

### Tests
- An SCF-level AFM equivalence test (sym vs no-sym energies/densities) would
  give us a real correctness signal for the path above. The current AFM test
  only checks group closure and k-weight sum.

---

## Key invariants

- `sum(basis.kweights) ≈ basis.model.n_spin_components` (1 for non-collinear, 2
  for collinear).
- Energies / densities / forces with TRS match `symmetries=false` to SCF
  tolerance on GaAs (equilibrium and rattled).
- `check_group` passes for the augmented group (`*`, `inv`, `isapprox` carry θ).
- For non-magnetic systems, `length(model.symmetries)` is exactly 2× the
  pre-TRS count.
