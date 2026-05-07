# Plan: implement time-reversal symmetry for k-point reduction

Tracking issue: [JuliaMolSim/DFTK.jl#224](https://github.com/JuliaMolSim/DFTK.jl/issues/224).
Related: discussion thread on [#203](https://github.com/JuliaMolSim/DFTK.jl/issues/203).

The goal is to halve the irreducible-BZ k-point count in inversion-asymmetric crystals
(GaAs, ZnO, h-BN…) by treating time-reversal as an additional symmetry of the
single-particle Hamiltonian. This is *not* about exploiting TRS at high-symmetry points
(e.g. Γ) to reduce real DOFs in the wavefunctions — that is a separate, harder problem.

A `breaks_TRS` predicate on terms already exists (PR #1316). When `!breaks_TRS(any_term)`
and the spin sector permits, this work synthesises antiunitary partners of every
spglib-detected symmetry and adds them to `model.symmetries`.

---

## Design (from antoine-levitt's sketch in #224)

Extend `SymOp` from `(W, w)` to `(W, w, θ)` with `θ ∈ {+1, −1}`. The action of an antiunitary
(`θ = −1`) on a wavefunction is

```
(Uu)(x) = conj(u(W·x + w))           (real space)
(Uu)(G) = e^{+iG·τ} conj(u(−S^{-1}·G))   (reciprocal space)
```

with `S = Wᵀ`, `τ = −W^{-1}·w` as today. The action on a k-point is `θ · S · k`.
`model.symmetries` continues to be the full group (now possibly antiunitary-augmented),
and `basis.symmetries` continues to be the subgroup preserving the kgrid.

---

## File-by-file change list

### `src/SymOp.jl` — core data type

- Add field `θ::Int` (or a `Bool`/`UnitaryKind` enum). Default constructor sets `θ = +1`.
- `Base.:(==)`, `Base.isapprox`: include `θ`.
- `Base.one(::Type{SymOp{T}})`: `θ = +1`.
- `Base.isone`: require `θ = +1`.
- **Composition `*`**:
  - `θ_out = θ₁ · θ₂`
  - `W_out = W₁ · W₂` (always — the spatial part composes the same way)
  - `w_out = w₁ + W₁ · w₂` if `θ₁ = +1`, else `w₁ + W₁ · (−w₂)`? **No** — work this out carefully.
    The product of antiunitaries `U₁ U₂` acting on `u`: `(U₁ U₂ u)(x) = conj((U₂u)(W₁x + w₁))`
    if `θ₁=−1`. Substitute `(U₂u)(y) = conj(u(W₂y+w₂))` if `θ₂=−1`; else without conj.
    Cases:
    - `(+,+)`: `u(W₁W₂x + W₁w₂ + w₁)`. Standard.
    - `(+,−)`: `conj(u(W₁W₂x + W₁w₂ + w₁))`. θ_out=−1, same affine.
    - `(−,+)`: `conj(u(W₁W₂x + W₁w₂ + w₁))`. θ_out=−1, same affine.
    - `(−,−)`: `u(W₁W₂x + W₁w₂ + w₁)`. θ_out=+1.
    So in all cases `(W_out, w_out) = (W₁W₂, w₁ + W₁w₂)` and `θ_out = θ₁·θ₂`. Simple — verify with a unit test.
- `Base.inv`: the inverse of `(W, w, θ)` has `θ_inv = θ`, `W_inv = W^{-1}`, `w_inv = −W^{-1}·w`.
  (Apply the case analysis above with `U·U^{-1} = id` to confirm.)
- `complete_symop_group` and `check_group`: should just work once `*`/`inv`/`isapprox`
  carry `θ`, but smoke-test it.
- The action helpers (currently inlined as `S = Wᵀ`, `τ = −W^{-1}·w` cached in struct):
  add a method `apply_to_kpoint(symop, k) = symop.θ * symop.S * k`.

### `src/symmetry.jl`

**Important: spglib already returns the antiunitary partners — we just throw them
away today.** `symmetry_operations` calls
`Spglib.get_symmetry_with_site_tensors(cell, tol_symmetry)` (line 93) which returns
a third array `spin_flips ∈ {+1, −1}`. The current code (line 96) does
`rotations[spin_flips.==1], translations[spin_flips.==1]`, dropping the spin-flipping
ops. The comment immediately below acknowledges this is leaving a 2× speedup on the
table for antiferromagnetic order. **This project is essentially: stop throwing
those away, and tag them as `θ=−1`.**

So:

- For `:none` / `:spinless`: spglib doesn't return spin-flips (no spin info). When
  `!any(breaks_TRS, terms)`, synthesise TRS partners ourselves by duplicating each
  `(W, w, +1)` to `(W, w, −1)`. (The TRS partner of identity is `(I, 0, −1)` which
  is plain complex conjugation — i.e. the `k → −k` reduction.)
- For `:collinear`: keep all rows from `get_symmetry_with_site_tensors`. Map
  `spin_flips == +1 → θ = +1` and `spin_flips == −1 → θ = −1`. The θ=−1 partners
  are antiunitaries that *also swap the two spin channels* — see "Spin gotchas".
  Drop these only when `any(breaks_TRS, terms)` (since antiunitarity is broken).
- For `:full` (SOC): out of scope; spglib's site-tensor call also handles vector
  moments differently. Skip.

- `_check_symmetries`: when `θ = −1`, the atom-position check `W·a + w ≡ a' (mod 1)`
  is *unchanged* (the spatial part is the same). What does change is the
  magnetic-moment check: `W·m_a` must equal `+m_{a'}` for `θ=+1` and `−m_{a'}` for
  `θ=−1`. Spglib has already verified this for us; we just need to not break the
  invariant when manipulating SymOps.

- `default_symmetries` (`Model.jl:324`): the gating logic above. Skip duplication
  / drop spin-flips when `any(breaks_TRS, terms)`.
- `symmetrize_ρ`, `accumulate_over_symmetries!`, `lowpass_for_symmetry!`:
  **For real total densities the antiunitary action equals the unitary action** because
  `ρ` is real (`conj` is no-op). With `n_spin = 1` you can short-circuit `θ=−1` partners
  and *do nothing* — they don't add information. With `n_spin = 2` the antiunitary
  swaps the two spin channels (see below). Decide whether to optimise by skipping
  `θ=−1` symops when symmetrising real scalars; if you don't optimise, the result is
  still correct (you double the count and divide by `length(symmetries)`), just wasteful.
- `symmetries_preserving_kgrid` (`symmetry.jl:162`, `175`): replace `symop.S * k` with
  `symop.θ * symop.S * k` everywhere in the orbit/preserves test.

### `src/bzmesh.jl`

- Line 66 (`rotations = [symop.W for symop in symmetries]`): this feeds spglib's
  k-mesh reducer (`get_ir_reciprocal_mesh` / `get_stabilized_reciprocal_mesh`).
  Spglib's k-mesh API does have a `is_time_reversal` flag — but checking the
  Spglib.jl wrapper before relying on it; if it's exposed cleanly, pass our `θ`
  through and let spglib handle the orbit. If not, the cheap workaround is to
  pre-pend `-W` for each `θ=−1` partner so spglib sees an ordinary spatial
  symmetry that gives the right orbit (`θ·S·k = −Sk`). Either way works; the
  flag-based path is cleaner if available.
- Line 294 (`symop.S * k` equivalence test): use `symop.θ * symop.S * k`.

### `src/transfer.jl`

- `find_equivalent_kpt` (line 180): currently looks for `kcoord` modulo a reciprocal
  lattice vector. With antiunitaries you also need to consider `−kcoord`, but
  **only** for code paths that actually want the antiunitary-equivalent k. Most
  callers want a concrete kpt-in-the-stored-list mapping; for those, the existing
  semantics is right. Decide on a per-caller basis. Likely safe default: don't
  change `find_equivalent_kpt`; only change *where* it's called.
- `transfer_blochwave_kpt`, `transfer_blochwave_equivalent_to_actual`: these act
  on coefficients of ψ. If the equivalence used to identify `kpt_in` and `kpt_out`
  is antiunitary, the coefficient transfer must include `conj` and a sign flip on
  G-indices. **This is the single trickiest site in the project.**

### `src/PlaneWaveBasis.jl`

- The k-point construction loop reduces the user's kgrid via `model.symmetries`.
  Make sure it filters by `basis.symmetries` (already does) and that the irreducible
  set respects `θ·S·k`.
- `find_equivalent_kpt` caller at line 115: likely only used in the q-shifted phonon
  flow; double-check.

### `src/response/chi0.jl`, `src/response/hessian.jl`

- The Sternheimer / response solvers iterate over irreducible k. With TRS k-reduction
  the orbit-completion step that reconstructs full-BZ quantities from irreducible
  ones must apply `conj` for `θ=−1` partners. Audit every loop that says
  "for each symop, accumulate" with eigenvectors, not just densities.

### `src/postprocess/current.jl`

- Currents are **odd under TRS**: `j(x) → −j(x)` under `θ=−1`. Symmetrising over
  a TRS-augmented group must include this sign flip, otherwise currents will be
  symmetrised to zero in non-magnetic systems (which is correct in equilibrium,
  but wrong if you're computing a response current).

### `src/terms/hubbard.jl`

- Hubbard occupation matrices `n_{mm'}^{Iσ}` transform as `n → S·n·S†` under unitary
  symmetries (in the spherical-harmonic basis). Antiunitaries add a complex
  conjugation: `n → S·conj(n)·S†` and swap spin in the collinear case. If we don't
  break TRS in the model, this should be consistent — but please add a regression
  test on a Hubbard system without inversion.

### `src/postprocess/phonon.jl`

- Already guarded by the `!breaks_TRS` assertion landed in #1316. Don't relax that
  assertion as part of this PR; phonons under broken TRS is a separate project
  (Dal Corso 2019). The TRS k-point reduction *helps* phonons in the unbroken case
  by halving k-point work, no special handling needed.

### `src/input_output.jl`, `ext/DFTKJLD2Ext`, `ext/DFTKJSON3Ext`

- Serialisation of `SymOp` gains a field. Bump the on-disk version tag and add
  a back-compat reader that defaults `θ = +1` for old files.

### `src/external/DFTKCalculator.jl`, `src/workarounds/forwarddiff_rules.jl`

- Anywhere a `SymOp` is constructed by hand, default `θ = +1`. Grep for `SymOp(`.

### `src/postprocess/elastic.jl`, `src/postprocess/forces.jl` (if it uses symmetrise)

- Forces and stresses are TRS-even. Symmetrising them over the doubled group is
  correct but wasteful; same optimisation note as `symmetrize_ρ`.

---

## Spin gotchas (read carefully before starting)

TRS acts on spin as `iσ_y K` (K = complex conjugation). Concretely:

| `spin_polarization` | TRS action on density | Action when adding `θ=−1` partners |
|---|---|---|
| `:none` | identity (ρ is real scalar) | always safe to add |
| `:spinless` | identity | always safe to add |
| `:collinear`, no net moment | swaps `ρ_↑ ↔ ρ_↓` | safe iff initial moments are zero or paired so that swapping is a self-symmetry |
| `:collinear`, net moment | swaps `ρ_↑ ↔ ρ_↓` | **breaks TRS at the model level**; do not add partners |
| `:full` (SOC, noncollinear) | `iσ_y K` on the spinor | requires Pauli-spinor support; **out of scope** |

In `default_symmetries`, gate the duplication / spin-flip retention on:
1. `!any(breaks_TRS, terms)`,
2. `spin_polarization ∈ (:none, :spinless)` (synthesise partners) *or*
   `:collinear` (keep spglib's spin-flip rows as `θ=−1` SymOps).

For `:collinear` with `θ=−1`, the antiunitary action on the spin-resolved density
swaps `ρ_↑ ↔ ρ_↓` *and* applies the spatial part `(W, w)`. Symmetrisation must
therefore be aware of the spin index — `symmetrize_ρ` currently iterates over the
spin axis independently per symop, which is wrong for `θ=−1` in the collinear
case. Specifically, line ~290 of `symmetry.jl` (`accumulate_over_symmetries!` or
its caller) needs branching on `θ`: for `θ=+1` keep the per-spin loop, for `θ=−1`
swap the spin index of the source density before accumulating.

Don't worry about double-counting between spglib's spin-flip rows and "synthesised"
TRS partners — for `:collinear` we *only* use spglib's output (no synthesis), and
for `:none`/`:spinless` spglib never returns spin-flip rows.

---

## Other gotchas

- **`accumulate_over_symmetries!` GPU kernel** (`symmetry.jl:301`): currently
  pushes `symop.S` and `symop.τ` to the device. Add `symop.θ` to the device array
  as well, and special-case it in the kernel. With `n_spin = 1`, `θ=−1` is no-op
  on real ρ and you can skip the kernel iteration; keep the array shapes consistent.

- **`SYMMETRY_CHECK = true`** (in `SymOp.jl:24`): the `_check_symmetries` routine
  must accept antiunitaries. The atom-position check `W·a + w ≡ a' (mod 1)` is
  *the same* for `θ=±1`. The check should pass without modification; verify.

- **Band structure plotting** (`docs/src/examples`, `ext/DFTKPlotsExt`): `compute_bands`
  evaluates eigenvalues on a user-provided k-path. With TRS, `ε_n(k) = ε_n(−k)`, so
  the path is unchanged but the irreducible BZ is half the size. Don't accidentally
  drop k-points from the path that the user explicitly asked for.

- **Wannier90 extension**: w90 cares about k-point lists and sometimes about the
  irreducible vs full BZ. The interface in `ext/DFTKWannier90Ext` may need
  updating if it expects "no antiunitaries". Check before releasing.

- **`compute_dynmat` factor-of-2 trick**: the derivation we wrote in
  `src/densities.jl` (above `compute_δρ`) assumes TRS, which is exactly the
  predicate enforced by `breaks_TRS`. With this PR, k-point reduction *uses*
  TRS too — make sure the two uses are consistent. There should be no
  double-counting.

- **k-weights**: the irreducible k-weights must reflect the orbit size *under the
  full augmented group*. Easy to get wrong by a factor of 2 if you augment the
  group but forget to recompute weights. Add a sanity check: `sum(basis.kweights) == 1`
  and that integrating `1` over BZ via the irreducible mesh gives 1.

- **Test data**: existing reference data in `test/` was generated with k-point
  reduction over the spatial group only. After this change, the *irreducible* sets
  shrink for inversion-asymmetric crystals, and accumulated quantities (which are
  averaged over the orbit) should be identical to within numerical noise — but
  intermediate per-k arrays (e.g. ψ at given irreducible k) are **not the same
  objects** as before. Tests that read raw ψ from disk, or compare against
  pre-computed eigenvalues at a specific k-index, may need regenerated data or
  index-agnostic comparisons.

- **Performance regression risk**: doubling the symop count temporarily (before
  optimising trivial `θ=−1` actions on real scalars) makes `symmetrize_ρ` 2× slower.
  If this shows up in benchmarks, add a `θ == +1` fast path.

- **MPI**: k-points are MPI-distributed. Halving the irreducible set could leave
  some ranks idle on small jobs. Not a correctness issue but worth noting.

---

## Suggested implementation order

1. **SymOp + tests** (1 day): add `θ` field, fix composition / inv / equality,
   write group-theory unit tests (associativity, closure, identity, inverse) on
   small fabricated groups including antiunitaries.
2. **default_symmetries duplication** (½ day): wire up the `breaks_TRS` gate and
   add antiunitary partners. Run the full test suite and confirm everything
   still passes (since antiunitaries are redundant for ρ at this stage and we
   haven't yet wired k-reduction to use them).
3. **k-mesh reduction** (1–2 days): teach `bzmesh.jl` and
   `symmetries_preserving_kgrid` about `θ`. Verify on Si (no change — already has
   inversion) and GaAs (irreducible count should halve from `(N+1)(N+2)(N+3)/6`
   territory; check against QE's `kpoints.x` for the same lattice).
4. **transfer / chi0 / response audit** (3–5 days): the wavefunction-touching code.
   This is the long tail. Cover with a forces-and-stresses regression test on
   GaAs (TRS-augmented vs forced-`θ=+1`-only — should agree to SCF tolerance).
5. **Currents** (½ day): the only quantity that's TRS-odd, easy to forget.
6. **Hubbard, Wannier90, serialisation** (1–2 days): polish.
7. **Performance pass + skip θ=−1 for real scalars** (½ day).

Total estimate: ~2 weeks for someone familiar with the symmetry code; ~3–4 weeks
otherwise.

## Reference materials to consult while coding

- `docs/src/developer/symmetries.md` — DFTK's symmetry conventions (read first).
- The discussion thread in [#224](https://github.com/JuliaMolSim/DFTK.jl/issues/224) and [#203](https://github.com/JuliaMolSim/DFTK.jl/issues/203).
- Quantum ESPRESSO's `Modules/symm_base.f90` for how `t_rev` (their θ) is
  threaded through. Useful sanity-check on conventions.
- Dal Corso, "Density functional perturbation theory within the projector
  augmented wave method", https://arxiv.org/abs/1906.11673 — for the broken-TRS
  case (out of scope here, but useful to know what we're *not* implementing).
