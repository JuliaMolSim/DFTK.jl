# Implementation plan: homogeneous electric-field response (#1138)

**Goal.** Compute the response of a periodic insulator to a *homogeneous* electric field `E`,
yielding the electronic (clamped-ion) **dielectric tensor `ε∞`** and the static **polarizability**.
Born effective charges are a natural follow-on (see §8).

This document is written for a future implementer. It assumes familiarity with `AGENTS.md`
(design philosophy, conventions, the warm-REPL workflow) and with the existing DFPT machinery in
`src/response/` and `src/postprocess/phonon.jl`. **Read those two before starting.**

The good news: **almost all the machinery already exists.** The field response is a `q = 0`
DFPT solve with a *new right-hand side*. The only genuinely new numerical kernel is the velocity
operator `∂H_k/∂k` applied to the orbitals, obtained by ForwardDiff.

---

## 1. Physics background (Baroni, de Gironcoli, Dal Corso, Giannozzi, RMP 73, 515 (2001))

HTML of the review: <https://ar5iv.labs.arxiv.org/abs/cond-mat/0012092> (§II.C, Eqs. 39–53).

The naive perturbing potential `δV = E·r` is unusable in a periodic system: the position
operator `r` is ill-defined under Born–von-Kármán boundary conditions and unbounded below.
The fix is the **"d/dk" trick** (modern theory of polarization):

* Off-diagonal matrix elements of `r` *are* well defined via the commutator (Eq. 39):

  `⟨ψₘ|r_α|ψₙ⟩ = ⟨ψₘ|[H, r_α]|ψₙ⟩ / (εₘ − εₙ)`,  `m ≠ n`.

* `[H, r_α]` is the **velocity operator** `∂H_k/∂k_α`. Its kinetic part is `i(G+k)_α` (Cartesian);
  crucially it *also* has a **non-local-pseudopotential** contribution (the "nonlocal commutator"),
  which is exactly why we compute the whole thing by ForwardDiff rather than analytically.

* Define the **projected-position orbital** `ψ̄ₙ^α ≡ P_c r_α ψₙ` (`P_c = 1 − P_occ` projects onto
  the unoccupied space). It is obtained from one non-interacting Sternheimer solve (Eq. 48):

  `(H − εₙ) ψ̄ₙ^α = P_c [H, r_α] ψₙ`.

  This `ψ̄ₙ^α` is precisely the `∂u_nk/∂k_α` derivative of the periodic Bloch part.

* The self-consistent field response `Δ^E ψₙ` (our `δψ`) then solves the standard Dyson/Sternheimer
  problem with `ψ̄^α` as the external RHS and the induced Hartree+xc local field folded in
  (Eqs. 50–51) — **this is exactly what `solve_ΩplusK_split` already does**, at `q = 0`.

* Contract to get the electronic dielectric tensor (Eq. 53):

  `ε∞^{βα} = δ_{βα} − (8π/Ω) Σ_k w_k Σₙ f_n Re⟨ψ̄ₙ^β | Δ^{E_α} ψₙ⟩`, with the constant `8π` *derived*
  from DFTK's own `δρ` normalization in §5 (not fitted).

The pipeline per Cartesian direction α is **two linear solves plus one operator application**:

1. `vα ψ  = (∂H_k/∂k_α) ψ_k`   — velocity operator application, *not* a solve (ForwardDiff; §3).
2. `ψ̄^α  = (H−ε)⁻¹ P_c vα ψ`   — non-interacting Sternheimer (`apply_χ0_4P`; §4).
3. `δψ^α = (Ω+K)⁻¹ ψ̄^α`        — full SCF response (`solve_ΩplusK_split`; §4).

The two nested inverses are intrinsic to the d/dk formulation (Baroni Eqs. 48 then 50) — the field's
external RHS `ψ̄^α = P_c r ψ` is itself "one Sternheimer deep" relative to the velocity `vα ψ`; there
is no way to collapse them. Then `ε∞[β,α]` is a set of inner products `⟨ψ̄^β | δψ^α⟩` (§5).

---

## 2. Scope of v1 (keep it small — cf. AGENTS.md "simplicity over feature count")

Mirror the phonon feature's staged, honest scope:

* **Insulators only.** Assert full occupation (see §6). Metals need a Drude/intraband term this
  formula does not capture — out of scope, documented as a limitation.
* **Spin-unpolarized (`:none`)** for v1. Collinear spin is a follow-on; the density is 4D
  `(nx,ny,nz,n_spin)` and the contraction sums spin channels, so nothing conceptual blocks it, but
  keep the first PR to `:none`.
* **Symmetry: full BZ (`symmetries = false`) is the simplest correct baseline; a stabilizer-subgroup
  IBZ (§7, "option B") is the recommended cheap optimization** — it reuses `solve_ΩplusK_split`
  unchanged (no χ0/Dyson rewrite) by building each direction's response basis with `symmetries = G_α`.
  Unlike phonons, this is *not* forced to `symmetries = false`.
* **Time-reversal symmetry assumed** (`q = 0`, no external B field). Reuse the existing
  `breaks_time_reversal_symmetry` assert from `phonon.jl`.
* **LDA/GGA** — whatever the existing `apply_kernel` / `solve_ΩplusK_split` path supports. Do not
  add new kernel code.

Non-goals for v1 (see §8 for follow-ons): Born effective charges; ionic/lattice contribution to `ε`;
collinear spin; the coupled-directions "option A" and symmetry-reduced phonons.

---

## 3. New kernel: the velocity operator `∂H_k/∂k · ψ` (ForwardDiff)

This is the **only** substantial new numerical code and the main implementation risk. **Prototype
it first in the warm REPL** (`~/.claude/tools/jlrepl.sh`) before building anything else.

### What we need
For each k-point and each Cartesian (reduced) direction α, `vα ψ_k = (∂/∂k_α) [H_k] ψ_k`, evaluated
on the *fixed* G-sphere of `kpt` (fixed integer `G_vectors(basis, kpt)` and `kpt.mapping`).

### Approach — ForwardDiff through the k-dependence, mirroring `compute_stresses_cart`
`compute_stresses_cart` (`src/postprocess/stresses.jl`) is the template: it rebuilds the basis with
a `Dual`-valued lattice and ForwardDiffs `energy`. Do the analogous thing for the **k-coordinate**:

* Parametrize `k(t) = kpt.coordinate + t · e_α` (reduced coords), `t` a `ForwardDiff.Dual` scalar.
* Rebuild the **k-dependent term operators** for a `Kpoint` whose `coordinate` is `k(t)` but whose
  `mapping`/integer `G_vectors` are unchanged (same sphere), apply them to `ψ_k`, and take the
  derivative at `t = 0`.

Which terms carry k-dependence in their *operator* (not just via ψ):
* **Kinetic** — yes, via `|G+k|²/2`. (Analytic value `i(G+k)_cart · ψ`, cf. `compute_current` in
  `postprocess/current.jl` — use this only as a *validation* check, not the implementation.)
* **Local potential** — no. Acts by real-space multiplication through the fixed sphere; k-independent.
  Its velocity contribution is zero.
* **Non-local (`TermAtomicNonlocal`)** — yes, via the projectors `P_k(G+k)`. This is the nonlocal
  commutator and *must* be included for accuracy. ForwardDiff handles it uniformly.

Because the projector construction (radial functions × spherical harmonics × structure factors,
evaluated at `Gplusk_vectors`) is already exercised under ForwardDiff by forces/stresses, a
`Dual`-valued `kpt.coordinate` should flow through. **Verify this** — the risk is a spot in the
`Kpoint`/nonlocal path that assumes a real k.

### Suggested shape
```julia
# Returns δHψ[α][ik] = (∂H_k/∂k_α) ψ[ik], α = 1:3, on kpt's own G-sphere.
function compute_δHψ_dk(basis::PlaneWaveBasis{T}, ψ) where {T}
    map(1:3) do α
        map(enumerate(basis.kpoints)) do (ik, kpt)
            ForwardDiff.derivative(zero(T)) do t
                kpt_t = _shift_kpoint(kpt, α, t)          # only .coordinate becomes Dual
                _apply_k_dependent_hamiltonian(basis, kpt_t, ψ[ik])  # kinetic + nonlocal
            end
        end
    end
end
```
Implementation notes:
* Start with kinetic-only, assert it matches the analytic `i(G+k)_cart · ψ` to ~1e-10, then add
  nonlocal.
* `_apply_k_dependent_hamiltonian` should build *only* the k-dependent operators (kinetic +
  nonlocal), not the full Hamiltonian — the local/Hartree/xc parts are k-independent and dropping
  them avoids an FFT-heavy rebuild. Look at how `TermKinetic`/`TermAtomicNonlocal` construct their
  `RealFourierOperator`s for a kpt and rebuild those at `kpt_t`.
* Watch the `@timing`/thread-safety note in AGENTS.md: no `@timing` inside threaded regions.
* Device-agnostic allocation (`zeros_like`/`similar`) if you allocate; but v1 need not target GPU.

### Deliverable of this section
A tested `compute_δHψ_dk` (or similarly named). Test against: (a) analytic kinetic; (b) a
finite-difference `(H_{k+h} − H_{k−h})ψ / 2h` including nonlocal, on silicon.

---

## 4. Reusing the existing DFPT solvers (no new solver code)

Per direction α, feed the velocity RHS through the existing chain (all args other than `vαψ` come
straight from `scfres`):

```julia
# ψ̄^α : projected-position orbitals = P_c r_α ψ  (non-interacting Sternheimer)
ψ̄ = apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues, vαψ;
                 q=zero(Vec3), scfres.occupation_threshold,
                 bandtolalg=BandtolBalanced(scfres)).δψ

# δψ^α : full self-consistent response  (Ω+K)⁻¹, incl. Hartree+xc local field
res = solve_ΩplusK_split(scfres, ψ̄; q=zero(Vec3))
δψ = res.δψ;  δρ = res.δρ
```
* `q = 0` is the homogeneous limit; the whole `k ± q` apparatus collapses to identity.
* `apply_χ0_4P` already applies the `(1 − P_occ)` projection, so the RHS need not be pre-projected.
* `solve_ΩplusK_split` internally handles metals via `δoccupation`, but we forbid metals in v1
  anyway (§6); the insulator path is what we exercise.
* `solve_ΩplusK_split` *does* call `apply_χ0` internally (through `DielectricAdjoint`), which
  symmetrizes `δV` over `basis.symmetries`. That is exactly right and needs no intervention: under
  option B the basis carries `G_α`, and `δV^α` is `G_α`-invariant, so the symmetrization is
  non-destructive (see §7). It is only a *full*-symmetry basis that would annihilate the field
  response — which is why option B uses the `G_α` subgroup basis, never the full-group one.

---

## 5. Contraction and prefactor — derive it from DFTK's own conventions (don't fit)

**Do not fit the constant.** Derive it from `compute_δρ`'s normalization, which the (validated) phonon
machinery already relies on. The chain is short and each step is a DFTK definition or an identity:

1. **DFTK's `δρ` normalization** (`src/densities.jl:98–101`, insulator so `δf = 0`):

   `δρ(r) = Σ_k w_k Σ_n 2 f_n Re( ψ*_nk(r) · δψ_nk(r) )`,   `w_k = kweights`, `f_n = occupation`.

   `f_n` already carries spin degeneracy (`= filled_occupation = 2` for `:none`); `Σ_k w_k` normalizes
   the BZ average. **This is the anchor** — the same `δρ` the dynamical matrix is built from.

2. **Induced macroscopic polarization** (electrons, `e = 1`): `P_β = -(1/Ω) ∫ r_β δρ(r) dr`.
   Because `r_β` is Hermitian and `δψ ⟂ occupied`, `∫ r_β δρ = Σ_k w_k Σ_n 2 f_n Re⟨ψ̄^β_nk | δψ_nk⟩`
   (this is why the well-defined `ψ̄^β = P_c r_β ψ` may replace the ill-defined `r_β` — the occupied
   part of `r_β ψ` is killed by `δψ ⟂ occupied`). So, for the response `δψ^α` to a unit field `E_α`,

   `χ_{βα} = ∂P_β/∂E_α = -(1/Ω) Σ_k w_k Σ_n 2 f_n Re⟨ψ̄^β_nk | δψ^α_nk⟩`.

3. **Dielectric tensor** `ε∞ = 1 + 4π χ` (Gaussian/atomic):

   `ε∞[β,α] = δ_{βα} − (8π/Ω) · Σ_k w_k Σ_n f_n · Re⟨ψ̄^β_nk | δψ^α_nk⟩`.

   **So `C = 8π`, by construction.** Cross-check against the paper: for `:none`, `f_n = 2`, and
   `8π·Σ_n f_n = 16π·Σ_{n=1}^{N/2}` — exactly Baroni Eq. 53. Spin degeneracy lives in `f_n`, so the
   *same* formula covers collinear spin (more k/spin channels, `f_n ≤ 1` per channel) unchanged.

**Sign is fixed, not fitted.** Feed `δHextψ = ψ̄^α` to `solve_ΩplusK_split`, which solves
`(Ω+K) δψ = -δHextψ`, so `δψ^α = -(Ω+K)⁻¹ ψ̄^α`. Since `Ω+K` (the energy Hessian) is positive definite
for a stable insulator, `Re⟨ψ̄^α | δψ^α⟩ = -⟨ψ̄^α|(Ω+K)⁻¹|ψ̄^α⟩ < 0`, so the `−(8π/Ω)` term is
**positive** and `ε∞ > 1`, as it must be. Also note `⟨ψ̄^β|δψ^α⟩ = -⟨ψ̄^β|(Ω+K)⁻¹|ψ̄^α⟩` is symmetric in
`α↔β` (self-adjoint `(Ω+K)⁻¹`), so `ε∞` comes out symmetric — another built-in check.

**Independent confirmations** (now checks, not the definition of `C`):
* **`χ0` small-q limit.** DFTK's macroscopic dielectric constant from the long-wavelength limit of
  `χ0` (head of the dielectric matrix, `ε_macro(q) = 1 + 4π/|q|²·(G{=}0 response)`) must agree as
  `q → 0`. This independently validates constant *and* sign.
* **Known value.** Silicon LDA `ε∞ ≈ 12–13` (LDA slightly overestimates).
* **Isotropy.** For cubic Si the assembled tensor must be `≈ ε·I` — check already on the
  `symmetries = false` full-BZ scaffold, where nothing enforces it.

Also expose the per-cell **polarizability** `Ω·χ` (so `P = χ·E`); it is the same contraction without
the `4π` and `1/Ω`.

---

## 6. Metal guard (assert a gap)

The projected-position / d/dk formula gives only the **interband (bound)** response; a metal's
static field response is dominated by the intraband Drude term this misses (and a metal's static
`ε` diverges — perfect screening). So **assert an insulator**:

* Reuse `check_full_occupation(basis, occupation)` (`src/occupation.jl`, already used by
  `solve_ΩplusK`). Partial occupations (metal, or `T > 0` smearing) → it errors. This is the natural,
  honest guard — no special "something for metals" in v1.
* Emit a clear error message pointing at the limitation ("electric-field response is implemented for
  insulators; the metallic intraband term is not included").

There is nothing better to do for metals cheaply; the intraband term is a separate physics effort.

---

## 7. Symmetries — compute per direction on a stabilizer-subgroup IBZ (no solver rewrite)

**Bottom line: exploit symmetry by reducing the k-grid with the subgroup that leaves the perturbation
invariant (option B, §7.3), reusing `solve_ΩplusK_split` verbatim.** You cannot have *both* the
smallest grid `IBZ(G_full)` *and* uncoupled directions — for the self-consistent part they are
mutually exclusive (§7.1). This is exactly how QE and ABINIT handle the electric-field response.

### 7.1 Why you can't have both "smallest IBZ" and "no coupling" (theory + prior art)

**The invariance rule.** You may reduce the k-sum to `IBZ(G')` for a perturbation `δV` **iff `δV` is
invariant (as a scalar) under `G'`.** Reason: reconstructing the full-BZ response from `IBZ(G')`
contributions uses the fact that `χ0` commutes with symmetry, `χ0_{Sk} δV = S · χ0_k · (S⁻¹ δV)`. The
sum over the star of each k closes on `δV` alone **only if `S⁻¹ δV = δV`** for every `S ∈ G'`.
Otherwise `S⁻¹ δV` drags in *other* perturbations, and you must solve them together (coupling).

For a uniform field:
* A **single** direction `δV^α ∝ e_α·r` is invariant only under `G_α = stabilizer(e_α)`: the mixing
  `S⁻¹ δV^α = Σ_{α'} W⁻¹_{αα'} δV^{α'}` collapses to `δV^α` exactly when `W_cart[S]·e_α = e_α`. ⇒ reduce
  to `IBZ(G_α)`, **no coupling** (option B). Symmetrizing `δρ^α` over the *full* group instead gives
  **zero** — ABINIT names this failure mode "over-symmetrization of the density response".
* The **full triplet** `(δV^x, δV^y, δV^z)` is closed under `G_full` (it spans the vector
  representation), so it *can* live on `IBZ(G_full)` — but then the star mixes the three directions and
  they must be solved **together** (option A, §7.4).

So **smallest grid ⇔ coupled; uncoupled ⇔ subgroup grid.** A genuine tradeoff, not a DFTK artifact.

**Where it *is* free — the non-interacting tensor.** The coupling comes **entirely** from
self-consistency: applying the kernel `K` needs the induced `δρ^α`, a full-BZ symmetry-breaking
density that can't be rebuilt from `IBZ(G_full)` alone. Drop self-consistency and there is no induced
`δV` to reconstruct — the independent-particle tensor `ε0[β,α] ∝ Σ_k Re⟨ψ̄^β_k | ψ̄^α_k⟩` is a plain
BZ integral of a **per-k** quantity obeying `T(Sk) = W T(k) Wᵀ`. So you compute the 3×3 `T(k)` at each
`IBZ(G_full)` point (three *independent* per-k Sternheimer solves — no coupling across k or
directions) and symmetrize the small 3×3 tensor at the very end. **This is the "compute on the IBZ and
symmetrize at the end" pattern** — it holds for the non-interacting part; only the local-field
(self-consistent) part forces the option-A/option-B choice.

**How QE and ABINIT do it.** Both treat the field as **three independent perturbations** (one per
Cartesian direction), not a coupled object, and reduce each perturbation's k-set with the **small
group of that perturbation** — i.e. option B. QE performs "an additional linear-response calculation
for electric fields in 3 independent directions"; ABINIT "select[s] the proper k-point set for each
perturbation by using the symmetries that leave each perturbation invariant", explicitly warning that
naive full-group analysis "leads to over-symmetrization of the density response". Both define the
dielectric tensor only at `q = 0`. See the sources listed at the end of this file.

### 7.2 The key enabler already in DFTK

No solver change is needed to switch symmetry groups, because both symmetrization points in the
response path key off `basis.symmetries`:
* `compute_δρ` (`src/densities.jl:107`) ends with `symmetrize_ρ(basis, δρ)` — reconstructs full-BZ
  `δρ` from IBZ k-points;
* `apply_χ0` symmetrizes `δV` via `symmetrize_ρ(basis, δV)` (`src/response/chi0.jl:527`), reached
  inside `solve_ΩplusK_split` through `DielectricAdjoint`.

The per-k Sternheimer solve is symmetry-agnostic. So exploiting symmetry needs **no change to
`apply_χ0`/`apply_χ0_4P`/`solve_ΩplusK_split`** — you only need the basis to carry the right group
`G_α`. Building it with the *full* group instead triggers the over-symmetrization-to-zero of §7.1.

### 7.3 Recommended (option B): per-direction, stabilizer-subgroup IBZ

For field direction α, both `δV^α` and `δρ^α` are invariant — *as a scalar* — under the **stabilizer
subgroup** `G_α = { S ∈ G_full : W_cart[S]·e_α = e_α }` (proof: `δV^α(S⁻¹r) = -E (W_cart[S] e_α)·r =
δV^α(r)` when `S ∈ G_α`). So symmetrizing `δρ^α` over `G_α` is correct and non-destructive. Pipeline:

1. Ground-state SCF as usual on `IBZ(G_full)`.
2. For each **inequivalent** direction α (1 for cubic, ≤3 in general): build a response basis with
   symmetry group `G_α`, and transfer the ground-state data onto its `IBZ(G_α)` grid. Concrete path:
   * `G_α = filter(S -> matrix_red_to_cart(model, S.W) * e_α ≈ e_α, model.symmetries)`.
   * `basis_α = PlaneWaveBasis(Model(scfres.basis.model; symmetries=G_α); Ecut, kgrid, ...)`
     (a basis inherits its symmetries from its `Model`; there is no direct `symmetries=` basis kwarg).
   * `IBZ(G_full) ⊆ IBZ(G_α)`, so `basis_α` has k-points the SCF basis lacks (images under the broken
     ops `G_full ∖ G_α`). So first `full = unfold_bz(scfres)` (ψ on the full BZ), then
     `ψ_α = transfer_blochwave(full.ψ, full.basis, basis_α)`. Eigenvalues and occupations are
     symmetry-invariant (`ε_{Sk,n}=ε_{k,n}`), so map them to `basis_α.kpoints` by k-coordinate.
     **Confirm `transfer_blochwave` matches k-points by coordinate** — this is the fiddliest bit.
3. Run `solve_ΩplusK_split` **unchanged** on `basis_α` with RHS `ψ̄^α` (§3–§4). `compute_δρ` and the
   internal `apply_χ0` both symmetrize over `G_α` automatically → correct full-BZ `δρ^α`.
4. Contract for the tensor column `ε∞[:, α]`; fill symmetry-equivalent columns from the point group.

**Cost:** `IBZ(G_α)` is larger than `IBZ(G_full)` (cubic Si, field ∥ z: `G_α = C4v`, 8 of 48 ops → ~6×
the wedge), but only the inequivalent directions are solved. **No χ0/Dyson changes** — the new code is
the stabilizer filter, the unfold/transfer bookkeeping (step 2, the bulk of it), the velocity RHS
(§3), and the contraction. Because that bookkeeping is the fiddly part, land the `symmetries = false`
full-BZ path first (§7.6) and add option B on top once the physics is validated.

### 7.4 Optional later optimization (option A): coupled full-`IBZ(G_full)`

The smallest grid keeps `IBZ(G_full)` and treats the field as one **3-vector** perturbation,
symmetrized **covariantly** (rotate real space by `S` *and* mix components by `W_cart[S]`):
`δρ^α ← (1/|G|) Σ_S Σ_{α'} W_cart[S]_{αα'} δρ^{α'}(S⁻¹(r−τ))`. A vector field is invariant (not zeroed)
under this. Because χ0 and `K` act component-wise, the three directions couple *only* through this
symmetrization — but that still means running the Dyson solve on a stacked 3-component density and
adding a covariant (vector) `symmetrize_ρ` (extend `accumulate_over_symmetries!` to carry the
`W_cart` component mixing). Smaller grid, but a real change to the response driver. **Defer** unless
the option-B grid cost bites.

### 7.5 Why phonons still need `symmetries = false` (and what would fix them)

Phonons are harder for a q-specific reason, not a symmetry-machinery one: `q ≠ 0` couples `k` and
`k+q`, so symmetry reduction needs the **little group of q** (`{S : S·q ≡ q}`) *and* symmetry-aware
`k±q` transfer. The field case is `q = 0`, where these complications vanish — which is exactly why the
simple option B works here but not (yet) for phonons. A future generic "response with symmetry group
`G_pert`" in `src/response/` (little group of q + the displacement representation) would let phonons
drop `symmetries = false`; that is strictly more general than the field case (§8).

### 7.6 v1 pragmatics

The one novel/risky part is the velocity operator (§3); the prefactor is settled by derivation (§5)
and the symmetry handling reuses existing machinery. A first cut may run `symmetries = false`
(`G_α` = trivial ⇒ full BZ) purely to validate §3/§5 in isolation; then switch on option B.
Cross-check: option B on `IBZ(G_α)` must reproduce the `symmetries = false` full-BZ tensor on silicon
(§10).

---

## 8. Follow-on work (separate PRs — keep v1 small)

1. **Born effective charges** `Z*_{s,αβ} = Ω ∂P_β/∂u_{s,α}`. These cross-couple the **field**
   response (`δψ^β` / `ψ̄^β` from here) with the **atomic-displacement** response `δρ` already
   produced by `compute_dynmat` (`src/postprocess/phonon.jl`). Two equivalent contractions (force
   linear in field, or polarization linear in displacement); implement one, cross-check with the
   other. Cheap once §3–§5 exist.
2. **Coupled full-`IBZ(G_full)` (option A, §7.4)** and/or **symmetry-reduced phonons**: the vector
   covariant `symmetrize_ρ` + a Dyson driver over a stacked multi-component density. This is the
   `src/response/` capability that also lets phonons (little group of q + displacement representation)
   drop `symmetries = false`. Only worth it if the option-B grid cost bites.
3. **Ionic/lattice dielectric contribution** (`ε_0 = ε∞ + phonon/Born term`) — combines Born charges
   with the phonon dynamical matrix.
4. **Metals**: intraband/Drude contribution.

---

## 9. Concrete deliverables & file layout

* `src/postprocess/dielectric.jl` (new): `compute_δHψ_dk` (velocity operator, §3), a small stabilizer
  filter `G_α` + ψ-transfer helper (§7.3), the per-direction driver, and the public
  `compute_dielectric` / `compute_polarizability` returning a NamedTuple `(; ε∞, polarizability,
  δψ, δρ, ψ̄, ...)` — mirror the shape and docstring style of `phonon_modes`. `include` + `export`
  in `src/DFTK.jl` (grep how `phonon_modes` is wired).
* `test/dielectric.jl` (new): a `:slow`-tagged `@testitem` on silicon LDA (small `Ecut`/`kgrid`),
  asserting (a) `ε∞` isotropy for cubic Si, (b) agreement with the `χ0(q→0)` limit, (c) `ε∞` in the
  expected Si range, and (d) **the option-B `IBZ(G_α)` result matches the `symmetries = false` full-BZ
  result** (the key correctness check for the symmetry handling). Follow the `@testitem`/`@testsetup`
  patterns and reuse `TestCases.silicon`.
* `examples/dielectric.jl` (new): mirror `examples/phonons.jl`, including a "preliminary
  implementation / insulators only" warning box. Add to docs if `phonons.jl` is.

## 10. Validation checklist (before opening the PR)

- [ ] `compute_δHψ_dk` matches analytic kinetic velocity and finite-difference full velocity (§3).
- [ ] `ε∞` (with the derived `C = 8π`, §5) matches the `χ0` small-q limit on silicon — confirms the
      derivation.
- [ ] `ε∞ ≈ 12–13` for Si LDA; isotropic tensor (§5).
- [ ] Metal / `T>0` input errors cleanly via `check_full_occupation` (§6).
- [ ] Option-B `IBZ(G_α)` result matches the `symmetries = false` full-BZ result on Si (§7) — the
      symmetry-handling correctness check.
- [ ] Runs through the warm-REPL single-testitem workflow (`TestItemRunner.run_tests(...)`).

## 11. Suggested implementation order

1. Spike `compute_δHψ_dk` in the REPL; validate vs analytic kinetic, then vs FD-with-nonlocal.
2. Wire the pipeline (velocity op → non-interacting Sternheimer → `solve_ΩplusK_split`) for a single
   direction on the full BZ (`symmetries = false` scaffold); get *a* number for Si.
3. Apply the derived `C = 8π` and sign (§5); confirm `ε∞ ≈ 13` and cross-check against `χ0(q→0)`.
4. Loop over 3 directions; switch to option B (§7.3): stabilizer filter `G_α`, ψ-transfer to the
   `IBZ(G_α)` basis, run `solve_ΩplusK_split` unchanged; check it reproduces the step-2 full-BZ number.
5. Metal guard, docstrings.
6. Test item + example + exports; run the single-testitem workflow, then `Pkg.test("DFTK";
   test_args=["minimal"])` as a sanity check.

---

## Sources

* Baroni, de Gironcoli, Dal Corso, Giannozzi, *Phonons and related crystal properties from
  density-functional perturbation theory*, Rev. Mod. Phys. **73**, 515 (2001) —
  <https://ar5iv.labs.arxiv.org/abs/cond-mat/0012092> (§II.C, Eqs. 39–53: electric fields, the d/dk
  trick, `ε∞`, effective charges).
* QE symmetry / 3-independent-field-directions and dielectric at `q = 0`:
  <https://arxiv.org/pdf/0906.2569> and <https://arxiv.org/pdf/1709.10010> (PHonon capabilities).
* ABINIT DFPT — per-perturbation k-point set and the over-symmetrization warning:
  <https://www.abinit.org/sites/default/files/infos/8.6.3/users/generated_files/help_respfn.html>;
  AD-DFPT discussion of over-symmetrization: <https://arxiv.org/pdf/2509.07785>.
* Gonze & Lee, *Dynamical matrices, Born effective charges, dielectric permittivity tensors …*,
  Phys. Rev. B **55**, 10355 (1997) — foundational DFPT electric-field + symmetry reference.
