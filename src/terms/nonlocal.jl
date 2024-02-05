@doc raw"""
Nonlocal term coming from norm-conserving pseudopotentials in Kleinmann-Bylander form.
```math
\text{Energy} =
    ∑_a ∑_{ij} ∑_{n} f_n \braket{ψ_n}{{\rm proj}_{ai}} D_{ij} \braket{{\rm proj}_{aj}}{ψ_n}.
```
"""
struct AtomicNonlocal end
function (::AtomicNonlocal)(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model

    # keep only pseudopotential atoms and positions
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    psps          = [model.atoms[first(group)].psp      for group in psp_groups]
    psp_positions = [model.positions[group] for group in psp_groups]

    isempty(psp_groups) && return TermNoop()
    ops = map(basis.kpoints) do kpt
        P = build_projection_vectors(basis, kpt, psps, psp_positions)
        D = build_projection_coefficients(T, psps, psp_positions)
        NonlocalOperator(basis, kpt, P, to_device(basis.architecture, D))
    end
    TermAtomicNonlocal(ops)
end

struct TermAtomicNonlocal <: Term
    ops::Vector{NonlocalOperator}
end

@timing "ene_ops: nonlocal" function ene_ops(term::TermAtomicNonlocal,
                                             basis::PlaneWaveBasis{T},
                                             ψ, occupation; kwargs...) where {T}
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(Inf), term.ops)
    end

    E = zero(T)
    for (ik, ψk) in enumerate(ψ)
        Pψk = term.ops[ik].P' * ψk  # nproj x nband
        band_enes = dropdims(sum(real.(conj.(Pψk) .* (term.ops[ik].D * Pψk)), dims=1), dims=1)
        E += basis.kweights[ik] * sum(band_enes .* occupation[ik])
    end
    E = mpi_sum(E, basis.comm_kpts)

    (; E, term.ops)
end

@timing "forces: nonlocal" function compute_forces(::TermAtomicNonlocal,
                                                   basis::PlaneWaveBasis{TT}, ψ, occupation;
                                                   kwargs...) where {TT}
    T = promote_type(TT, real(eltype(ψ[1])))
    model = basis.model
    unit_cell_volume = model.unit_cell_volume
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]

    # Early return if no pseudopotential atoms.
    isempty(psp_groups) && return nothing

    # Energy terms are of the form <ψ, P C P' ψ>, where
    #   P(G) = form_factor(G) * structure_factor(G).
    forces = [zero(Vec3{T}) for _ = 1:length(model.positions)]

    for group in psp_groups
        element = model.atoms[first(group)]

        C = build_projection_coefficients(T, element.psp)
        for (ik, kpt) in enumerate(basis.kpoints)
            # We compute the forces from the irreductible BZ; they are symmetrized later.
            G_plus_k = Gplusk_vectors(basis, kpt)
            G_plus_k_cart = to_cpu(Gplusk_vectors_cart(basis, kpt))
            form_factors = build_form_factors(element.psp, G_plus_k_cart)
            for idx in group
                r = model.positions[idx]
                structure_factors = [cis2pi(-dot(p, r)) for p in G_plus_k]
                P = structure_factors .* form_factors ./ sqrt(unit_cell_volume)

                forces[idx] += map(1:3) do α
                    dPdR = [-2T(π)*im*p[α] for p in G_plus_k] .* P
                    ψk = ψ[ik]
                    δHψk = P * (C * (dPdR' * ψk))
                    -sum(occupation[ik][iband] * basis.kweights[ik] *
                             2real(dot(ψk[:, iband], δHψk[:, iband]))
                         for iband in axes(ψk, 2))
                end  # α
            end  # r
        end  # kpt
    end  # group

    mpi_sum!(forces, basis.comm_kpts)
    symmetrize_forces(basis, forces)
end

# TODO possibly move over to pseudo/NormConservingPsp.jl ?
# Build projection coefficients for a atoms array generated by term_nonlocal
# The ordering of the projector indices is (A,l,m,i), where A is running over all
# atoms, l, m are AM quantum numbers and i is running over all projectors for a
# given l. The matrix is block-diagonal with non-zeros only if A, l and m agree.
function build_projection_coefficients(T, psps, psp_positions)
    # TODO In the current version the proj_coeffs still has a lot of zeros.
    #      One could improve this by storing the blocks as a list or in a
    #      BlockDiagonal data structure
    n_proj = count_n_proj(psps, psp_positions)
    proj_coeffs = zeros(T, n_proj, n_proj)

    count = 0
    for (psp, positions) in zip(psps, psp_positions), _ in positions
        n_proj_psp = count_n_proj(psp)
        block = count+1:count+n_proj_psp
        proj_coeffs[block, block] = build_projection_coefficients(T, psp)
        count += n_proj_psp
    end
    @assert count == n_proj

    proj_coeffs
end

# Builds the projection coefficient matrix for a single atom
# The ordering of the projector indices is (l,m,i), where l, m are the
# AM quantum numbers and i is running over all projectors for a given l.
# The matrix is block-diagonal with non-zeros only if l and m agree.
function build_projection_coefficients(T, psp::NormConservingPsp)
    n_proj = count_n_proj(psp)
    proj_coeffs = zeros(T, n_proj, n_proj)
    count = 0
    for l = 0:psp.lmax, _ = -l:l
        n_proj_l = count_n_proj_radial(psp, l)  # Number of i's
        range = count .+ (1:n_proj_l)
        proj_coeffs[range, range] = psp.h[l + 1]
        count += n_proj_l
    end
    proj_coeffs
end


@doc raw"""
Build projection vectors for a atoms array generated by term_nonlocal

```math
\begin{aligned}
H_{\rm at}  &= \sum_{ij} C_{ij} \ket{{\rm proj}_i} \bra{{\rm proj}_j} \\
H_{\rm per} &= \sum_R \sum_{ij} C_{ij} \ket{{\rm proj}_i(x-R)} \bra{{\rm proj}_j(x-R)}
\end{aligned}
```

```math
\begin{aligned}
\braket{e_k(G') \middle| H_{\rm per}}{e_k(G)}
        &= \ldots \\
        &= \frac{1}{Ω} \sum_{ij} C_{ij} \widehat{\rm proj}_i(k+G') \widehat{\rm proj}_j^*(k+G),
\end{aligned}
```

where ``\widehat{\rm proj}_i(p) = ∫_{ℝ^3} {\rm proj}_i(r) e^{-ip·r} dr``.

We store ``\frac{1}{\sqrt Ω} \widehat{\rm proj}_i(k+G)`` in `proj_vectors`.
"""
function build_projection_vectors(basis::PlaneWaveBasis{T}, kpt::Kpoint,
                                  psps, psp_positions) where {T}
    unit_cell_volume = basis.model.unit_cell_volume
    n_proj = count_n_proj(psps, psp_positions)
    n_G    = length(G_vectors(basis, kpt))
    proj_vectors = zeros(Complex{eltype(psp_positions[1][1])}, n_G, n_proj)
    G_plus_k = to_cpu(Gplusk_vectors(basis, kpt))

    # Compute the columns of proj_vectors = 1/√Ω \hat proj_i(k+G)
    # Since the proj_i are translates of each others, \hat proj_i(k+G) decouples as
    # \hat proj_i(p) = ∫ proj(r-R) e^{-ip·r} dr = e^{-ip·R} \hat proj(p).
    # The first term is the structure factor, the second the form factor.
    offset = 0  # offset into proj_vectors
    for (psp, positions) in zip(psps, psp_positions)
        # Compute position-independent form factors
        G_plus_k_cart = to_cpu(Gplusk_vectors_cart(basis, kpt))
        form_factors = build_form_factors(psp, G_plus_k_cart)

        # Combine with structure factors
        for r in positions
            # k+G in this formula can also be G, this only changes an unimportant phase factor
            structure_factors = map(p -> cis2pi(-dot(p, r)), G_plus_k)
            @views for iproj = 1:count_n_proj(psp)
                proj_vectors[:, offset+iproj] .=
                    structure_factors .* form_factors[:, iproj] ./ sqrt(unit_cell_volume)
            end
            offset += count_n_proj(psp)
        end
    end
    @assert offset == n_proj

    # Offload potential values to a device (like a GPU)
    to_device(basis.architecture, proj_vectors)
end

"""
Build form factors (Fourier transforms of projectors) for an atom centered at 0.
"""
function build_form_factors(psp, G_plus_k::AbstractVector{Vec3{TT}}) where {TT}
    T = real(TT)

    # Pre-compute the radial parts of the non-local projectors at unique |p| to speed up
    # the form factor calculation (by a lot). Using a hash map gives O(1) lookup.

    # Maximum number of projectors over angular momenta so that form factors
    # for a given `p` can be stored in an `nproj x (lmax + 1)` matrix.
    n_proj_max = maximum(l -> count_n_proj_radial(psp, l), 0:psp.lmax; init=0)

    radials = IdDict{T,Matrix{T}}()  # IdDict for Dual compatibility
    for p in G_plus_k
        p_norm = norm(p)
        if !haskey(radials, p_norm)
            radials_p = Matrix{T}(undef, n_proj_max, psp.lmax + 1)
            for l = 0:psp.lmax, iproj_l = 1:count_n_proj_radial(psp, l)
                radials_p[iproj_l, l+1] = eval_psp_projector_fourier(psp, iproj_l, l, p_norm)
            end
            radials[p_norm] = radials_p
        end
    end

    form_factors = Matrix{Complex{T}}(undef, length(G_plus_k), count_n_proj(psp))
    for (ip, p) in enumerate(G_plus_k)
        radials_p = radials[norm(p)]
        count = 1
        for l = 0:psp.lmax, m = -l:l
            # see "Fourier transforms of centered functions" in the docs for the formula
            angular = (-im)^l * ylm_real(l, m, p)
            for iproj_l = 1:count_n_proj_radial(psp, l)
                form_factors[ip, count] = radials_p[iproj_l, l+1] * angular
                count += 1
            end
        end
        @assert count == count_n_proj(psp) + 1
    end
    form_factors
end

# Helpers for phonon computations.
function build_projection_coefficients(basis::PlaneWaveBasis{T}, psp_groups) where {T}
    psps          = [basis.model.atoms[first(group)].psp for group in psp_groups]
    psp_positions = [basis.model.positions[group] for group in psp_groups]
    build_projection_coefficients(T, psps, psp_positions)
end
function build_projection_vectors(basis, kpt, psp_groups; positions=basis.model.positions)
    psps          = [basis.model.atoms[first(group)].psp for group in psp_groups]
    psp_positions = [positions[group] for group in psp_groups]
    build_projection_vectors(basis, kpt, psps, psp_positions)
end
function PDPψk(basis, positions, psp_groups, kpt, kpt_minus_q, ψk)
    D = build_projection_coefficients(basis, psp_groups)
    P = build_projection_vectors(basis, kpt, psp_groups; positions)
    P_minus_q = build_projection_vectors(basis, kpt_minus_q, psp_groups; positions)
    P * (D * P_minus_q' * ψk)
end

function compute_dynmat_δH(::TermAtomicNonlocal, basis::PlaneWaveBasis{T}, ψ, occupation,
                           δψ, δoccupation, q) where {T}
    S = complex(T)
    model = basis.model
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]

    # Early return if no pseudopotential atoms.
    isempty(psp_groups) && return nothing

    forces = [zero(Vec3{S}) for _ = 1:length(model.positions)]
    for group in psp_groups
        element = model.atoms[first(group)]

        C = build_projection_coefficients(S, element.psp)
        ψ_plus_q = multiply_by_expiqr(basis, δψ, q)
        for (ik, kpt) in enumerate(basis.kpoints)
            ψk = ψ[ik]
            δψk_plus_q = ψ_plus_q[ik].ψk
            kpt_plus_q = ψ_plus_q[ik].kpt

            # We compute the forces from the irreductible BZ; they are symmetrized later.
            for idx in group
                forces[idx] += map(1:3) do α
                    δHψk = derivative_wrt_αs(model.positions, α, idx) do positions_αs
                        PDPψk(basis, positions_αs, psp_groups, kpt_plus_q, kpt, ψ[ik])
                    end
                    δHψk_plus_q = derivative_wrt_αs(model.positions, α, idx) do positions_αs
                        PDPψk(basis, positions_αs, psp_groups, kpt, kpt, ψ[ik])
                    end
                    -sum(  2occupation[ik][iband] * basis.kweights[ik]
                               * dot(δψk_plus_q[:, iband], δHψk[:, iband])
                         + δoccupation[ik][iband]  * basis.kweights[ik]
                               * 2real(dot(ψk[:, iband], δHψk_plus_q[:, iband]))
                         for iband in axes(ψk, 2))
                end
            end
        end
    end

    mpi_sum!(forces, basis.comm_kpts)
    symmetrize_forces(basis, forces)
end

@views function compute_dynmat(term::TermAtomicNonlocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                               δψs, δoccupations, q=zero(Vec3{T}), kwargs...) where {T}
    S = complex(T)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    # Two contributions: dynmat_δH and dynmat_δ²H.

    # dynmat_δH
    dynmat_δH = zeros(S, 3, n_atoms, 3, n_atoms)
    for s = 1:n_atoms, α = 1:n_dim
        dynmat_δH[:, :, α, s] .-= stack(
            compute_dynmat_δH(term, basis, ψ, occupation, δψs[α, s], δoccupations[α, s], q)
        )
    end

    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    isempty(psp_groups) && return dynmat_δH

    # dynmat_δ²H
    dynmat_δ²H = zeros(S, 3, n_atoms, 3, n_atoms)
    for s = 1:n_atoms, α = 1:n_dim, β = 1:n_dim  # zero if s ≠ t
        for (ik, kpt) in enumerate(basis.kpoints)
            δ²Hψ = derivative_wrt_αs(basis.model.positions, β, s) do positions_βs
                derivative_wrt_αs(positions_βs, α, s) do positions_βsαs
                    PDPψk(basis, positions_βsαs, psp_groups, kpt, kpt, ψ[ik])
                end
            end
            dynmat_δ²H[β, s, α, s] += sum(occupation[ik][n] * basis.kweights[ik] *
                                              dot(ψ[ik][:, n], δ²Hψ[:, n])
                                          for n in axes(ψ[ik], 2))
        end
    end

    dynmat_δH + dynmat_δ²H
end

function compute_δHψ_αs(::TermAtomicNonlocal, basis::PlaneWaveBasis{T}, ψ, α, s, q) where {T}
    model = basis.model
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]

    ψ_minus_q = multiply_by_expiqr(basis, ψ, -q)
    map(enumerate(basis.kpoints)) do (ik, kpt)
        derivative_wrt_αs(model.positions, α, s) do positions_αs
            PDPψk(basis, positions_αs, psp_groups, kpt, ψ_minus_q[ik].kpt, ψ_minus_q[ik].ψk)
        end
    end  # δHψ
end
