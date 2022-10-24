@doc raw"""
Nonlocal term coming from norm-conserving pseudopotentials in Kleinmann-Bylander form.
``\text{Energy} = \sum_a \sum_{ij} \sum_{n} f_n <ψ_n|p_{ai}> D_{ij} <p_{aj}|ψ_n>.``
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
        P = build_projection_vectors_(basis, kpt, psps, psp_positions)
        D = build_projection_coefficients_(T, psps, psp_positions, array_type = array_type(basis))
        NonlocalOperator(basis, kpt, P, D)
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
        return (E=T(Inf), ops=term.ops)
    end

    E = zero(T)
    for (ik, ψk) in enumerate(ψ)
        Pψk = term.ops[ik].P' * ψk  # nproj x nband
        band_enes = dropdims(sum(real.(conj.(Pψk) .* (term.ops[ik].D * Pψk)), dims=1), dims=1)
        E += basis.kweights[ik] * sum(band_enes .* occupation[ik])
    end
    E = mpi_sum(E, basis.comm_kpts)

    (E=E, ops=term.ops)
end

@timing "forces: nonlocal" function compute_forces(::TermAtomicNonlocal,
                                                   basis::PlaneWaveBasis{TT},
                                                   ψ, occupation; kwargs...) where {TT}
    T = promote_type(TT, real(eltype(ψ[1])))
    model = basis.model
    unit_cell_volume = model.unit_cell_volume
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]

    # early return if no pseudopotential atoms
    isempty(psp_groups) && return nothing

    # energy terms are of the form <psi, P C P' psi>, where P(G) = form_factor(G) * structure_factor(G)
    forces = [zero(Vec3{T}) for _ in 1:length(model.positions)]
    for group in psp_groups
        element = model.atoms[first(group)]

        C = build_projection_coefficients_(T, element.psp)
        for (ik, kpt) in enumerate(basis.kpoints)
            # we compute the forces from the irreductible BZ; they are symmetrized later
            qs_cart = Gplusk_vectors_cart(basis, kpt)
            qs = Gplusk_vectors(basis, kpt)
            form_factors = build_form_factors(element.psp, qs_cart)
            for idx in group
                r = model.positions[idx]
                structure_factors = [cis2pi(-dot(q, r)) for q in qs]
                P = structure_factors .* form_factors ./ sqrt(unit_cell_volume)

                forces[idx] += map(1:3) do α
                    dPdR = [-2T(π)*im*q[α] for q in qs] .* P
                    ψk = ψ[ik]
                    dHψk = P * (C * (dPdR' * ψk))
                    -sum(occupation[ik][iband] * basis.kweights[ik] *
                         2real(dot(ψk[:, iband], dHψk[:, iband]))
                         for iband=1:size(ψk, 2))
                end  # α
            end  # r
        end  # kpt
    end  # group

    forces = mpi_sum!(forces, basis.comm_kpts)
    symmetrize_forces(basis, forces)
end

# TODO possibly move over to pseudo/NormConservingPsp.jl ?
# Build projection coefficients for a atoms array generated by term_nonlocal
# The ordering of the projector indices is (A,l,m,i), where A is running over all
# atoms, l, m are AM quantum numbers and i is running over all projectors for a
# given l. The matrix is block-diagonal with non-zeros only if A, l and m agree.
function build_projection_coefficients_(T, psps, psp_positions; array_type = Array)
    # TODO In the current version the proj_coeffs still has a lot of zeros.
    #      One could improve this by storing the blocks as a list or in a
    #      BlockDiagonal data structure
    n_proj = count_n_proj(psps, psp_positions)
    proj_coeffs = zeros(T, n_proj, n_proj)

    count = 0
    for (psp, positions) in zip(psps, psp_positions), _ in positions
        n_proj_psp = count_n_proj(psp)
        block = count+1:count+n_proj_psp
        proj_coeffs[block, block] = build_projection_coefficients_(T, psp)
        count += n_proj_psp
    end  # psp, r
    @assert count == n_proj

    # GPU computation only : build the coefficients on CPU then offload them to the GPU
    convert(array_type, proj_coeffs)
end

# Builds the projection coefficient matrix for a single atom
# The ordering of the projector indices is (l,m,i), where l, m are the
# AM quantum numbers and i is running over all projectors for a given l.
# The matrix is block-diagonal with non-zeros only if l and m agree.
function build_projection_coefficients_(T, psp::NormConservingPsp)
    n_proj = count_n_proj(psp)
    proj_coeffs = zeros(T, n_proj, n_proj)
    count = 0
    for l in 0:psp.lmax, m in -l:l
        n_proj_l = count_n_proj_radial(psp, l)  # Number of i's
        range = count .+ (1:n_proj_l)
        proj_coeffs[range, range] = psp.h[l + 1]
        count += n_proj_l
    end # l, m
    proj_coeffs
end


"""
Build projection vectors for a atoms array generated by term_nonlocal

H_at  = sum_ij Cij |pi> <pj|
H_per = sum_R sum_ij Cij |pi(x-R)> <pj(x-R)|
      = sum_R sum_ij Cij |pi(x-R)> <pj(x-R)|

<e_kG'|H_per|e_kG> = ...
                   = 1/Ω sum_ij Cij pihat(k+G') pjhat(k+G)^*

where pihat(q) = ∫_R^3 pi(r) e^{-iqr} dr

We store 1/√Ω pihat(k+G) in proj_vectors.
"""
function build_projection_vectors_(basis::PlaneWaveBasis{T}, kpt::Kpoint,
                                   psps, psp_positions) where {T}
    unit_cell_volume = basis.model.unit_cell_volume
    n_proj = count_n_proj(psps, psp_positions)
    n_G    = length(G_vectors(basis, kpt))
    proj_vectors = zeros(Complex{T}, n_G, n_proj)

    # Compute the columns of proj_vectors = 1/√Ω pihat(k+G)
    # Since the pi are translates of each others, pihat(k+G) decouples as
    # pihat(q) = ∫ p(r-R) e^{-iqr} dr = e^{-iqR} phat(q).
    # The first term is the structure factor, the second the form factor.
    offset = 0  # offset into proj_vectors
    for (psp, positions) in zip(psps, psp_positions)
        # Compute position-independent form factors
        form_factors = build_form_factors(psp, Gplusk_vectors_cart(basis, kpt))

        # Combine with structure factors
        for r in positions
            # k+G in this formula can also be G, this only changes an unimportant phase factor
            Gs = Array(Gplusk_vectors(basis, kpt))  # GPU computation only: get Gs on CPU for the following map
            structure_factors = map(q -> cis2pi(-dot(q, r)), Gs)
            @views for iproj = 1:count_n_proj(psp)
                proj_vectors[:, offset+iproj] .= (
                    structure_factors .* form_factors[:, iproj] ./ sqrt(unit_cell_volume)
                )
            end
            offset += count_n_proj(psp)
        end
    end
    @assert offset == n_proj
    # GPU computation only : build the vectors on CPU then offload them to the GPU
    convert(array_type(basis), proj_vectors)
end

"""
Build form factors (Fourier transforms of projectors) for an atom centered at 0.
"""
function build_form_factors(psp, qs)
    qs = Array(qs)  # GPU computation only : get qs back on CPU
    T = real(eltype(first(qs)))

    # Pre-compute the radial parts of the non-local projectors at unique |q| to speed up
    # the form factor calculation (by a lot). Using a hash map gives O(1) lookup.

    # Maximum number of projectors over angular momenta so that form factors
    # for a given `q` can be stored in an `nproj x (lmax + 1)` matrix.
    n_proj_max = maximum(l -> count_n_proj_radial(psp, l), 0:psp.lmax; init=0)

    radials = IdDict{T,Matrix{T}}()  # IdDict for Dual compatability
    for q in qs
        q_norm = norm(q)
        if !haskey(radials, q_norm)
            radials_q = Matrix{T}(undef, n_proj_max, psp.lmax + 1)
            for l in 0:psp.lmax, iproj_l in 1:count_n_proj_radial(psp, l)
                radials_q[iproj_l, l+1] = eval_psp_projector_fourier(psp, iproj_l, l, q_norm)
            end
            radials[q_norm] = radials_q
        end
    end

    form_factors = Matrix{Complex{T}}(undef, length(qs), count_n_proj(psp))
    for (iq, q) in enumerate(qs)
        radials_q = radials[norm(q)]
        count = 1
        for l in 0:psp.lmax, m in -l:l
            angular = im^l * ylm_real(l, m, q)
            for iproj_l in 1:count_n_proj_radial(psp, l)
                form_factors[iq, count] = radials_q[iproj_l, l+1] * angular
                count += 1
            end
        end
        @assert count == count_n_proj(psp) + 1
    end
    form_factors
end
