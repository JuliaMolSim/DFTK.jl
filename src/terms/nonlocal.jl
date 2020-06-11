@doc raw"""
Nonlocal term coming from norm-conserving pseudopotentials in Kleinmann-Bylander form.
``\text{Energy} = \sum_a \sum_{ij} \sum_{n} f_n <ψ_n|p_{ai}> D_{ij} <p_{aj}|ψ_n>.``
"""
struct AtomicNonlocal end
function (::AtomicNonlocal)(basis::PlaneWaveBasis)
    # keep only pseudopotential atoms
    atoms = [elem.psp => positions
             for (elem, positions) in basis.model.atoms
             if elem isa ElementPsp]

    isempty(atoms) && return NoopTerm(basis)
    ops = map(basis.kpoints) do kpt
        P = build_projection_vectors_(basis, atoms, kpt)
        D = build_projection_coefficients_(basis, atoms)
        NonlocalOperator(basis, kpt, P, D)
    end
    TermAtomicNonlocal(basis, ops)
end

struct TermAtomicNonlocal <: Term
    basis::PlaneWaveBasis
    ops::Vector{NonlocalOperator}
end

function ene_ops(term::TermAtomicNonlocal, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)
    ψ === nothing && return (E=T(Inf), ops=term.ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            Pψnk = term.ops[ik].P' * ψnk
            E += basis.kweights[ik] * occ[ik][iband] * real(dot(Pψnk, term.ops[ik].D * Pψnk))
        end
    end

    (E=E, ops=term.ops)
end

@timing "forces_nonlocal" function forces(term::TermAtomicNonlocal, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)
    atoms = basis.model.atoms
    unit_cell_volume = basis.model.unit_cell_volume

    # early return if no pseudopotential atoms
    any(attype isa ElementPsp for (attype, positions) in atoms) || return nothing

    # energy terms are of the form <psi, P C P' psi>, where P(G) = form_factor(G) * structure_factor(G)
    forces = [zeros(Vec3{T}, length(positions)) for (el, positions) in atoms]
    for (iel, (el, positions)) in enumerate(atoms)
        el isa ElementPsp || continue

        C = build_projection_coefficients_(el.psp)
        # TODO optimize: switch this loop and the kpoint loop
        for (ir, r) in enumerate(positions)
            fr = zeros(T, 3)
            for α = 1:3
                tot_red_kpt_number = sum([length(symops) for symops in basis.ksymops])
                ind_red = 1
                for (ik, kpt_irred) in enumerate(basis.kpoints)
                    # Here we need to do an explicit loop over
                    # symmetries, because the atom displacement might break them
                    for isym in 1:length(basis.ksymops[ik])
                        (S, τ) = basis.ksymops[ik][isym]
                        Skpoint, ψSk = apply_ksymop((S, τ), basis, kpt_irred, ψ[ik])
                        Skcoord = Skpoint.coordinate
                        # energy terms are of the form <psi, P C P' psi>,
                        # where P(G) = form_factor(G) * structure_factor(G)
                        qs = [basis.model.recip_lattice * (Skcoord + G)
                              for G in G_vectors(Skpoint)]
                        form_factors = build_form_factors(el.psp, qs)
                        structure_factors = [cis(-2T(π) * dot(Skcoord + G, r))
                                             for G in G_vectors(Skpoint)]
                        P = structure_factors .* form_factors ./ sqrt(unit_cell_volume)
                        dPdR = [-2T(π)*im*(Skcoord + G)[α] for G in G_vectors(Skpoint)] .* P

                        # TODO BLASify this further
                        for iband = 1:size(ψ[ik], 2)
                            ψnSk = @view ψSk[:, iband]
                            fr[α] -= (occ[ik][iband] / tot_red_kpt_number
                                      * real(  dot(ψnSk, P * C * dPdR' * ψnSk)
                                             + dot(ψnSk, dPdR * C * P' * ψnSk)))
                        end
                        ind_red += 1
                    end
                end
            end
            forces[iel][ir] += fr
        end
    end
    forces
end

# Count the number of projection vectors implied by the potential array
# generated by term_nonlocal
function count_n_proj_(psp::PspHgh)
    psp.lmax < 0 ? 0 : sum(size(psp.h[l + 1], 1) * (2l + 1) for l in 0:psp.lmax)::Int
end
function count_n_proj_(atoms)
    sum(count_n_proj_(psp)*length(positions) for (psp, positions) in atoms)::Int
end


# Build projection coefficients for a atoms array generated by term_nonlocal
# The ordering of the projector indices is (A,l,m,i), where A is running over all
# atoms, l, m are AM quantum numbers and i is running over all projectors for a
# given l. The matrix is block-diagonal with non-zeros only if A, l and m agree.
function build_projection_coefficients_(basis::PlaneWaveBasis{T}, atoms) where {T}
    # TODO In the current version the proj_coeffs still has a lot of zeros.
    #      One could improve this by storing the blocks as a list or in a
    #      BlockDiagonal data structure
    n_proj = count_n_proj_(atoms)
    proj_coeffs = zeros(T, n_proj, n_proj)

    count = 0
    for (psp, positions) in atoms, r in positions
        n_proj_psp = count_n_proj_(psp)
        proj_coeffs[count+1:count+n_proj_psp, count+1:count+n_proj_psp] = build_projection_coefficients_(psp)
        count += n_proj_psp
    end # psp, r
    @assert count == n_proj

    proj_coeffs
end

# Builds the projection coefficient matrix for a single atom
# The ordering of the projector indices is (l,m,i), where l, m are the
# AM quantum numbers and i is running over all projectors for a given l.
# The matrix is block-diagonal with non-zeros only if l and m agree.
function build_projection_coefficients_(psp::PspHgh)
    n_proj = count_n_proj_(psp)
    proj_coeffs = zeros(n_proj, n_proj)
    count = 0
    for l in 0:psp.lmax, m in -l:l
        n_proj_l = size(psp.h[l + 1], 1)  # Number of i's
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
function build_projection_vectors_(basis::PlaneWaveBasis{T}, atoms, kpt::Kpoint) where {T}
    n_proj = count_n_proj_(atoms)
    model = basis.model

    proj_vectors = zeros(Complex{T}, length(G_vectors(kpt)), n_proj)
    qs = [model.recip_lattice * (kpt.coordinate + G) for G in G_vectors(kpt)]

    # Compute the columns of proj_vectors = 1/√Ω pihat(k+G)
    # Since the pi are translates of each others, pihat(k+G) decouples as
    # pihat(q) = ∫ p(r-R) e^{-iqr} dr = e^{-iqR} phat(q).
    # The first term is the structure factor, the second the form factor.
    offset = 0 # offset into proj_vectors
    for (psp, positions) in atoms
        # Compute position-independent form factors
        form_factors = build_form_factors(psp, qs)

        # Combine with structure factors
        for r in positions
            # k+G in this formula can also be G, this only changes an unimportant phase factor
            structure_factors = [cis(-2T(π)*dot(kpt.coordinate + G, r)) for G in G_vectors(kpt)]
            for iproj = 1:count_n_proj_(psp)
                @views proj_vectors[:, offset+iproj] = structure_factors .* form_factors[:, iproj]
            end
            offset += count_n_proj_(psp)
        end
    end
    @assert offset == n_proj
    proj_vectors ./ sqrt(model.unit_cell_volume)
end

"""
Build form factors (Fourier transforms of projectors) for an atom centered at 0.
"""
function build_form_factors(psp, qs)
    qnorms = norm.(qs)
    T = real(eltype(qnorms))
    # Compute position-independent form factors
    form_factors = zeros(Complex{T}, length(qs), count_n_proj_(psp))
    count = 1
    for l in 0:psp.lmax, m in -l:l
        prefac_lm = im^l .* ylm_real.(l, m, qs)
        n_proj_l = size(psp.h[l + 1], 1)

        for iproj in 1:n_proj_l
            radial_il = eval_psp_projection_radial.(psp, iproj, l, qnorms)
            form_factors[:, count] = prefac_lm .* radial_il
            count += 1
        end
    end
    @assert count == count_n_proj_(psp) + 1
    form_factors
end
