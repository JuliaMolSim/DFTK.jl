@doc raw"""
Nonlocal term coming from norm-conserving pseudopotentials in Kleinmann-Bylander form.
``\text{Energy} = \sum_a \sum_{ij} \sum_{n} f_n <ψ_n|p_{ai}> D_{ij} <p_{aj}|ψ_n>.``
"""
struct AtomicNonlocal end

function _prepare_non_local(basis::PlaneWaveBasis)
    model = basis.model

    # Filter for atom groups whos potential has a non-local part
    atom_groups = [group for group in model.atom_groups
                   if hasquantity(model.atoms[first(group)].potential, :non_local_potential)]
    # Get one element from each atom group
    atoms = [model.atoms[first(group)] for group in atom_groups]
    # Collect positions by atom group
    positions = [model.positions[group] for group in atom_groups]

    # Construct callables for each projector of each species (a)
    evaluators = map(atoms) do atom
        non_local_potential_real = atom.potential.non_local_potential
        non_local_potential_fourier = rft(non_local_potential_real, basis.atom_qgrid;
                                          quadrature_method=basis.atom_rft_quadrature_method)
        # Angular momentum (l)
        map(non_local_potential_fourier.projectors) do proj_l
            # Projector (i)
            map(proj_l) do proj_li
                evaluate(proj_li, basis.atom_q_interpolation_method)
            end  # i
        end  # l
    end  # a

    # Collect coupling matrices for each species
    couplings = [atom.potential.non_local_potential.coupling for atom in atoms]

    return (evaluators, couplings, positions)
end

function (::AtomicNonlocal)(basis::PlaneWaveBasis{T}) where {T}
    (evaluators, couplings, positions) = _prepare_non_local(basis)
    isempty(evaluators) && return TermNoop()
    ops = map(basis.kpoints) do kpt
        P = projection_vectors_to_matrix(
            build_projection_vectors(basis, kpt, evaluators, positions)
        )
        D = projection_coupling_to_matrix(
            build_projection_coupling(couplings, positions)
        )
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

# TODO: Forces are incorrect
@timing "forces: nonlocal" function compute_forces(term::TermAtomicNonlocal,
                                                   basis::PlaneWaveBasis{TT},
                                                   ψ, occupation; kwargs...) where {TT}
    T = promote_type(TT, real(eltype(ψ[1])))
    model = basis.model

    # Early return if no atom has a non-local potential
    has_non_local = any(hasquantity(model.atoms[first(group)], :non_local_potential)
                        for group in model.atom_groups)
    !has_non_local && return nothing

    # Build up atom-grouped evaluators, coupling matrices, and positions
    (evaluators, couplings, positions) = _prepare_non_local(basis)
    D = build_projection_coupling(couplings, positions)

    # Initialize forces
    forces = [zero(Vec3{T}) for _ in 1:length(basis.model.positions)]
    # K-point (k)
    for (kpt, wk, ψk, θk) in zip(basis.kpoints, basis.kweights, ψ, occupation)
        P = build_projection_vectors(basis, kpt, evaluators, positions)
        #  P =  sf * ff_a * ff_r / √Ω =        cis2pi(G⋅r) * ff_a * ff_r / √Ω
        # ∇P = ∇sf * ff_a * ff_r / √Ω = -2πi G cis2pi(G⋅r) * ff_a * ff_r / √Ω
        P_to_∇P = -2T(π) .* im .* Gplusk_vectors(basis, kpt)

        i_position = 1
        # Atom group (a), Position w/in atom group (j)
        for (Pa, Da) in zip(P, D)
            for (Paj, Daj) in zip(Pa, Da)
                # P_aj[l][m][i][q] -> P_aj[q,lmi]
                Paj = Paj |> flatten |> flatten |> collect |> Base.Fix1(reduce, hcat)
                Daj = Daj |> flatten .|> sparse |> splat(blockdiag)

                ∇Paj = P_to_∇P .* Paj
                # Paj[q,lmi][α] -> Paj[q,lmi,α]
                ∇Paj = permutedims(reshape(reinterpret(Complex{T}, ∇Paj), 3, size(∇Paj)...), (2, 3, 1))
                # Force component (α)
                forces[i_position] += map(1:3) do α
                    dPaj_dRα = ∇Paj[:,:,α]
                    dHψk = Paj * (Daj * (dPaj_dRα' * ψk))
                    # Band index (n)
                    -sum(zip(θk, eachcol(ψk), eachcol(dHψk))) do (θkn, ψkn, dHψkn)
                        θkn * wk * 2 * real(dot(ψkn, dHψkn))
                    end  # n
                end  # α
                i_position += 1
            end  # j
        end  # a
    end  # k

    forces = mpi_sum!(forces, basis.comm_kpts)
    symmetrize_forces(basis, forces)
end

# @timing "forces: nonlocal" function compute_forces(::TermAtomicNonlocal,
#                                                    basis::PlaneWaveBasis{TT},
#                                                    ψ, occupation; kwargs...) where {TT}
#     T = promote_type(TT, real(eltype(ψ[1])))
#     model = basis.model
#     unit_cell_volume = model.unit_cell_volume
#     (evaluators, couplings, positions) = prepare_non_local(basis)

#     # early return if no pseudopotential atoms
#     isempty(evaluators) && return nothing

#     # energy terms are of the form <psi, P C P' psi>, where P(G) = form_factor(G) * structure_factor(G)
#     forces = [zero(Vec3{T}) for _ in 1:length(model.positions)]
#     for group in psp_groups
#         element = basis.fourier_atoms[first(group)]

#         C = build_projection_coefficients_(T, element)
#         for (ik, kpt) in enumerate(basis.kpoints)
#             # we compute the forces from the irreductible BZ; they are symmetrized later
#             qs = Gplusk_vectors(basis, kpt)
#             qs_cart = to_cpu(Gplusk_vectors_cart(basis, kpt))
#             form_factors = build_form_factors(element, qs_cart)
#             for idx in group
#                 r = model.positions[idx]
#                 structure_factors = [cis2pi(-dot(q, r)) for q in qs]
#                 P = structure_factors .* form_factors ./ sqrt(unit_cell_volume)

#                 forces[idx] += map(1:3) do α
#                     dPdR = [-2T(π)*im*q[α] for q in qs] .* P
#                     ψk = ψ[ik]
#                     dHψk = P * (C * (dPdR' * ψk))
#                     -sum(occupation[ik][iband] * basis.kweights[ik] *
#                          2real(dot(ψk[:, iband], dHψk[:, iband]))
#                          for iband=1:size(ψk, 2))
#                 end  # α
#             end  # r
#         end  # kpt
#     end  # group

#     forces = mpi_sum!(forces, basis.comm_kpts)
#     symmetrize_forces(basis, forces)
# end
