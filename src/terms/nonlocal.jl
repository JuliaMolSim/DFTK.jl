@doc raw"""
Nonlocal term coming from norm-conserving pseudopotentials in Kleinmann-Bylander form.
``\text{Energy} = \sum_a \sum_{ij} \sum_{n} f_n <ψ_n|p_{ai}> D_{ij} <p_{aj}|ψ_n>.``
"""
struct AtomicNonlocal end

function (::AtomicNonlocal)(basis::PlaneWaveBasis)
    (projectors, couplings, positions) = _group_non_locals(basis)
    isempty(projectors) && return TermNoop()
    ops = map(basis.kpoints) do kpt
        P = projection_vectors_to_matrix(
            build_projection_vectors(basis, kpt, projectors, positions)
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
    (projectors, couplings, positions) = _group_non_locals(basis)
    D = build_projection_coupling(couplings, positions)

    # Initialize forces
    forces = [zero(Vec3{T}) for _ in 1:length(basis.model.positions)]
    # K-point (k)
    for (kpt, wk, ψk, θk) in zip(basis.kpoints, basis.kweights, ψ, occupation)
        P = build_projection_vectors(basis, kpt, projectors, positions)
        #  P =  sf * ff_a * ff_r / √Ω =        cis2pi(G⋅r) * ff_a * ff_r / √Ω
        # ∇P = ∇sf * ff_a * ff_r / √Ω = -2πi G cis2pi(G⋅r) * ff_a * ff_r / √Ω
        P_to_∇P = -2T(π) .* im .* Gplusk_vectors(basis, kpt)

        i_position = 1
        # Atom group (a), Position w/in atom group (j)
        for (Pa, Da) in zip(P, D)
            for (Paj, Daj) in zip(Pa, Da)
                # Paj[l][m][i][q] -> Paj[q,lmi]
                Paj = Paj |> flatten |> flatten |> collect |> Base.Fix1(reduce, hcat)
                # Daj[l][m][i,i'] -> Daj[lmi,l'm'i']
                Daj = Daj |> flatten .|> sparse |> splat(blockdiag)
                # Paj[q,lmi] -> ∇Paj[q,lmi][α]
                ∇Paj = P_to_∇P .* Paj
                # Paj[q,lmi][α] -> Paj[q,lmi,α]
                ∇Paj = permutedims(
                    reshape(reinterpret(Complex{T}, ∇Paj), 3, size(∇Paj)...),
                    (2, 3, 1)
                )
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

function _group_non_locals(basis::PlaneWaveBasis)
    model = basis.model
    arch = basis.architecture
    qgrid = basis.atom_qgrid
    quadrature_method = basis.atom_rft_quadrature_method
    interpolation_method = basis.atom_q_interpolation_method

    # Filter for atom groups whose potentials have a non-local part
    atom_groups = [group for group in model.atom_groups
                   if hasquantity(model.atoms[first(group)].potential, :non_local_potential)]
    # Get the atom of each atom group
    atoms = [model.atoms[first(group)] for group in atom_groups]
    # Fourier transform and interpolate the projectors once outside the k-point loop
    projectors = map(atoms) do atom
        map(atom.potential.non_local_potential.projectors) do projs_al
            map(projs_al) do proj_ali
                evaluate(
                    rft(to_device(arch, proj_ali), qgrid; quadrature_method),
                    interpolation_method
                )
            end  # i
        end  # l
    end  # a
    # Collect coupling matrices for each species
    couplings = map(atoms) do atom
        map(atom.potential.non_local_potential.coupling) do coupling_l
            to_device(arch, coupling_l)
        end
    end
    # Collect positions by atom group
    positions = [model.positions[group] for group in atom_groups]

    return (projectors, couplings, positions)
end
