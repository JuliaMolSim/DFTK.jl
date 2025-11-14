using LinearAlgebra
using Random

"""
Structure for Hubbard manifold choice and projectors extraction.

It is to be noted that, despite the name used in literature, this is
not a manifold in the mathematical sense.

Overview of fields:
- `psp`: Pseudopotential containing the atomic orbital projectors
- `iatoms`: Atom indices that are part of the manifold.
- `l`: Angular momentum quantum number of the manifold.
- `i`: Index of the atomic orbital within the given l.

See also the convenience constructors, to construct a manifold more easily.
"""
struct OrbitalManifold
    psp::NormConservingPsp
    iatoms::Vector{Int64}
    l::Int64
    i::Int64
end
function OrbitalManifold(atoms::Vector{<:Element}, atom::ElementPsp, label::AbstractString)
    OrbitalManifold(atom, findall(at -> at === atom, atoms), label)
end
function OrbitalManifold(atom::ElementPsp, iatoms::Vector{Int64}, label::AbstractString)
    OrbitalManifold(atom.psp, iatoms, label)
end
function OrbitalManifold(psp::NormConservingPsp, iatoms::Vector{Int64}, label::AbstractString)
    (; l, i) = find_pswfc(psp, label)
    OrbitalManifold(psp, iatoms, l, i)
end

function check_hubbard_manifold(manifold::OrbitalManifold, model::Model)
    for atom in model.atoms[manifold.iatoms]
        atom isa ElementPsp || error("Orbital manifold elements must have a psp.")
        atom.psp === manifold.psp || error("Orbital manifold psp $(manifold.psp.identifier) " *
                                           "does not match the psp of atom $atom")
    end
    isempty(manifold.iatoms) && error("Orbital manifold has no atoms.")
    # Tricky: make sure that the iatoms are consistent with the symmetries of the model,
    #         i.e. that the manifold would be a valid atom group
    atom_positions = model.positions[manifold.iatoms]
    for symop in model.symmetries
        W, w = symop.W, symop.w
        for coord in atom_positions
            # If all elements of a difference in diffs is integer, then
            # W * coord + w and pos are equivalent lattice positions
            if !any(c -> is_approx_integer(W * coord + w - c; atol=SYMMETRY_TOLERANCE),
                    atom_positions)
                error("Inconsistency between orbital manifold and model symmetries: " *
                      "Cannot map the atom at position $coord to another atom of the manifold " *
                      "under the symmetry operation (W, w):\n($W, $w)")
            end
        end
    end

    nothing
end

function extract_manifold(manifold::OrbitalManifold, projectors, labels)
    # We extract the labels only for orbitals belonging to the manifold
    proj_indices = findall(orb -> (orb.iatom ∈  manifold.iatoms
                                && orb.l     == manifold.l
                                && orb.n     == manifold.i), labels)
    isempty(proj_indices) && @warn "Projector for $(manifold) not found."
    manifold_labels = labels[proj_indices]
    manifold_projectors = map(enumerate(projectors)) do (ik, projk)
        projk[:, proj_indices]
    end
    (; manifold_labels, manifold_projectors)
end

struct Hubbard{T}
    manifolds::Vector{OrbitalManifold}
    U::Vector{T}
    function Hubbard(manifolds::Vector{OrbitalManifold}, U)
        U = austrip.(U)
        new{typeof(U[1])}(manifolds, U)
    end
end
function (hubbard::Hubbard)(basis::AbstractBasis)
    for manifold in hubbard.manifolds
        check_hubbard_manifold(manifold, basis.model)
    end

    projs, labels = atomic_orbital_projectors(basis)
    manifold_labels, projs_matrices = [], []
    for manifold in hubbard.manifolds
        push!(manifold_labels, extract_manifold(manifold, projs, labels).manifold_labels)
        push!(projs_matrices, extract_manifold(manifold, projs, labels).manifold_projectors)
    end
    TermHubbard(hubbard.manifolds, hubbard.U, projs_matrices, manifold_labels)
end

struct TermHubbard{T, PT, L} <: Term
    manifolds::Vector{OrbitalManifold}
    U::Vector{T}
    P::Vector{PT}
    labels::Vector{L}
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard, 
                                            basis::PlaneWaveBasis{T}, 
                                            ψ, occupation; hubbard_n=nothing,
                                            kwargs...) where {T}
    if isnothing(hubbard_n)
       return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
    # Concatenate projectors from all manifolds per k-point as well as U value for each projector
    projs = [hcat((P_manifold[ik] for P_manifold in term.P)...) for ik in 1:length(term.P[1])]
    U = [term.U[iman] for iman in 1:length(term.manifolds) for _ in 1:size(term.P[iman][1],2)]

    filled_occ = filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components
    # For the ops we have to reshape hubbard_n to match the NonlocalOperator structure, 
    # using a block diagonal form whit one block per atom per manifold
    # 2l+1 is the number of projectors per atom, i.e. the dimension of the blocks
    nhub_size = sum((2*manifold.l+1)*length(manifold.iatoms) for manifold in term.manifolds)
    nhub = [zeros(Complex{T}, nhub_size, nhub_size) for _ in 1:n_spin]
    E = zero(T)
    offset = 0
    for (iman, manifold) in enumerate(term.manifolds)
        nproj_atom = (2*manifold.l+1)
        for σ in 1:n_spin, iatom in 1:length(manifold.iatoms)
            proj_range = (offset + (iatom-1)*nproj_atom + 1):(offset + iatom*nproj_atom)
            nhub[σ][proj_range, proj_range] = hubbard_n[iman][σ, iatom, iatom]
            E += filled_occ * 1/T(2) * term.U[iman] *
                 real(tr(hubbard_n[iman][σ, iatom, iatom] * (I - hubbard_n[iman][σ, iatom, iatom])))
        end
        offset += nproj_atom*length(manifold.iatoms)
    end
    ops = [NonlocalOperator(basis, kpt, projs[ik], 1/T(2) * U .* (I - 2*nhub[kpt.spin])) 
           for (ik,kpt) in enumerate(basis.kpoints)]
    return (; E, ops)
end

"""
    compute_hubbard_n(term::TermHubbard, manifold_index, basis, ψ, occupation)

Computes a matrix hubbard_n of size (n_spin, natoms, natoms), where each entry hubbard_n[σ, iatom, jatom]
contains the submatrix of the occupation matrix corresponding to the projectors
of atom iatom and atom jatom, with dimensions determined by the number of projectors for each atom.
The atoms and orbitals are defined by the manifold tuple.

    hubbard_n[σ, iatom, jatom][m1, m2] = Σₖ₍ₛₚᵢₙ₎Σₙ weights[ik, ibnd] * ψₙₖ' * Pᵢₘ₁ * Pᵢₘ₂' * ψₙₖ

where n or ibnd is the band index, ``weights[ik, ibnd] = kweights[ik] * occupation[ik, ibnd]``
and ``Pᵢₘ₁`` is the pseudoatomic orbital projector for atom i and orbital m₁
(just the magnetic quantum number, since l is fixed, as is usual in the literature).
For details on the projectors see `atomic_orbital_projectors`.
"""
function compute_hubbard_n(term::TermHubbard,
                           basis::PlaneWaveBasis{T},
                           ψ, occupation) where {T}
    filled_occ = filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components

    hubbard_ns = Vector{Array}(undef, length(term.manifolds))
    for (iman, manifold) in enumerate(term.manifolds)
        manifold_atoms = manifold.iatoms
        natoms = length(manifold_atoms)
        l = manifold.l
        projectors = reshape_hubbard_proj(term.P[iman], term.labels[iman], manifold)
        hubbard_n = Array{Matrix{Complex{T}}}(undef, n_spin, natoms, natoms)
        for σ in 1:n_spin
            for idx in 1:natoms, jdx in 1:natoms
                hubbard_n[σ, idx, jdx] = zeros(Complex{T}, 2l+1, 2l+1)
                for ik = krange_spin(basis, σ)
                    j_projection = ψ[ik]' * projectors[ik][jdx] # <ψ|ϕJ>
                    i_projection = projectors[ik][idx]' * ψ[ik] # <ϕI|ψ>
                    # Sums over the bands, dividing by filled_occ to deal
                    # with the physical two spin channels separately
                    hubbard_n[σ, idx, jdx] .+= (basis.kweights[ik] * i_projection *
                                                diagm(occupation[ik]/filled_occ) * j_projection)
                end
                hubbard_n[σ, idx, jdx] = mpi_sum(hubbard_n[σ, idx, jdx], basis.comm_kpts)
            end
        end
        hubbard_n = symmetrize_hubbard_n(basis.model, manifold, hubbard_n; basis.symmetries)
        hubbard_ns[iman] = hubbard_n
    end
    hubbard_ns
end


"""
This function reshapes for each kpoint the projectors matrix to a vector of matrices,
taking only the columns corresponding to orbitals in the manifold and splitting them
into different matrices, one for each atom. Columns in the same matrix differ only in
the value of the magnetic quantum number m of the corresponding orbitals.
"""
function reshape_hubbard_proj(projectors::Vector{Matrix{Complex{T}}},
                              labels, manifold) where {T}
    natoms = length(manifold.iatoms)
    l = manifold.l
    @assert all(label -> label.l == manifold.l, labels) "$(labels)"
    @assert length(labels) == natoms * (2l+1)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for _ = 1:length(projectors)]
    for (idx, iatom) in enumerate(manifold.iatoms)
        for i = 1:2l+1:length(labels)
            iatom != labels[i].iatom && continue
            for (ik, projk) in enumerate(projectors)
                p_I[ik][idx] = projk[:, i:i+2l]
            end
        end
    end

    p_I
end
