using LinearAlgebra
using Random

"""
Structure for manifold choice and projectors extraction.

Overview of fields:
- `iatom`: Atom position in the atoms array.
- `species`: Chemical Element as in ElementPsp.
- `label`: Orbital name, e.g.: "3S".

All fields are optional, only the given ones will be used for selection.
Can be called with an orbital NamedTuple and returns a boolean
  stating whether the orbital belongs to the manifold.
"""
@kwdef struct OrbitalManifold
    iatom   ::Union{Int64,  Nothing} = nothing
    species ::Union{Symbol, AtomsBase.ChemicalSpecies, Nothing} = nothing
    label   ::Union{String, Nothing} = nothing
end
function (s::OrbitalManifold)(orbital)
    iatom_match    = isnothing(s.iatom)   || (s.iatom == orbital.iatom)
    species_match  = isnothing(s.species) || (s.species == orbital.species)
    label_match    = isnothing(s.label)   || (s.label == orbital.label)
    iatom_match && species_match && label_match
end

function extract_manifold(basis::PlaneWaveBasis{T}, projectors, labels,
                          manifold::OrbitalManifold) where {T}
    # We extract the labels only for orbitals belonging to the manifold
    manifold_labels = filter(manifold, labels)
    isempty(manifold_labels) && @warn "Projector for $(manifold) not found."
    proj_indices = findall(orb->manifold(orb)==true, labels)
    manifold_projectors = map(enumerate(projectors)) do (ik, projk)
        projk[:, proj_indices]
    end
    (; manifold_labels, manifold_projectors)
end

"""
    compute_nhubbard(manifold, basis, ψ, occupation; [projectors, labels, positions])

Computes a matrix nhubbard of size (nspins, natoms, natoms), where each entry nhubbard[iatom, jatom]
  contains the submatrix of the occupation matrix corresponding to the projectors
  of atom iatom and atom jatom, with dimensions determined by the number of projectors for each atom.
  The atoms and orbitals are defined by the manifold tuple.

    nhubbard[σ, iatom, jatom][m1, m2] = Σₖ₍ₛₚᵢₙ₎Σₙ weights[ik, ibnd] * ψₙₖ' * Pᵢₘ₁ * Pᵢₘ₂' * ψₙₖ

  where n or ibnd is the band index, ``weights[ik ibnd] = kweights[ik] * occupation[ik, ibnd]``
  and ``Pᵢₘ₁`` is the pseudoatomic orbital projector for atom i and orbital m₁
  (usually just the magnetic quantum number, since l is usually fixed).
 For details on the projectors see `atomic_orbital_projectors`.

Overview of inputs:
- `manifold`: OrbitalManifold with the atomic orbital type to define the Hubbard manifold.
- `occupation`: Occupation matrix for the bands.
- `projectors` (kwarg): Vector of projection matrices. For each matrix, each column corresponds
                        to a different atomic orbital projector, as specified in labels.
- `labels` (kwarg): Vector of NamedTuples. Each projectors[ik][:,iproj] column has all relevant 
                    chemical information stored in the corresponding labels[iproj] NamedTuple.

Overviw of outputs:
- `nhubbard`: 3-tensor of matrices. Outer indices select spin, iatom and jatom,
    inner indices select m1 and m2 in the manifold.
"""
function compute_nhubbard(manifold::OrbitalManifold,
                          basis::PlaneWaveBasis{T},
                          ψ, occupation;
                          projectors, labels,
                          positions = basis.model.positions) where {T}
    filled_occ = filled_occupation(basis.model)
    nspins = basis.model.n_spin_components

    manifold_atoms = findall(at -> at.species==manifold.species, basis.model.atoms)
    natoms = length(manifold_atoms)  # Number of atoms of the species in the manifold
    l = labels[1].l
    projectors = reshape_hubbard_proj(basis, projectors, labels, manifold)
    nhubbard = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    for σ in 1:nspins
        for idx in 1:length(manifold_atoms), jdx in 1:length(manifold_atoms)
            nhubbard[σ, idx, jdx] = zeros(Complex{T}, 2*l+1, 2*l+1)
            for ik = krange_spin(basis, σ)
                j_projection = ψ[ik]' * projectors[ik][jdx] # <ψ|ϕJ>
                i_projection = projectors[ik][idx]' * ψ[ik] # <ϕI|ψ>
                # Sums over the bands, dividing by filled_occ to deal 
                # with the physical two spin channels separately
                nhubbard[σ, idx, jdx] .+= basis.kweights[ik] * (i_projection *
                                          (diagm(occupation[ik]/filled_occ) * j_projection))
            end
            nhubbard[σ, idx, jdx] = mpi_sum(nhubbard[σ, idx, jdx], basis.comm_kpts)
        end
    end
    nhubbard = symmetrize_nhubbard(nhubbard, basis.model,
                                   basis.symmetries, basis.model.positions[manifold_atoms])
end

function reshape_hubbard_proj(basis, projectors::Vector{Matrix{Complex{T}}}, 
                              labels, manifold) where {T}
    manifold_atoms = findall(at -> at.species==manifold.species, basis.model.atoms)
    natoms = length(manifold_atoms)
    nprojs = length(labels)
    l = labels[1].l
    @assert all(label -> label.l==l, labels)
    @assert length(labels) == natoms * (2*l+1) "Labels length error: 
                                                $(length(labels)) != $(natoms)*$(2*l+1)"
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for i in 1:length(projectors)]
    for (idx, iatom) in enumerate(manifold_atoms)
        for i in 1:2*l+1:nprojs
            if iatom == labels[i].iatom
                for (ik, projk) in enumerate(projectors)
                    p_I[ik][idx] = projk[:, i:i+2*l]
                end
            end
        end
    end

    return p_I
end

@doc raw"""
Hubbard energy:
```math
1/2 Σ_{σI} U * Tr[nhubbard[σ,i,i] * (1 - nhubbard[σ,i,i])]
```
"""
struct Hubbard{T}
    manifold::OrbitalManifold
    U::T
    function Hubbard(manifold::OrbitalManifold, U)
        if isnothing(manifold.label) || isnothing(manifold.species)
            error("Hubbard term needs both a species and a label inside OrbitalManifold")
        elseif !isnothing(manifold.iatom)
            error("Hubbard term does not support iatom specification inside OrbitalManifold")
        end
        U = austrip(U)
        new{typeof(U)}(manifold, U)
    end
end
function (hubbard::Hubbard)(basis::AbstractBasis)
    projs, labels = atomic_orbital_projectors(basis)
    labels, projectors_matrix = extract_manifold(basis, projs, labels, hubbard.manifold)
    TermHubbard(hubbard.manifold, hubbard.U, projectors_matrix, labels)
end

struct TermHubbard{T, PT, L} <: Term
    manifold::OrbitalManifold
    U::T     
    P::PT
    labels::L
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard,
                                            basis::PlaneWaveBasis{T},
                                            ψ, occupation; nhubbard=nothing,
                                            labels=term.labels,
                                            kwargs...) where {T}
    if isnothing(nhubbard)
       return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
    proj = term.P

    filled_occ = filled_occupation(basis.model)
    types = findall(at -> at.species==term.manifold.species, basis.model.atoms)
    natoms = length(types)
    nspins = basis.model.n_spin_components
    nproj_atom = size(nhubbard[1,1,1], 1) # This is the number of projectors per atom, namely 2l+1
    # For the ops we have to reshape nhubbard to match the NonlocalOperator structure, using a block diagonal form
    nhub = [zeros(Complex{T}, nproj_atom*natoms, nproj_atom*natoms) 
            for _ in 1:nspins]
    E = zero(T)
    for σ in 1:nspins, iatom in 1:natoms
        nhub[σ][1+nproj_atom*(iatom-1):nproj_atom*iatom, 1+nproj_atom*(iatom-1):nproj_atom*iatom] =
             nhubbard[σ, iatom, iatom]
        E += filled_occ * 1/2 * term.U *
             real(tr(nhubbard[σ, iatom, iatom] * (I - nhubbard[σ, iatom, iatom])))
    end
    ops = [NonlocalOperator(basis, kpt, proj[ik], 1/2 * term.U * (I - 2*nhub[kpt.spin])) 
           for (ik,kpt) in enumerate(basis.kpoints)]
    return (; E, ops)
end
