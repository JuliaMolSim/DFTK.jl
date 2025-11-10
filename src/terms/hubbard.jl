using LinearAlgebra
using Random

"""
Structure for Hubbard manifold choice and projectors extraction.
  It is to be noted that, despite the name used in literature, this is 
  not a manifold in the mathematical sense.

Overview of fields:
- `iatom`: Atom position in the atoms array.
- `species`: Chemical Element as in ElementPsp.
- `label`: Orbital name, e.g.: "3S".

All fields are optional, only the given ones will be used for selection.
Can be called with an orbital NamedTuple and returns a boolean
  stating whether the orbital belongs to the manifold.
"""
@kwdef struct OrbitalManifold
    iatoms::Union{Vector{Int64},  Nothing} = nothing
    psp = nothing
    projector_l::Union{Int64, Nothing} = nothing
    projector_i::Union{Int64, Nothing} = nothing
end
function is_on_manifold(orbital; iatoms=nothing, species=nothing, 
                                 l=nothing, n=nothing, label=nothing)
    iatom_match   = isnothing(iatoms)  || (orbital.iatom in iatoms)
    species_match = isnothing(species) || (species == orbital.species)
    label_match   = isnothing(label)   || (label == orbital.label)
    l_match = isnothing(l) || (l == orbital.l)
    n_match = isnothing(n) || (n == orbital.n)
    iatom_match && species_match && l_match && n_match && label_match
end
function is_on_manifold(orbital, manifold::OrbitalManifold)
    is_on_manifold(orbital; iatoms=manifold.iatoms, l=manifold.projector_l, n=manifold.projector_i)
end

function OrbitalManifold(atoms, labels; iatoms::Union{Vector{Int64}, Nothing}=nothing,
                         label::Union{String, Nothing}=nothing,
                         species::Union{Symbol, AtomsBase.ChemicalSpecies, Nothing}=nothing)
    hub_atoms = Int64[]
    for orbital in labels
        if is_on_manifold(orbital; iatoms, species, label) 
            append!(hub_atoms, orbital.iatom)
        end
    end
    isempty(hub_atoms) && error("Unable to create Hubbard manifold. No atom matches the given keywords")
    # If species is nothing, there can be errors if the iatoms correspond to different atomic species
    model_atom = atoms[hub_atoms[1]]
    !all(atom -> atom.psp == model_atom.psp, atoms[hub_atoms]) && 
        error("The given Hubbard manifold contains more than one atomic pseudopotential species")
    projector_l = isnothing(label) ? nothing : labels[hub_atoms[1]].l
    projector_i = isnothing(label) ? nothing : labels[hub_atoms[1]].n
    OrbitalManifold(;iatoms=hub_atoms, psp=model_atom.psp, projector_l, projector_i)
end

function extract_manifold(basis::PlaneWaveBasis{T}, projectors, labels,
                          manifold::OrbitalManifold) where {T}
    # We extract the labels only for orbitals belonging to the manifold
    proj_indices = findall(orb -> is_on_manifold(orb, manifold), labels)
    isempty(proj_indices) && @warn "Projector for $(manifold) not found."
    manifold_labels = labels[proj_indices]
    manifold_projectors = map(enumerate(projectors)) do (ik, projk)
        projk[:, proj_indices]
    end
    (; manifold_labels, manifold_projectors)
end

"""
    compute_hubbard_n(manifold, basis, ψ, occupation; [projectors, labels, positions])

Computes a matrix hubbard_n of size (n_spin, natoms, natoms), where each entry hubbard_n[iatom, jatom]
  contains the submatrix of the occupation matrix corresponding to the projectors
  of atom iatom and atom jatom, with dimensions determined by the number of projectors for each atom.
  The atoms and orbitals are defined by the manifold tuple.

    hubbard_n[σ, iatom, jatom][m1, m2] = Σₖ₍ₛₚᵢₙ₎Σₙ weights[ik, ibnd] * ψₙₖ' * Pᵢₘ₁ * Pᵢₘ₂' * ψₙₖ

  where n or ibnd is the band index, ``weights[ik ibnd] = kweights[ik] * occupation[ik, ibnd]``
  and ``Pᵢₘ₁`` is the pseudoatomic orbital projector for atom i and orbital m₁
  (just the magnetic quantum number, since l is fixed, as is usual in the literature).
 For details on the projectors see `atomic_orbital_projectors`.

Overview of inputs:
- `manifold`: OrbitalManifold with the atomic orbital type to define the Hubbard manifold.
- `occupation`: Occupation matrix for the bands.
- `projectors` (kwarg): Vector of projection matrices. For each matrix, each column corresponds
                        to a different atomic orbital projector, as specified in labels.
- `labels` (kwarg): Vector of NamedTuples. Each projectors[ik][:,iproj] column has all relevant 
                    chemical information stored in the corresponding labels[iproj] NamedTuple.

Overview of outputs:
- `hubbard_n`: 3-tensor of matrices. See above for details.
"""
function compute_hubbard_n(manifold::OrbitalManifold,
                          basis::PlaneWaveBasis{T},
                          ψ, occupation;
                          projectors, labels,
                          positions = basis.model.positions) where {T}
    filled_occ = filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components

    manifold_atoms = findall(at -> at.psp == manifold.psp, basis.model.atoms)
    natoms = length(manifold_atoms)  # Number of atoms of the species in the manifold
    l = labels[1].l
    projectors = reshape_hubbard_proj(basis, projectors, labels, manifold)
    hubbard_n = Array{Matrix{Complex{T}}}(undef, n_spin, natoms, natoms)
    for σ in 1:n_spin
        for idx in 1:length(manifold_atoms), jdx in 1:length(manifold_atoms)
            hubbard_n[σ, idx, jdx] = zeros(Complex{T}, 2*l+1, 2*l+1)
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
    hubbard_n = symmetrize_hubbard_n(hubbard_n, basis.model,
                                   basis.symmetries, basis.model.positions[manifold_atoms])
end

"""
This function reshapes for each kpoint the projectors matrix to a vector of matrices,
    taking only the columns corresponding to orbitals in the manifold and splitting them
    into different matrices, one for each atom. Columns in the same matrix differ only in 
    the value of the magnetic quantum number m of the corresponding orbitals.
"""
function reshape_hubbard_proj(basis, projectors::Vector{Matrix{Complex{T}}}, 
                              labels, manifold) where {T}
    manifold_atoms = findall(at -> at.psp == manifold.psp, basis.model.atoms)
    natoms = length(manifold_atoms)
    nprojs = length(labels)
    l = labels[1].l
    @assert all(label -> label.l==l, labels) "$(labels)"
    @assert length(labels) == natoms * (2*l+1)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for i in 1:length(projectors)]
    for (idx, iatom) in enumerate(manifold_atoms)
        for i in 1:2*l+1:nprojs
            iatom != labels[i].iatom && continue
            for (ik, projk) in enumerate(projectors)
                p_I[ik][idx] = projk[:, i:i+2*l]
            end
        end
    end

    p_I
end

@doc raw"""
Hubbard energy:
```math
1/2 Σ_{σI} U * Tr[hubbard_n[σ,i,i] * (1 - hubbard_n[σ,i,i])]
```
"""
struct Hubbard{T}
    manifold::OrbitalManifold
    U::T
    function Hubbard((manifold::OrbitalManifold), U)
        if isnothing(manifold.iatoms) || isnothing(manifold.projector_l) ||
           isnothing(manifold.projector_i)
            error("Hubbard term needs specification of atoms and orbital")
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
                                            ψ, occupation; hubbard_n=nothing,
                                            labels=term.labels,
                                            kwargs...) where {T}
    if isnothing(hubbard_n)
       return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
    proj = term.P

    filled_occ = filled_occupation(basis.model)
    types = findall(at -> at.psp == term.manifold.psp, basis.model.atoms)
    natoms = length(types)
    n_spin = basis.model.n_spin_components
    nproj_atom = size(hubbard_n[1,1,1], 1) # This is the number of projectors per atom, namely 2l+1
    # For the ops we have to reshape hubbard_n to match the NonlocalOperator structure, using a block diagonal form
    nhub = [zeros(Complex{T}, nproj_atom*natoms, nproj_atom*natoms) for _ in 1:n_spin]
    E = zero(T)
    for σ in 1:n_spin, iatom in 1:natoms
        proj_range = (1+nproj_atom*(iatom-1)):(nproj_atom*iatom)
        nhub[σ][proj_range, proj_range] = hubbard_n[σ, iatom, iatom]
        E += filled_occ * 1/T(2) * term.U *
             real(tr(hubbard_n[σ, iatom, iatom] * (I - hubbard_n[σ, iatom, iatom])))
    end
    ops = [NonlocalOperator(basis, kpt, proj[ik], 1/T(2) * term.U * (I - 2*nhub[kpt.spin])) 
           for (ik,kpt) in enumerate(basis.kpoints)]
    return (; E, ops)
end
