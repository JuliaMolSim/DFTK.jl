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
function (s::OrbitalManifold)(orb)
    iatom_match    = isnothing(s.iatom)   || (s.iatom == orb.iatom)
    # See JuliaMolSim/AtomsBase.jl#139 why both species equalities are needed
    species_match  = isnothing(s.species) || (s.species == orb.species) || (orb.species == s.species)
    label_match    = isnothing(s.label)   || (s.label == orb.label)
    iatom_match && species_match && label_match
end

function extract_manifold(basis::PlaneWaveBasis{T}, projectors, labels,
                          manifold::OrbitalManifold) where {T}
    manifold_labels = filter(manifold, labels)
    isempty(manifold_labels) && @warn "Projector for $(manifold) not found."
    manifold_projectors = Vector{Matrix{Complex{T}}}(undef, length(basis.kpoints))
    for (ik, projk) in enumerate(projectors)
        manifold_projectors[ik] = zeros(Complex{T}, size(projectors[ik], 1), length(manifold_labels))
        iproj = 0
        for (proj_index, orb) in enumerate(labels)
            # Find the index of the projector that matches the manifold label
            if manifold(orb)
                iproj += 1
                manifold_projectors[ik][:, iproj] = projk[:, proj_index]
            end
        end
    end
    return (; manifold_labels, manifold_projectors)
end

"""
Symmetrize the Hubbard occupation matrix according to the l quantum number of the manifold.
"""
function symmetrize_nhubbard(nhubbard::Array{Matrix{Complex{T}}}, 
                         model, symmetries, positions) where {T}
    # For now we apply symmetries only on nII terms, not on cross-atom terms (nIJ)
    # WARNING: To implement +V this will need to be changed!

    nspins = size(nhubbard, 1)
    natoms = size(nhubbard, 2)
    nsym = length(symmetries)
    l = Int64((size(nhubbard[1, 1, 1], 1)-1)/2)
    ldim = 2*l+1

     # Initialize the nhubbard matrix
    ns = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    for σ in 1:nspins, iatom in 1:natoms, jatom in 1:natoms
        ns[σ, iatom, jatom] = zeros(Complex{T}, ldim, ldim)
    end
    for symmetry in symmetries
        Wcart = model.lattice * symmetry.W * model.inv_lattice
        WigD = wigner_d_matrix(l, Wcart)
        for σ in 1:nspins, iatom in 1:natoms
            sym_atom = find_symmetry_preimage(positions, positions[iatom], symmetry)
            ns[σ, iatom, iatom] .+= WigD' * nhubbard[σ, sym_atom, sym_atom] * WigD
        end
    end
    ns .= ns / nsym
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
- `positions`: Positions of the atoms in the unit cell. Default is model.positions.

Overviw of outputs:
- `nhubbard`: 3-tensor of matrices. Outer indices select spin, iatom and jatom,
    inner indices select m1 and m2 in the manifold.
- `manifold_labels`: Labels for all manifold orbitals, corresponding to different columns of p_I.
- `p_I`: Projectors for the manifold. Those are orthonormalized against all orbitals,
    also against those outside of the manifold.
"""
function compute_nhubbard(manifold::OrbitalManifold,
                          basis::PlaneWaveBasis{T},
                          ψ, occupation;
                          projectors, labels,
                          positions = basis.model.positions) where {T}
    filled_occ = filled_occupation(basis.model)
    nprojs = length(labels)
    nspins = basis.model.n_spin_components

    manifold_atoms = findall(at -> at.species==manifold.species, basis.model.atoms)
    natoms = length(manifold_atoms)  # Number of atoms of the species in the manifold
    l = labels[1].l
    nhubbard = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    for σ in 1:nspins, (idx, iatom) in enumerate(manifold_atoms), (jdx, jatom) in enumerate(manifold_atoms)
        nhubbard[σ, idx, jdx] = zeros(Complex{T}, 2*l+1, 2*l+1)
        for ik = krange_spin(basis, σ)
            # We divide by filled_occ to deal with the physical two spin channels separately.
            j_projection = ψ[ik]' * projectors[ik][jdx] # <ψ|ϕJ>
            i_projection = projectors[ik][idx]' * ψ[ik] # <ϕI|ψ>
            # Sums over the bands
            nhubbard[σ, idx, jdx] .+= basis.kweights[ik] * i_projection *
                                          diagm(occupation[ik]/filled_occ) * j_projection
        end
    end
    nhubbard = symmetrize_nhubbard(nhubbard, basis.model,
                               basis.symmetries, basis.model.positions[manifold_atoms])

    return (; nhubbard, manifold_labels=labels)
end

function reshape_hubbard_proj(basis, projectors::Vector{Matrix{Complex{T}}}, 
                              labels, manifold) where {T}
    manifold_atoms = findall(at -> at.species==manifold.species, basis.model.atoms)
    natoms = length(manifold_atoms)
    nprojs = length(labels)
    l = labels[1].l
    @assert all(label -> label.l==l, labels)
    @assert length(labels) == natoms * (2*l+1)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for i in 1:length(projectors)]
    for (idx, iatom) in enumerate(manifold_atoms)
        for i in 1:2*l+1:nprojs
            if iatom != labels[i].iatom
                continue
            else
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
struct Hubbard
    manifold :: OrbitalManifold
    U        
end
function (hubbard::Hubbard)(basis::AbstractBasis)
    if isnothing(hubbard.manifold.label) || isnothing(hubbard.manifold.species)
        error("TermHubbard needs both a species and a label inside OrbitalManifold")
    end
    projs, labels = atomic_orbital_projectors(basis)
    labels, projectors = extract_manifold(basis, projs, labels, hubbard.manifold)
    projectors = reshape_hubbard_proj(basis, projectors, labels, hubbard.manifold)
    U = austrip(hubbard.U)
    TermHubbard(hubbard.manifold, U, projectors, labels)
end

struct TermHubbard{PT, L} <: Term
    manifold :: OrbitalManifold
    U        
    P        :: PT
    labels   :: L
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard,
                                            basis::PlaneWaveBasis{T},
                                            ψ, occupation; nhubbard=nothing,
                                            labels=term.labels,
                                            kwargs...) where {T}
    if isnothing(ψ)
        if isnothing(nhubbard)
           return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
        end
    else
        nhubbard = compute_nhubbard(term.manifold, basis, ψ, occupation; projectors=term.P, 
                                    labels).nhubbard
    end
    proj = term.P

    ops = [HubbardUOperator(basis, kpt, term.U, nhubbard, proj[ik]) 
           for (ik,kpt) in enumerate(basis.kpoints)]
    filled_occ = filled_occupation(basis.model)
    types = findall(at -> at.species==term.manifold.species, basis.model.atoms)
    natoms = length(types)
    nspins = basis.model.n_spin_components
    E = zero(T)
    for σ in 1:nspins, iatom in 1:natoms
        E += filled_occ * 1/2 * term.U * 
             real(tr(nhubbard[σ, iatom, iatom] * (I - nhubbard[σ, iatom, iatom])))
    end
    return (; E, ops)
end
