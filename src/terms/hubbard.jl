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
    manifold_labels = []
    manifold_projectors = Vector{Matrix{Complex{T}}}(undef, length(basis.kpoints))
    for (iproj, orb) in enumerate(labels)
        if Symbol(manifold.species) == Symbol(orb.species) && 
           lowercase(manifold.label) == lowercase(orb.label)
            # If the label matches the manifold, we add it to the labels
            # This is useful for extracting specific orbitals from the basis
            # e.g., (:Si, "3S") will match all 3S orbitals of Si atoms
            push!(manifold_labels, (; orb.iatom, orb.species, orb.n, orb.l, orb.m, orb.label))
        end
    end
    for (ik, projk) in enumerate(projectors)
        manifold_projectors[ik] = zeros(Complex{T}, size(projectors[ik], 1), length(manifold_labels))
        for (iproj, orb) in enumerate(manifold_labels)
            # Find the index of the projector that matches the manifold label
            proj_index = findfirst(p -> p.iatom == orb.iatom && p.species == orb.species &&
                                       p.n == orb.n && p.l == orb.l && p.m == orb.m, labels)
            if proj_index !== nothing
                manifold_projectors[ik][:, iproj] = projk[:, proj_index]
            else
                @warn "Projector for $(orb.species) with n=$(orb.n), l=$(orb.l), m=$(orb.m)
                       not found in projectors."
            end
        end
    end
    return (; manifold_labels, manifold_projectors)
end

"""
Symmetrize the Hubbard occupation matrix according to the l quantum number of the manifold.
"""
function symmetrize_nhub(nhubbard::Array{Matrix{Complex{T}}}, lattice, symmetries, positions) where {T}
    # For now we apply symmetries only on nII terms, not on cross-atom terms (nIJ)
    # WARNING: To implement +V this will need to be changed!

    nspins = size(nhubbard, 1)
    natoms = size(nhubbard, 2)
    nsym = length(symmetries)
    l = Int64((size(nhubbard[1, 1, 1], 1)-1)/2)

     # Initialize the nhubbard matrix
    ns = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    for σ in 1:nspins, iatom in 1:natoms, jatom in 1:natoms
        ns[σ, iatom, jatom] = zeros(Complex{T},
                                    size(nhubbard[σ, iatom, jatom], 1),
                                    size(nhubbard[σ, iatom, jatom], 2))
    end

    for symmetry in symmetries
        Wcart = lattice * symmetry.W * inv(lattice)
        WigD = wigner_d_matrix(l, Wcart)
        for σ in 1:nspins, iatom in 1:natoms
            sym_atom = find_symmetry_preimage(positions, positions[iatom], symmetry)
            for m1 in 1:size(ns[σ, iatom, iatom], 1), m2 in 1:size(ns[σ, iatom, iatom], 2)
                # TODO: Here QE flips spin for time-reversal in collinear systems, should we?
                for m0 in 1:size(nhubbard[σ, iatom, iatom], 1), m00 in 1:size(nhubbard[σ, iatom, iatom], 2)
                    ns[σ, iatom, iatom][m1, m2] += WigD[m0, m1] *
                                                   nhubbard[σ, sym_atom, sym_atom][m0, m00] *
                                                   WigD[m00, m2]
                end
            end
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
    n_matrix = zeros(Complex{T}, nspins, nprojs, nprojs) 

    for σ in 1:nspins, ik = krange_spin(basis, σ)  
        # We divide by filled_occ to deal with the physical two spin channels separately.
        ψk, projk, nk = @views ψ[ik], projectors[ik], occupation[ik]/filled_occ  
        c = projk' * ψk      # <ϕ|ψ>
        # The matrix product is done over the bands. In QE, basis.kweights[ik]*nk[ibnd] would be wg(ik,ibnd)
        n_matrix[σ, :, :] .+= basis.kweights[ik] * c * diagm(nk) * c' 
    end
    n_matrix = mpi_sum(n_matrix, basis.comm_kpts)

    # Now I want to reshape it to match the notation used in the papers.
    # Reshape into n[I, J, σ][m1, m2] where I, J indicate the atom in the Hubbard manifold, σ is the spin, 
    # m1 and m2 are magnetic quantum numbers (n, l are fixed)
    manifold_atoms = findall(at -> at.species == Symbol(manifold.species), basis.model.atoms)
    natoms = length(manifold_atoms)  # Number of atoms of the species in the manifold
    nhubbard = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for ik in 1:length(basis.kpoints)]
    # Very low-level, but works
    for σ in 1:nspins, iatom in eachindex(manifold_atoms)
        i = 1
        while i <= nprojs
            il = labels[i].l
            if !(manifold_atoms[iatom] == labels[i].iatom)
                i += 2*il + 1
                continue
            end
            for jatom in eachindex(manifold_atoms)
                j = 1
                while j <= nprojs
                    jl = labels[j].l
                    (manifold_atoms[jatom] == labels[j].iatom) && 
                      (nhubbard[σ, iatom, jatom] = n_matrix[σ, i:i+2*il, j:j+2*jl])
                    j += 2*jl + 1
                end
            end
            for (ik, projk) in enumerate(projectors)
                p_I[ik][iatom] = projk[:, i:i+2*il]  
            end
            i += 2*il + 1
        end
    end

    nhubbard = symmetrize_nhub(nhubbard, basis.model.lattice, 
                               basis.symmetries, basis.model.positions[manifold_atoms])

    return (; nhubbard, manifold_labels=labels, p_I)
end

function reshape_hubbard_proj(basis, projectors::Vector{Matrix{Complex{T}}}, labels, manifold) where {T}
    manifold_atoms = findall(at -> at.species == Symbol(manifold.species), basis.model.atoms)
    natoms = length(manifold_atoms)
    nprojs = length(labels)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for i in 1:length(projectors)]
    for iatom in eachindex(manifold_atoms)
        i = 1
        while i <= nprojs
            il = labels[i].l
            if !(manifold_atoms[iatom] == labels[i].iatom)
                i += 2*il + 1
                continue
            end
            for (ik, projk) in enumerate(projectors)
                p_I[ik][iatom] = projk[:, i:i+2*il]  
            end
            i += 2*il + 1
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
    iszero(hubbard.U) && return TermNoop()
    projs, labs = atomic_orbital_projectors(basis)
    labels, projectors = extract_manifold(basis, projs, labs, hubbard.manifold)
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
                                            ψ, occupation; nhubbard=nothing, ψ_hub=nothing,
                                            labels=term.labels,
                                            kwargs...) where {T}
    if isnothing(ψ)
        if isnothing(nhubbard)
           return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
        end
        ψ = ψ_hub
        proj = reshape_hubbard_proj(basis, term.P, term.labels, term.manifold)
    else
        Hubbard = compute_nhubbard(term.manifold, basis, ψ, occupation; projectors=term.P, 
        labels)
        nhubbard = Hubbard.nhubbard
        proj = Hubbard.p_I
    end

    ops = [HubbardUOperator(basis, kpt, term.U, nhubbard, proj[ik]) 
           for (ik,kpt) in enumerate(basis.kpoints)]

    filled_occ = filled_occupation(basis.model)
    types = findall(at -> at.species == Symbol(term.manifold.species), basis.model.atoms)
    natoms = length(types)
    nspins = basis.model.n_spin_components

    E = zero(T)
    for σ in 1:nspins, iatom in 1:natoms
        E += filled_occ * 1/2 * term.U * 
             real(tr(nhubbard[σ, iatom, iatom] * (I - nhubbard[σ, iatom, iatom])))
    end
    return (; E, ops, nhubbard)
end
