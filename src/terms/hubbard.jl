using LinearAlgebra
using Random

"""
Structure for Hubbard manifold choice and projectors extraction.

"Manifold" is the standard name used in the literature to refer
to the set of atomic orbitals used to compute the Hubbard correction.
It is to be noted that this is not meant in the mathematical sense of "manifold".

The manifold contains two pieces of information:
- The atomic sites at which the correction is applied.
  These can either be specified as a chemical species for convenience,
  or as a list of atom indices for maximum flexibility.
- The pseudoatomic orbitals to use for the projectors.
  These can either be specified by their label (e.g. `"3D"` for PseudoDojo),
  or by their angular momentum number `l` + the corresponding index `i`.
"""
struct OrbitalManifold
    atoms::Union{ChemicalSpecies, Vector{Int}}
    projectors::Union{AbstractString, @NamedTuple{l::Int, i::Int}}
end
function OrbitalManifold(atom::ElementPsp, projectors)
    OrbitalManifold(atom.species, projectors)
end
function OrbitalManifold(atom::Symbol, projectors)
    OrbitalManifold(ChemicalSpecies(atom), projectors)
end

"""
DFTK-internal version of [`OrbitalManifold`](@ref)
with resolved atom indices and projectors.

This separation allows the manifold to be defined
without requiring access to the pseudopotential.
"""
struct ResolvedOrbitalManifold
    psp::NormConservingPsp
    iatoms::Vector{Int}
    l::Int
    i::Int
end

function resolve_hubbard_manifold(manifold::OrbitalManifold, model::Model)
    if manifold.atoms isa ChemicalSpecies
        iatoms = findall(at -> species(at) == manifold.atoms, model.atoms)
    else
        # guaranteed by union
        @assert manifold.atoms isa Vector{Int}
        iatoms = manifold.atoms
    end
    isempty(iatoms) && error("Orbital manifold has no atoms.")

    atom = first(model.atoms[iatoms])
    atom isa ElementPsp || error("Orbital manifold elements must have a psp.")
    psp = atom.psp
    for atom in model.atoms[iatoms]
        atom isa ElementPsp || error("Orbital manifold elements must have a psp.")
        atom.psp === psp || error("Orbital manifold contains multiple psps: " *
                                  "$(psp.identifier) and $(atom.psp.identifier)")
    end
    # Tricky: make sure that the iatoms are consistent with the symmetries of the model,
    #         i.e. that the manifold would be a valid atom group
    atom_positions = model.positions[iatoms]
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

    if manifold.projectors isa AbstractString
        l, i = find_pswfc(psp, manifold.projectors)
    else
        (; l, i) = manifold.projectors
    end

    ResolvedOrbitalManifold(psp, iatoms, l, i)
end

function extract_manifold(manifold::ResolvedOrbitalManifold, projectors, labels)
    # We extract the labels only for orbitals belonging to the manifold
    proj_indices = findall(orb -> (orb.iatom in manifold.iatoms
                                && orb.l     == manifold.l
                                && orb.n     == manifold.i), labels)
    isempty(proj_indices) && @warn "Projector for $(manifold) not found."
    manifold_labels = labels[proj_indices]
    manifold_projectors = map(enumerate(projectors)) do (ik, projk)
        projk[:, proj_indices]
    end
    (; manifold_labels, manifold_projectors)
end

@doc raw"""
Hubbard energy, following the Dudarev et al. (1998) rotationally invariant formalism:
```math
1/2 Σ_{σI} U * Tr[hubbard_n[σ,I,I] * (1 - hubbard_n[σ,I,I])]
```
"""
struct Hubbard{T}
    manifold::OrbitalManifold
    U::T
    function Hubbard(manifold::OrbitalManifold, U)
        U = austrip(U)
        new{typeof(U)}(manifold, U)
    end
end
function (hubbard::Hubbard)(basis::AbstractBasis)
    manifold = resolve_hubbard_manifold(hubbard.manifold, basis.model)

    projs, labels = atomic_orbital_projectors(basis)
    labels, projectors_matrix = extract_manifold(manifold, projs, labels)
    TermHubbard(manifold, hubbard.U, projectors_matrix, labels)
end

struct TermHubbard{T, PT, L} <: Term
    manifold::ResolvedOrbitalManifold
    U::T   # U value
    P::PT  # projectors
    labels::L
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard,
                                            basis::PlaneWaveBasis{T},
                                            ψ, occupation; hubbard_n=nothing,
                                            kwargs...) where {T}
    if isnothing(hubbard_n)
       return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
    proj = term.P

    filled_occ = filled_occupation(basis.model)
    natoms = length(term.manifold.iatoms)
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

"""
    compute_hubbard_n(term::TermHubbard, basis, ψ, occupation)

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
function compute_hubbard_n(manifold::ResolvedOrbitalManifold, projectors, labels,
                           basis::PlaneWaveBasis{T}, ψ, occupation) where {T}
    filled_occ = filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components

    manifold_atoms = manifold.iatoms
    natoms = length(manifold_atoms)
    l = manifold.l
    projectors = reshape_hubbard_proj(projectors, labels, manifold)
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
end
function compute_hubbard_n(term::TermHubbard, basis::PlaneWaveBasis, ψ, occupation)
    compute_hubbard_n(term.manifold, term.P, term.labels, basis, ψ, occupation)
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
