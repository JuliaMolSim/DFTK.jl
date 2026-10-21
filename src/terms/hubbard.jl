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
        # guaranteed by union
        @assert manifold.projectors isa @NamedTuple{l::Int, i::Int}
        (; l, i) = manifold.projectors
    end

    ResolvedOrbitalManifold(psp, iatoms, l, i)
end

function extract_manifold(manifold::ResolvedOrbitalManifold, projectors, labels)
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

@doc raw"""
Hubbard energy, following the Dudarev et al. (1998) rotationally invariant formalism:
```math
1/2 Σ_{σI} U * Tr[hubbard_n[σ,I,I] * (1 - hubbard_n[σ,I,I])]
```
"""
struct Hubbard{T}
    manifolds::Vector{OrbitalManifold}
    U::Vector{T}
    function Hubbard{T}(manifolds::Vector{OrbitalManifold}, U::Vector{T}) where {T}
        length(manifolds) == length(U) ||
            error("Number of U values ($(length(U))) must match "
                  * "number of manifolds ($(length(manifolds))).")
        new(manifolds, U)
    end
end
function Hubbard(manifolds::Vector{OrbitalManifold}, U::Vector{T}) where {T}
    U = austrip.(U)
    Hubbard{eltype(U)}(manifolds, U)
end
Hubbard() = Hubbard(OrbitalManifold[], Float64[])
function Hubbard(manifold_to_U::Vararg{T}) where {T <: Pair}
    Hubbard([m[1] for m in manifold_to_U], [m[2] for m in manifold_to_U])
end

function (hubbard::Hubbard{T})(basis::AbstractBasis) where {T}
    manifolds = [resolve_hubbard_manifold(manifold, basis.model) for manifold in hubbard.manifolds]
    projs, labels = atomic_orbital_projectors(basis)
    manifold_data = map(manifold -> extract_manifold(manifold, projs, labels), manifolds)
    manifold_labels = [data.manifold_labels for data in manifold_data]
    projs_matrices = [data.manifold_projectors for data in manifold_data]
    # Concatenate projectors from all manifolds per each k-point
    projs_per_kpt = map(1:length(basis.kpoints)) do ik
        reduce(hcat, map(P -> P[ik], projs_matrices))
    end
    TermHubbard(manifolds, hubbard.U, projs_matrices, projs_per_kpt, manifold_labels)
end

struct TermHubbard{T, PT, PV, L} <: Term
    manifolds::Vector{ResolvedOrbitalManifold}
    U::Vector{T}  # U values
    P::Vector{PT} # projectors per manifold
    P_vec::Vector{PV} # projectors per k-point (all manifolds concatenated)
    labels::Vector{L}
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard,
                                            basis::PlaneWaveBasis{T},
                                            ψ, occupation; hubbard_n=nothing,
                                            kwargs...) where {T}
    if isnothing(hubbard_n) || isempty(term.U)
       return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
   
    filled_occ = filled_occupation(basis.model)
    E = zero(T)
    for (iman, manifold) in enumerate(term.manifolds)
        for σ in 1:basis.model.n_spin_components, iatom in 1:length(manifold.iatoms)
            n_II = hubbard_n[iman][σ, iatom, iatom]
            E += filled_occ * 1/T(2) * term.U[iman] * real(tr(n_II * (I - n_II)))
        end
    end
    D = hubbard_potential_coefficients(term, basis, hubbard_n)
    ops = [NonlocalOperator(basis, kpt, term.P_vec[ik], D[kpt.spin])
           for (ik, kpt) in enumerate(basis.kpoints)]
    return (; E, ops)
end

"""
Coefficients `D[σ]` of the Hubbard potential `P D[σ] P'` with `P = term.P_vec[ik]`, i.e.
`U/2 (1 - 2 hubbard_n)`. To match the `NonlocalOperator` structure, `D[σ]` is block diagonal
with one block of size `2l+1` per atom per manifold.
"""
function hubbard_potential_coefficients(term::TermHubbard, basis::PlaneWaveBasis{T},
                                        hubbard_n) where {T}
    D_size = sum(manifold -> (2manifold.l + 1) * length(manifold.iatoms), term.manifolds)
    D = [zeros(Complex{T}, D_size, D_size) for _ in 1:basis.model.n_spin_components]
    offset = 0
    for (iman, manifold) in enumerate(term.manifolds)
        labels = term.labels[iman]
        for σ in 1:basis.model.n_spin_components, (idx, iatom) in enumerate(manifold.iatoms)
            n_II = hubbard_n[iman][σ, idx, idx]
            # The columns of P are in label order, not in the order of manifold.iatoms
            proj_range = offset .+ findall(label -> label.iatom == iatom, labels)
            D[σ][proj_range, proj_range] = 1/T(2) * term.U[iman] * (I - 2*n_II)
        end
        offset += length(labels)
    end
    D
end

@timing "forces: hubbard" function compute_forces(term::TermHubbard, basis::PlaneWaveBasis{TT},
                                                  ψ, occupation; hubbard_n, kwargs...) where {TT}
    T = promote_type(TT, real(eltype(ψ[1])))
    isempty(term.U) && return nothing
    # At fixed ψ, the energy depends on the positions only through hubbard_n, and
    #   force = -∂E/∂r = -Σ_k w_k Σ_n f_n 2Re⟨ψ_n|∂P D P'|ψ_n⟩
    # with D = U/2 (1 - 2 hubbard_n), the same structure as for the nonlocal term. The
    # projectors P are Löwdin-orthogonalized against the atomic orbitals of all atoms, so every
    # atom carrying orbitals feels a force.
    D = hubbard_potential_coefficients(term, basis, hubbard_n)
    hubbard_labels = reduce(vcat, term.labels)  # columns of term.P_vec

    # We compute the forces from the irreducible BZ; they are symmetrized later.
    (; dprojectors, labels, projections) = atomic_orbital_projector_derivatives(
        basis; ψ, selected=(label -> label in hubbard_labels))
    P_columns = [findfirst(isequal(label), labels) for label in hubbard_labels]

    forces = [zero(Vec3{T}) for _ in basis.model.positions]
    for (ik, kpt) in enumerate(basis.kpoints)
        Pψk = projections[ik][:, P_columns]'  # = term.P_vec[ik]' * ψ[ik]
        DPψk = D[kpt.spin] * Pψk
        for iatom in eachindex(forces)
            forces[iatom] -= map(1:3) do α
                ψdPk = dprojectors[ik][iatom][α][:, P_columns]  # ⟨ψ_n|∂P⟩
                dE_bands = 2 * real.(columnwise_dots(ψdPk', DPψk))
                basis.kweights[ik] * sum(occupation[ik] .* dE_bands)
            end
        end
    end
    mpi_sum!(forces, basis.comm_kpts)
    symmetrize_forces(basis, forces)
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
function compute_hubbard_n(manifold::ResolvedOrbitalManifold,
                           projectors, labels,
                           basis::PlaneWaveBasis{T},
                           ψ, occupation) where {T}
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
    symmetrize_hubbard_n(basis.model, manifold, hubbard_n; basis.symmetries)
end
function compute_hubbard_n(term::TermHubbard, basis::PlaneWaveBasis, ψ, occupation)
    [compute_hubbard_n(manifold, P, labels, basis, ψ, occupation)
     for (manifold, P, labels) in zip(term.manifolds, term.P, term.labels)]
end

"""
    compute_hubbard_n(basis, ψ, occupation)

`hubbard_n` of the Hubbard term of `basis`, or `nothing` if `basis` has no Hubbard term.
"""
function compute_hubbard_n(basis::PlaneWaveBasis, ψ, occupation)
    ihubbard = findfirst(term -> term isa TermHubbard, basis.terms)
    isnothing(ihubbard) && return nothing
    compute_hubbard_n(basis.terms[ihubbard], basis, ψ, occupation)
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
