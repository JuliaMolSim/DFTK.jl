using LinearAlgebra

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
        if Symbol(manifold.species) == Symbol(orb.species) && lowercase(manifold.label) == lowercase(orb.label)
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

function compute_overlap_matrix(basis::PlaneWaveBasis{T};
                                manifold  = nothing,
                                positions = basis.model.positions) where {T}
    proj = atomic_orbital_projectors(basis; manifold, positions)
    projectors = proj.projectors
    labels = proj.labels
    overlap_matrix = Vector{Matrix{T}}(undef, length(basis.kpoints))

    for (ik, projk) in enumerate(projectors)
        overlap_matrix[ik] = abs2.(projk' * projk)
    end

    return (; overlap_matrix, projectors, labels)
end

"""
Symmetrize the Hubbard occupation matrix according to the l quantum number of the manifold.
"""
function symmetrize_nhub(n_IJ::Array{Matrix{Complex{T}}}, lattice, symmetry, positions) where {T}
    # For now we apply symmetries only on nII terms, not on cross-atom terms (nIJ)
    # WARNING: To implement +V this will need to be changed!

    nspins = size(n_IJ, 1)
    natoms = size(n_IJ, 2)
    nsym = length(symmetry)
    l = Int64((size(n_IJ[1, 1, 1], 1)-1)/2)
    WigD = Wigner_sym(l, lattice, symmetry)

     # Initialize the n_IJ matrix
    ns = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    for σ in 1:nspins, iatom in 1:natoms, jatom in 1:natoms
        ns[σ, iatom, jatom] = zeros(Complex{T},
                                    size(n_IJ[σ, iatom, jatom], 1),
                                    size(n_IJ[σ, iatom, jatom], 2))
    end

    for σ in 1:nspins, iatom in 1:natoms, isym in 1:nsym
        for m1 in 1:size(ns[σ, iatom, iatom], 1), m2 in 1:size(ns[σ, iatom, iatom], 2)
            sym_atom = find_symmetric(iatom, symmetry, isym, positions)
            # TODO: Here QE flips spin for time-reversal in collinear systems, should we?
            for m0 in 1:size(n_IJ[σ, iatom, iatom], 1), m00 in 1:size(n_IJ[σ, iatom, iatom], 2)
                ns[σ, iatom, iatom][m1, m2] += WigD[m0, m1, isym] *
                                               n_IJ[σ, sym_atom, sym_atom][m0, m00] *
                                               WigD[m00, m2, isym]
            end
        end
    end
    ns .= ns / nsym

    return ns
end

"""
Find the symmetric atom index for a given atom and symmetry operation
"""
function find_symmetric(iatom::Int64, symmetry::Vector{SymOp{T}},
                        isym::Int64, positions) where {T}
    sym_atom = iatom
    W, w = symmetry[isym].W, symmetry[isym].w
    p = positions[iatom]
    p2 = W * p + w
    for (jatom, pos) in enumerate(positions)
        if isapprox(pos, p2, atol=1e-8)
            sym_atom = jatom
            break
        end
    end
    return sym_atom
end

"""
 This function returns the Wigner matrix for a given l and symmetry operation
    solving a randomized linear system.
    The lattice L is needed to convert reduced symmetries to Cartesian space.
"""
function Wigner_sym(l::Int64, L, symmetries::Vector{SymOp{T}}) where {T}
    nsym = length(symmetries)
    D = Array{Float64}(undef, 2*l+1, 2*l+1, nsym)
    if l == 0
        return D .= 1
    end
    for (isym, symmetry) in enumerate(symmetries)
        W = symmetry.W
        for m1 in -l:l
            b = Vector{Float64}(undef, 2*l+1)
            A = Matrix{Float64}(undef, 2*l+1, 2*l+1)
            for n in 1:2*l+1
                r = rand(Float64, 3)
                r0 = L * W * inv(L) * r
                b[n] = DFTK.ylm_real(l, m1, r0)
                for m2 in -l:l
                    A[n,m2+l+1] = DFTK.ylm_real(l, m2, r)
                end
            end
            D[m1+l+1,:,isym] = A\b
        end
    end

    return D
end

"""
    compute_hubbard_nIJ(manifold, basis, ψ, occupation; [projectors, labels, positions])

Computes a matrix nᴵᴶ of size (nspins, natoms, natoms), where each entry nᴵᴶ[iatom, jatom]
  contains the submatrix of the occupation matrix corresponding to the projectors
  of atom iatom and atom jatom, with dimensions determined by the number of projectors for each atom.
  The atoms and orbitals are defined by the manifold tuple.

Overview of inputs:
- `manifold`: OrbitalManifold with the atomic orbital type to define the Hubbard manifold.
- `occupation`: Occupation matrix for the bands.
- `positions`: Positions of the atoms in the unit cell. Default is model.positions.

Overviw of outputs:
- `n_IJ`: 3-tensor of matrices. Outer indices select spin, iatom and jatom,
    inner indices select m1 and m2 in the manifold.
- `manifold_labels`: Labels for all manifold orbitals, corresponding to different columns of p_I.
- `p_I`: Projectors for the manifold. Those are orthonormalized against all orbitals,
    also against those outside of the manifold.
"""
function compute_hubbard_nIJ(manifold::OrbitalManifold,
                             basis::PlaneWaveBasis{T},
                             ψ, occupation;
                             projectors = nothing, labels = nothing,
                             positions = basis.model.positions) where {T}
    for (iatom, atom) in enumerate(basis.model.atoms)
        @assert !iszero(size(atom.psp.r2_pswfcs[1], 1)) "FATAL ERROR: No Atomic projector found within the provided PseudoPotential."
    end

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
    # Reshape into n[I, J, σ][m1, m2] where I, J indicate the atom in the Hubbard manifold, σ is the spin, m1 and m2 are magnetic quantum numbers (n, l are fixed)
    manifold_atoms = findall(at -> at.species == Symbol(manifold.species), basis.model.atoms)
    natoms = length(manifold_atoms)  # Number of atoms of the species in the manifold
    n_IJ = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
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
                    (manifold_atoms[jatom] == labels[j].iatom) && (n_IJ[σ, iatom, jatom] = n_matrix[σ, i:i+2*il, j:j+2*jl])
                    j += 2*jl + 1
                end
            end
            for (ik, projk) in enumerate(projectors)
                p_I[ik][iatom] = projk[:, i:i+2*il]  
            end
            i += 2*il + 1
        end
    end

    n_IJ = symmetrize_nhub(n_IJ, basis.model.lattice, basis.symmetries, basis.model.positions)

    return (; n_IJ=n_IJ, manifold_labels=labels, p_I=p_I)
end

function reshape_hubbard_proj(projectors::Vector{Matrix{Complex{T}}}, labels) where {T}
    natoms = max([labels[i].iatom for i in 1:length(labels)]...)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for i in 1:length(projectors)]
    i = 1
    while i <= size(projectors, 2)
        il = labels[i].l
        iatom = labels[i].iatom
        for (ik, projk) in enumerate(projectors)
            p_I[ik][iatom] = Matrix{Complex{T}}(undef, size(projk,1), 2*il + 1)
            p_I[ik][iatom] = copy(projk[:, i:i+2*il])
        end
        i += 2*il + 1
    end

    return p_I
end

struct Hubbard
    manifold :: OrbitalManifold
    U        :: Float64
end
function (hubbard::Hubbard)(basis::AbstractBasis)
    isempty(hubbard.U) && return TermNoop()
    projs, labs = atomic_orbital_projectors(basis)
    labels, projectors = extract_manifold(basis, projs, labs, hubbard.manifold)
    TermHubbard(hubbard.manifold, hubbard.U, projectors, labels)
end

struct TermHubbard{PT, L} <: Term
    manifold :: OrbitalManifold
    U        :: Float64
    P        :: PT
    labels   :: L
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard,
                                            basis::PlaneWaveBasis{T},
                                            ψ, occupation; n_hub=nothing, ψ_hub=nothing,
                                            labels=term.labels,
                                            kwargs...) where {T}
    if isnothing(ψ)
        if isnothing(n_hub)
           return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
        end
        n = n_hub
        ψ = ψ_hub
        proj = reshape_hubbard_proj(term.P, term.labels)
    else
        Hubbard = compute_hubbard_nIJ(term.manifold, basis, ψ, occupation; projectors=term.P, 
        labels)
        n = Hubbard.n_IJ
        n_hub = n
        proj = Hubbard.p_I
    end
    to_unit = ustrip(auconvert(u"eV", 1.0))
    U = term.U / to_unit  # U is usally expressed in eV in the literature, but we work in au

    ops = [HubbardUOperator(basis, kpt, U, n, proj[ik]) for (ik,kpt) in enumerate(basis.kpoints)]

    filled_occ = filled_occupation(basis.model)
    types = findall(at -> at.species == Symbol(term.manifold.species), basis.model.atoms)
    natoms = length(types)
    nspins = basis.model.n_spin_components

    E = zero(T)
    for σ in 1:nspins, iatom in 1:natoms
        E += filled_occ * 0.5 * U * real(tr(n[σ, iatom,iatom] * (I - n[σ, iatom,iatom])))
    end
    return (; E, ops, n)
end
