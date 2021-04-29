## This file contains functions to handle the symetries

# A symmetry operation (symop) is a couple (Stilde, τtilde) of a
# unitary (in cartesian coordinates, but not in reduced coordinates)
# matrix Stilde and a translation τtilde such that, for each atom of
# type A at position a, Stilde a + τtilde is also an atom of type A.

# This induces a symmetry in the Brillouin zone that the Hamiltonian
# at S k is unitary equivalent to that at k, which we exploit to
# reduce computations. The relationship is
# S = Stilde'
# τ = -Stilde^-1 τtilde
# (valid both in reduced and cartesian coordinates)

# The full (reducible) Brillouin zone is implicitly represented by
# a set of (irreducible) kpoints (see explanation in docs). Each
# irreducible kpoint k comes with a list of symmetry operations
# (S,τ) (containing at least the trivial operation (I,0)), where S
# is a rotation matrix (/!\ not unitary in reduced coordinates)
# and τ a translation vector. The kpoint is then used to represent
# implicitly the information at all the kpoints Sk. The
# relationship between the Hamiltonians is
# H_{Sk} = U H_k U*, with
# (Uu)(x) = u(S^-1(x-τ))
# or in Fourier space
# (Uu)(G) = e^{-i G τ} u(S^-1 G)
# In particular, we can choose the eigenvectors at Sk as u_{Sk} = U u_k

# We represent then the BZ as a set of irreducible points `kpoints`, a
# list of symmetry operations `ksymops` allowing the reconstruction of
# the full (reducible) BZ, and a set of weights `kweights` (summing to
# 1). The value of an observable (eg energy) per unit cell is given as
# the value of that observable at each irreducible kpoint, weighted by
# kweight

# There is by decreasing cardinality
# - The group of symmetry operations of the lattice
# - The group of symmetry operations of the crystal (model.symmetries)
# - The group of symmetry operations of the crystal that preserves the BZ mesh (basis.symmetries)
# - The set of symmetry operations that we use to reduce the
#   reducible Brillouin zone (RBZ) to the irreducible (IBZ) (basis.ksymops)

@doc raw"""
Return the ``k``-point symmetry operations associated to a lattice and atoms.
"""
function symmetry_operations(lattice, atoms, magnetic_moments=[]; tol_symmetry=1e-5)
    symmetries = []
    # Get symmetries from spglib
    Stildes, τtildes = spglib_get_symmetry(lattice, atoms, magnetic_moments;
                                           tol_symmetry=tol_symmetry)

    for isym = 1:length(Stildes)
        S = Stildes[isym]'                  # in fractional reciprocal coordinates
        τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
        τ = τ .- floor.(τ)
        @assert all(0 .≤ τ .< 1)
        push!(symmetries, (S, τ))
    end

    unique(symmetries)
end

"""
Filter out the symmetry operations that respect the symmetries of the discrete BZ grid
"""
function symmetries_preserving_kgrid(symmetries, kcoords)
    function preserves_grid(S)
        all(normalize_kpoint_coordinate(S * k) in kcoords
            for k in normalize_kpoint_coordinate.(kcoords))
    end
    filter(symop -> preserves_grid(symop[1]), symmetries)
end

"""
Implements a primitive search to find an irreducible subset of kpoints
amongst the provided kpoints.
"""
function find_irreducible_kpoints(kcoords, Stildes, τtildes)

    # This function is required because spglib sometimes flags kpoints
    # as reducible, where we cannot find a symmetry operation to
    # generate them from the provided irreducible kpoints. This
    # reimplements that part of spglib, with a possibly very slow
    # algorithm.

    # Flag which kpoints have already been mapped to another irred.
    # kpoint or which have been decided to be irreducible.
    kcoords_mapped = zeros(Bool, length(kcoords))
    kirreds = empty(kcoords)           # Container for irreducible kpoints
    ksymops = Vector{Vector{SymOp}}()  # Corresponding symops

    while !all(kcoords_mapped)
        # Select next not mapped kpoint as irreducible
        ik = findfirst(isequal(false), kcoords_mapped)
        push!(kirreds, kcoords[ik])
        thisk_symops = [identity_symop()]
        kcoords_mapped[ik] = true

        for jk in findall(.!kcoords_mapped)
            isym = findfirst(1:length(Stildes)) do isym
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kcoords[jk] - (Stildes[isym]' * kcoords[ik]))
            end

            if !isnothing(isym)  # Found a reducible kpoint
                kcoords_mapped[jk] = true
                S = Stildes[isym]'                  # in fractional reciprocal coordinates
                τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
                τ = τ .- floor.(τ)
                @assert all(0 .≤ τ .< 1)
                push!(thisk_symops, (S, τ))
            end
        end  # jk

        push!(ksymops, thisk_symops)
    end
    kirreds, ksymops
end

"""
Apply a symmetry operation to eigenvectors `ψk` at a given `kpoint` to obtain an
equivalent point in [-0.5, 0.5)^3 and associated eigenvectors (expressed in the
basis of the new kpoint).
"""
function apply_ksymop(ksymop::SymOp, basis, kpoint, ψk::AbstractVecOrMat)
    S, τ = ksymop
    S == I && iszero(τ) && return kpoint, ψk

    # Apply S and reduce coordinates to interval [-0.5, 0.5)
    # Doing this reduction is important because
    # of the finite kinetic energy basis cutoff
    @assert all(-0.5 .≤ kpoint.coordinate .< 0.5)
    Sk_raw = S * kpoint.coordinate
    Sk = normalize_kpoint_coordinate(Sk_raw)
    kshift = convert.(Int, Sk - Sk_raw)
    @assert all(-0.5 .≤ Sk .< 0.5)

    # Check whether the resulting kpoint is in the basis:
    ikfull = findfirst(1:length(basis.kpoints)) do idx
        all(isinteger, basis.kpoints[idx].coordinate - Sk)
    end
    if isnothing(ikfull)
        # Build a new kpoint datastructure:
        Skpoint = build_kpoints(basis, [Sk])[1]
    else
        Skpoint = basis.kpoints[ikfull]
        @assert Skpoint.coordinate ≈ Sk
    end

    # Since the eigenfunctions of the Hamiltonian at k and Sk satisfy
    #      u_{Sk}(x) = u_{k}(S^{-1} (x - τ))
    # their Fourier transform is related as
    #      ̂u_{Sk}(G) = e^{-i G \cdot τ} ̂u_k(S^{-1} G)
    invS = Mat3{Int}(inv(S))
    Gs_full = [G + kshift for G in G_vectors(Skpoint)]
    ψSk = zero(ψk)
    for iband in 1:size(ψk, 2)
        for (ig, G_full) in enumerate(Gs_full)
            igired = index_G_vectors(basis, kpoint, invS * G_full)
            @assert igired !== nothing
            ψSk[ig, iband] = cis(-2π * dot(G_full, τ)) * ψk[igired, iband]
        end
    end

    Skpoint, ψSk
end

"""
Apply a `k`-point symmetry operation (the tuple (S, τ)) to a partial density.
"""
function apply_ksymop(symop::SymOp, basis, ρin)
    S, τ = ksymop
    S == I && iszero(τ) && return ρin
    symmetrize(basis, ρin, [symop])
end


# Accumulates the symmetrized versions of the density ρin into ρout (in Fourier space).
# No normalization is performed
function accumulate_over_symmetries!(ρaccu, ρin, basis, symmetries)
    T = eltype(basis)
    for (S, τ) in symmetries
        invS = Mat3{Int}(inv(S))
        # Common special case, where ρin does not need to be processed
        if S == I && iszero(τ)
            ρaccu .+= ρin
            continue
        end

        # Transform ρin -> to the partial density at S * k.
        #
        # Since the eigenfunctions of the Hamiltonian at k and Sk satisfy
        #      u_{Sk}(x) = u_{k}(S^{-1} (x - τ))
        # with Fourier transform
        #      ̂u_{Sk}(G) = e^{-i G \cdot τ} ̂u_k(S^{-1} G)
        # equivalently
        #      ̂ρ_{Sk}(G) = e^{-i G \cdot τ} ̂ρ_k(S^{-1} G)
        for (ig, G) in enumerate(G_vectors(basis))
            igired = index_G_vectors(basis, invS * G)
            if igired !== nothing
                @inbounds ρaccu[ig] += cis(-2T(π) * dot(G, τ)) * ρin[igired]
            end
        end
    end  # (S, τ)
    ρaccu
end

# Low-pass filters ρ (in Fourier) so that symmetry operations acting on it stay in the grid
function lowpass_for_symmetry!(ρ, basis; symmetries=basis.model.symmetries)
    for (S, τ) in symmetries
        S == I && iszero(τ) && continue
        for (ig, G) in enumerate(G_vectors(basis))
            if index_G_vectors(basis, S * G) === nothing
                ρ[ig] = 0
            end
        end
    end
    ρ
end

"""
Symmetrize a density by applying all the model symmetries (by default) and forming the average.
"""
@views function symmetrize(basis, ρin; symmetries=basis.model.symmetries)
    ρin_fourier = r_to_G(basis, ρin)
    ρout_fourier = copy(ρin_fourier)
    for σ = 1:size(ρin, 4)
        lowpass_for_symmetry!(ρin_fourier[:, :, :, σ], basis; symmetries=symmetries)
        ρout_fourier[:, :, :, σ] = accumulate_over_symmetries!(zero(ρin_fourier[:, :, :, σ]),
                                                               ρin_fourier[:, :, :, σ], basis, symmetries)
    end
    G_to_r(basis, ρout_fourier ./ length(symmetries))
end

function check_symmetric(basis, ρin; tol=1e-10, symmetries=ρin.basis.model.symmetries)
    for symop in symmetries
        @assert norm(symmetrize(ρin, [symop]) - ρin) < tol
    end
end
