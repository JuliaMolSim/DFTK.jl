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
# irreducible k-point k comes with a list of symmetry operations
# (S,τ) (containing at least the trivial operation (I,0)), where S
# is a rotation matrix (/!\ not unitary in reduced coordinates)
# and τ a translation vector. The k-point is then used to represent
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
# the value of that observable at each irreducible k-point, weighted by
# kweight

# There is by decreasing cardinality
# - The group of symmetry operations of the lattice
# - The group of symmetry operations of the crystal (model.symmetries)
# - The group of symmetry operations of the crystal that preserves the BZ mesh (basis.symmetries)
# - The set of symmetry operations that we use to reduce the
#   reducible Brillouin zone (RBZ) to the irreducible (IBZ) (basis.ksymops)

# See https://juliamolsim.github.io/DFTK.jl/stable/advanced/symmetries for details.

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
    # k-point or which have been decided to be irreducible.
    kcoords_mapped = zeros(Bool, length(kcoords))
    kirreds = empty(kcoords)           # Container for irreducible kpoints
    ksymops = Vector{Vector{SymOp}}()  # Corresponding symops

    while !all(kcoords_mapped)
        # Select next not mapped k-point as irreducible
        ik = findfirst(isequal(false), kcoords_mapped)
        push!(kirreds, kcoords[ik])
        thisk_symops = [identity_symop()]
        kcoords_mapped[ik] = true

        for jk in findall(.!kcoords_mapped)
            isym = findfirst(1:length(Stildes)) do isym
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-points
                all(isinteger, kcoords[jk] - (Stildes[isym]' * kcoords[ik]))
            end

            if !isnothing(isym)  # Found a reducible k-point
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
basis of the new ``k``-point).
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

    # Check whether the resulting k-point is in the basis:
    ikfull = findfirst(1:length(basis.kpoints)) do idx
        all(isinteger, basis.kpoints[idx].coordinate - Sk)
    end
    if isnothing(ikfull)
        # Build a new k-point datastructure:
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
    Gs_full = [G + kshift for G in G_vectors(basis, Skpoint)]
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
    symmetrize_ρ(basis, ρin, [symop])
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
        for (ig, G) in enumerate(G_vectors_generator(basis.fft_size))
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
        for (ig, G) in enumerate(G_vectors_generator(basis.fft_size))
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
@views function symmetrize_ρ(basis, ρin; symmetries=basis.model.symmetries)
    ρin_fourier = r_to_G(basis, ρin)
    ρout_fourier = copy(ρin_fourier)
    for σ = 1:size(ρin, 4)
        lowpass_for_symmetry!(ρin_fourier[:, :, :, σ], basis; symmetries=symmetries)
        ρout_fourier[:, :, :, σ] = accumulate_over_symmetries!(zero(ρin_fourier[:, :, :, σ]),
                                                               ρin_fourier[:, :, :, σ], basis, symmetries)
    end
    G_to_r(basis, ρout_fourier ./ length(symmetries))
end
# symmetrize the stress tensor, which is a rank-2 contravariant tensor in reduced coordinates
function symmetrize_stresses(lattice, symmetries, stresses)
    stresses_symmetrized = zero(stresses)
    for (S, τ) in symmetries
        S_reduced = inv(lattice) * S * lattice
        stresses_symmetrized += S_reduced' * stresses * S_reduced
    end
    stresses_symmetrized /= length(symmetries)
    stresses_symmetrized
end

function check_symmetric(basis, ρin; tol=1e-10, symmetries=ρin.basis.model.symmetries)
    for symop in symmetries
        @assert norm(symmetrize_ρ(ρin, [symop]) - ρin) < tol
    end
end

""""
Convert a `basis` into one that doesn't use BZ symmetry.
This is mainly useful for debug purposes (e.g. in cases we don't want to
bother thinking about symmetries).
"""
function unfold_bz(basis::PlaneWaveBasis)
    if all(length.(basis.ksymops_global) .== 1)
        return basis
    else
        kcoords = []
        for (ik, kpt) in enumerate(basis.kcoords_global)
            for (S, τ) in basis.ksymops_global[ik]
                push!(kcoords, normalize_kpoint_coordinate(S * kpt))
            end
        end
        new_basis = PlaneWaveBasis(basis.model,
                                   basis.Ecut, basis.fft_size, basis.variational,
                                   kcoords, [[identity_symop()] for _ in 1:length(kcoords)],
                                   basis.kgrid, basis.kshift, basis.symmetries, basis.comm_kpts)
    end
end

# find where in the irreducible basis `basis_irred` the k-point `kpt_unfolded` is handled
function unfold_mapping(basis_irred, kpt_unfolded)
    for ik_irred = 1:length(basis_irred.kpoints)
        kpt_irred = basis_irred.kpoints[ik_irred]
        for symop in basis_irred.ksymops[ik_irred]
            Sk_irred = normalize_kpoint_coordinate(symop[1] * kpt_irred.coordinate)
            k_unfolded = normalize_kpoint_coordinate(kpt_unfolded.coordinate)
            if (Sk_irred ≈ k_unfolded) && (kpt_unfolded.spin == kpt_irred.spin)
                return ik_irred, symop
            end
        end
    end
    error("Invalid unfolding of BZ")
end

function unfold_array_(basis_irred, basis_unfolded, data, is_ψ)
    if basis_irred == basis_unfolded
        return data
    end
    if !(basis_irred.comm_kpts == basis_irred.comm_kpts == MPI.COMM_WORLD)
        error("Brillouin zone symmetry unfolding not supported with MPI yet")
    end
    data_unfolded = similar(data, length(basis_unfolded.kpoints))
    for ik_unfolded in 1:length(basis_unfolded.kpoints)
        kpt_unfolded = basis_unfolded.kpoints[ik_unfolded]
        ik_irred, symop = unfold_mapping(basis_irred, kpt_unfolded)
        if is_ψ
            # transform ψ_k from data into ψ_Sk in data_unfolded
            kunfold_coord = kpt_unfolded.coordinate
            @assert normalize_kpoint_coordinate(kunfold_coord) ≈ kunfold_coord
            _, ψSk = apply_ksymop(symop, basis_irred,
                                  basis_irred.kpoints[ik_irred], data[ik_irred])
            data_unfolded[ik_unfolded] = ψSk
        else
            # simple copy
            data_unfolded[ik_unfolded] = data[ik_irred]
        end
    end
    data_unfolded
end

function unfold_bz(scfres)
    basis_unfolded = unfold_bz(scfres.basis)
    ψ = unfold_array_(scfres.basis, basis_unfolded, scfres.ψ, true)
    eigenvalues = unfold_array_(scfres.basis, basis_unfolded, scfres.eigenvalues, false)
    occupation = unfold_array_(scfres.basis, basis_unfolded, scfres.occupation, false)
    E, ham = energy_hamiltonian(basis_unfolded, ψ, occupation;
                                scfres.ρ, eigenvalues, scfres.εF)
    @assert E.total ≈ scfres.energies.total
    new_scfres = (; basis=basis_unfolded, ψ, ham, eigenvalues, occupation)
    merge(scfres, new_scfres)
end
