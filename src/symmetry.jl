## This file contains functions to handle the symetries
## See the docs for the theoretical explanation

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

# There is by decreasing cardinality
# - The group of symmetry operations of the lattice
# - The group of symmetry operations of the crystal
# - The group of symmetry operations of the crystal that preserves the BZ mesh
# - The set of symmetry operations that we use to reduce the RBZ to the IBZ

# (S,τ)
const SymOp = Tuple{Mat3{Int}, Vec3{Float64}}

@doc raw"""
Return the ``k``-point symmetry operations associated to a lattice, model or basis.
Since the ``k``-point discretisations may break some of the symmetries, the latter
case will return a subset of the symmetries of the former two.
"""
function symmetry_operations(lattice, atoms; tol_symmetry=1e-5, kcoords=nothing)
    symops = []
    # Get symmetries from spglib
    Stildes, τtildes = spglib_get_symmetry(lattice, atoms, tol_symmetry=tol_symmetry)

    # Notice: In the language of the latex document in the docs
    # spglib returns \tilde{S} and \tilde{τ} in integer real-space coordinates, such that
    # (A Stilde A^{-1}) is the actual \tilde{S} from the document as a unitary matrix.
    #
    # Still we have the following properties for S and τ given in *integer* and
    # *fractional* real-space coordinates:
    #      \tilde{S}^{-1} = S^T (if applied to a vector in frac. coords in reciprocal space)

    for isym = 1:length(Stildes)
        S = Stildes[isym]'                  # in fractional reciprocal coordinates
        τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
        τ = τ .- floor.(τ)
        @assert all(0 .≤ τ .< 1)
        push!(symops, (S, τ))
    end

    symops = unique(symops)

    if kcoords !== nothing
        # filter only the operations that respect the symmetries of the discrete BZ grid
        function preserves_grid(S)
            all(normalize_kpoint_coordinate(S * k) in kcoords
                for k in normalize_kpoint_coordinate.(kcoords))
        end
        symops = filter(symop -> preserves_grid(symop[1]), symops)
    end

    symops
end
symmetry_operations(model::Model; kwargs...) = symmetry_operations(model.lattice, model.atoms; kwargs...)

function symmetry_operations(basis::PlaneWaveBasis)
    res = Set()
    for ik = 1:length(basis.ksymops)
        for isym = 1:length(basis.ksymops[ik])
            push!(res, basis.ksymops[ik][isym])
        end
    end
    res
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
        thisk_symops = [(Mat3{Int}(I), Vec3(zeros(3)))]
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
function apply_ksymop(ksymop, basis, kpoint, ψk::AbstractVecOrMat)
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
function apply_ksymop(symop, ρin::RealFourierArray)
    symop[1] == I && iszero(symop[2]) && return ρin
    from_fourier(ρin.basis, symmetrize(ρin, [symop]))
end


# Accumulates the symmetrized versions of the density ρin into ρout (in Fourier space).
# No normalization is performed
function _symmetrize!(ρaccu, ρin, basis, symops, Gs)
    T = eltype(basis)
    for (S, τ) in symops
        invS = Mat3{Int}(inv(S))
        # Common special case, where ρin does not need to be processed
        if iszero(S - I) && iszero(τ)
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
        for (ig, G) in enumerate(Gs)
            igired = index_G_vectors(basis, invS * G)
            if igired !== nothing
                @inbounds ρaccu[ig] += cis(-2T(π) * dot(G, τ)) * ρin[igired]
            end
        end
    end  # (S, τ)
    ρaccu
end

"""
Symmetrize a `RealFourierArray` by applying all symmetry operations of
the basis (or all symmetries passed as the second argument) and forming
the average.
"""
function symmetrize(ρin::RealFourierArray, symops=symmetry_operations(ρin.basis))
    ρout_fourier = _symmetrize!(zero(ρin.fourier), ρin.fourier, ρin.basis, symops,
                                G_vectors(ρin.basis)) ./ length(symops)
    from_fourier(ρin.basis, ρout_fourier)
end
