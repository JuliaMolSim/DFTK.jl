# This file contains functions to handle the symetries.
# The type SymOp is defined in Symop.jl

# A symmetry (W, w) (or (S, τ)) induces a symmetry in the Brillouin zone
# that the Hamiltonian at S k is unitary equivalent to that at k, which we exploit to
# reduce computations. The relationship is
# S = W'
# τ = -W^-1 w
# (valid both in reduced and cartesian coordinates). In our notation
# the rotation matrix W and translation w are such that, for each atom of
# type A at position a, W a + w is also an atom of type A.

# The full (reducible) Brillouin zone is implicitly represented by
# a set of (irreducible) kpoints (see explanation in docs). Each
# irreducible k-point k comes with a list of symmetry operations
# (S, τ) (containing at least the trivial operation (I, 0)), where S
# is a unitary matrix (/!\ in cartesian but not in reduced coordinates)
# and τ a translation vector. The k-point is then used to represent
# implicitly the information at all the kpoints Sk. The relationship
# between the Hamiltonians is
# H_{Sk} = U H_k U*, with
# (Uu)(x) = u(W x + w)
# or in Fourier space
# (Uu)(G) = e^{-i G τ} u(S^-1 G)
# In particular, we can choose the eigenvectors at Sk as u_{Sk} = U u_k

# We represent then the BZ as a set of irreducible points `kpoints`,
# and a set of weights `kweights` (summing to 1). The value of
# observables is given by a weighted sum over the irreducible kpoints,
# plus a symmetrization operation (which depends on the particular way
# the observable transforms under the symmetry)

# There is by decreasing cardinality
# - The group of symmetry operations of the lattice
# - The group of symmetry operations of the crystal (model.symmetries)
# - The group of symmetry operations of the crystal that preserves the BZ mesh (basis.symmetries)

# See https://juliamolsim.github.io/DFTK.jl/stable/advanced/symmetries for details.

@doc raw"""
Return the symmetry operations associated to a lattice and atoms.
"""
function symmetry_operations(lattice, atoms, magnetic_moments=[];
                             is_time_reversal=false, tol_symmetry=SYMMETRY_TOLERANCE)
    Ws, ws = spglib_get_symmetry(lattice, atoms, magnetic_moments; tol_symmetry)
    symmetries = ([SymOp(W, w) for (W, w) in zip(Ws, ws)])
    if is_time_reversal
        symmetries = vcat(symmetries,
                          [SymOp(symop.W, symop.w, true) for symop in symmetries])
    end
    symmetries
end

"""
Filter out the symmetry operations that respect the symmetries of the discrete BZ grid
"""
function symmetries_preserving_kgrid(symmetries, kcoords)
    kcoords_normalized = normalize_kpoint_coordinate.(kcoords)
    T = eltype(kcoords[1])
    atol = T <: Rational ? 0 : sqrt(eps(T))
    is_approx_in(x, X) = any(y -> isapprox(x, y; atol), X)
    function preserves_grid(symop)
        all(is_approx_in(normalize_kpoint_coordinate(symop.S * k), kcoords_normalized)
            for k in kcoords_normalized)
    end
    filter(preserves_grid, symmetries)
end

"""
Implements a primitive search to find an irreducible subset of kpoints
amongst the provided kpoints.
"""
function find_irreducible_kpoints(kcoords, symmetries)
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
        thisk_symops = [one(SymOp)]
        kcoords_mapped[ik] = true

        for jk in findall(.!kcoords_mapped)
            isym = findfirst(1:length(symmetries)) do isym
                # If the difference between kred and S*k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S * k are equivalent k-points
                all(isinteger, kcoords[jk] - (symmetries[isym].S * kcoords[ik]))
            end

            if !isnothing(isym)  # Found a reducible k-point
                kcoords_mapped[jk] = true
                push!(thisk_symops, symmetries[isym])
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
function apply_symop(symop::SymOp, basis, kpoint, ψk::AbstractVecOrMat)
    S, τ, θ= symop.S, symop.τ, symop.θ
    symop == one(SymOp) && return kpoint, ψk

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
            ψSk[ig, iband] = cis(-2π * dot(G_full, τ)) * 
                             (θ ? conj(ψk[igired, iband]) : ψk[igired, iband])
        end
    end

    Skpoint, ψSk
end

"""
Apply a symmetry operation to a density.
"""
function apply_symop(symop::SymOp, basis, ρin)
    symop == one(SymOp) && return ρin
    symmetrize_ρ(basis, ρin; symmetries=[symop])
end


# Accumulates the symmetrized versions of the density ρin into ρout (in Fourier space).
# No normalization is performed
function accumulate_over_symmetries!(ρaccu, ρin, basis, symmetries)
    T = eltype(basis)
    for symop in symmetries
        # Common special case, where ρin does not need to be processed
        if symop == one(SymOp)
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
        #     ρ ̂_{Sk}(G) = e^{-i G \cdot τ} ̂ρ_k(S^{-1} G)
        invS = Mat3{Int}(inv(symop.S))
        for (ig, G) in enumerate(G_vectors_generator(basis.fft_size))
            igired = index_G_vectors(basis, invS * G)
            if igired !== nothing
                @inbounds ρaccu[ig] += cis(-2T(π) * T(dot(G, symop.τ))) *
                                       (symop.θ ? conj(ρin[igired]) : ρin[igired])
            end
        end
    end  # symop
    ρaccu
end

# Low-pass filters ρ (in Fourier) so that symmetry operations acting on it stay in the grid
function lowpass_for_symmetry!(ρ, basis; symmetries=basis.symmetries)
    for symop in symmetries
        symop == one(SymOp) && continue
        for (ig, G) in enumerate(G_vectors_generator(basis.fft_size))
            if index_G_vectors(basis, symop.S * G) === nothing
                ρ[ig] = 0
            end
        end
    end
    ρ
end

"""
Symmetrize a density by applying all the basis (by default) symmetries and forming the average.
"""
@views @timing function symmetrize_ρ(basis, ρ; symmetries=basis.symmetries)
    ρin_fourier = r_to_G(basis, ρ)
    ρout_fourier = zero(ρin_fourier)
    for σ = 1:size(ρ, 4)
        accumulate_over_symmetries!(ρout_fourier[:, :, :, σ],
                                    ρin_fourier[:, :, :, σ], basis, symmetries)
        lowpass_for_symmetry!(ρout_fourier[:, :, :, σ], basis; symmetries)
    end
    G_to_r(basis, ρout_fourier ./ length(symmetries))
end

"""
Symmetrize the stress tensor, given as a Matrix in cartesian coordinates
"""
function symmetrize_stresses(model::Model, stresses; symmetries)
    # see (A.28) of https://arxiv.org/pdf/0906.2569.pdf
    stresses_symmetrized = zero(stresses)
    for symop in symmetries
        W_cart = matrix_red_to_cart(model, symop.W)
        stresses_symmetrized += W_cart * stresses / W_cart
    end
    stresses_symmetrized /= length(symmetries)
    stresses_symmetrized
end
function symmetrize_stresses(basis::PlaneWaveBasis, stresses)
    symmetrize_stresses(basis.model, stresses; basis.symmetries)
end

"""
Symmetrize the forces in *reduced coordinates*, forces given as an
array forces[iel][α,i]
"""
function symmetrize_forces(model::Model, forces; symmetries)
    atoms = model.atoms
    symmetrized_forces = zero.(forces)
    for (iel, (element, positions)) in enumerate(atoms)
        for symop in symmetries
            W, w = symop.W, symop.w
            for (iat, at) in enumerate(positions)
                # see (A.27) of https://arxiv.org/pdf/0906.2569.pdf
                # (but careful that our symmetries are r -> Wr+w, not R(r+f))
                other_at = W \ (at - w)
                is_approx_integer(r) = all(ri -> abs(ri - round(ri)) ≤ SYMMETRY_TOLERANCE, r)
                i_other_at = findfirst(a -> is_approx_integer(a - other_at), positions)
                symmetrized_forces[iel][iat] += W * forces[iel][i_other_at]
            end
        end
        symmetrized_forces[iel] /= length(symmetries)
    end
    symmetrized_forces
end
function symmetrize_forces(basis::PlaneWaveBasis, forces)
    symmetrize_forces(basis.model, forces; basis.symmetries)
end

""""
Convert a `basis` into one that doesn't use BZ symmetry.
This is mainly useful for debug purposes (e.g. in cases we don't want to
bother thinking about symmetries).
"""
function unfold_bz(basis::PlaneWaveBasis)
    if length(basis.symmetries) == 1
        return basis
    else
        kcoords = unfold_kcoords(basis.kcoords_global, basis.symmetries)
        new_basis = PlaneWaveBasis(basis.model,
                                   basis.Ecut, basis.fft_size, basis.variational,
                                   kcoords, [[one(SymOp)] for _ in 1:length(kcoords)],
                                   basis.kgrid, basis.kshift, basis.symmetries, basis.comm_kpts)
    end
end

# find where in the irreducible basis `basis_irred` the k-point `kpt_unfolded` is handled
function unfold_mapping(basis_irred, kpt_unfolded)
    for ik_irred = 1:length(basis_irred.kpoints)
        kpt_irred = basis_irred.kpoints[ik_irred]
        for symop in basis_irred.symmetries
            Sk_irred = normalize_kpoint_coordinate(symop.S * kpt_irred.coordinate)
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
            _, ψSk = apply_symop(symop, basis_irred,
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

function unfold_kcoords(kcoords, symmetries)
    all_kcoords = [symop.S * kcoord for kcoord in kcoords, symop in symmetries]
    # the above multiplications introduce an error
    unique(k -> normalize_kpoint_coordinate(round.(k; digits=ceil(Int, -log10(SYMMETRY_TOLERANCE)))),
           normalize_kpoint_coordinate.(all_kcoords))
end
