# This file contains functions to handle the symetries.
# The type SymOp is defined in Symop.jl

include("external/spglib.jl")

# A symmetry (W, w) (or (S, τ)) induces a symmetry in the Brillouin zone that the
# Hamiltonian at S k is unitary equivalent to that at k, which we exploit to reduce
# computations. The relationship is
#   S = W'
#   τ = -W^-1 w
# (valid both in reduced and Cartesian coordinates). In our notation the rotation matrix
# W and translation w are such that, for each atom of type A at position a, W a + w is also
# an atom of type A.

# The full (reducible) Brillouin zone is implicitly represented by a set of (irreducible)
# kpoints (see explanation in docs). Each irreducible k-point k comes with a list of
# symmetry operations (S, τ) (containing at least the trivial operation (I, 0)), where S is
# a unitary matrix (/!\ in Cartesian but not in reduced coordinates) and τ a translation
# vector. The k-point is then used to represent implicitly the information at all the
# kpoints Sk. The relationship between the Hamiltonians is
#   H_{Sk} = U H_k U*, with
#   (Uu)(x) = u(W x + w)
# or in Fourier space
#   (Uu)(G) = e^{-i G τ} u(S^-1 G)
# In particular, we can choose the eigenvectors at Sk as u_{Sk} = U u_k

# We represent then the BZ as a set of irreducible points `kpoints`, and a set of weights
# `kweights` (summing to 1). The value of observables is given by a weighted sum over the
# irreducible kpoints, plus a symmetrization operation (which depends on the particular way
# the observable transforms under the symmetry).

# There is by decreasing cardinality
# - The group of symmetry operations of the lattice
# - The group of symmetry operations of the crystal (model.symmetries)
# - The group of symmetry operations of the crystal that preserves the BZ mesh (basis.symmetries)

# See https://juliamolsim.github.io/DFTK.jl/stable/developer/symmetries/ for details.

"""
`:none` if no element has a magnetic moment, else `:collinear` or `:full`.
"""
function determine_spin_polarization(magnetic_moments)
    isempty(magnetic_moments) && return :none
    all_magmoms = normalize_magnetic_moment.(magnetic_moments)
    all(iszero, all_magmoms) && return :none
    all(iszero(magmom[1:2]) for magmom in all_magmoms) && return :collinear

    :full
end


"""
Return the Symmetry operations given a `hall_number`.

This function allows to directly access to the space group operations in the
`spglib` database. To specify the space group type with a specific choice,
`hall_number` is used.

The definition of `hall_number` is found at
[Space group type](https://spglib.readthedocs.io/en/latest/dataset.html#dataset-spg-get-dataset-spacegroup-type).
"""
function symmetry_operations(hall_number::Integer)
    Ws, ws = Spglib.get_symmetry_from_database(hall_number)
    [SymOp(W, w) for (W, w) in zip(Ws, ws)]
end

@doc raw"""
Return the symmetries given an atomic structure with optionally designated magnetic moments
on each of the atoms. The symmetries are determined using spglib.
"""
@timing function symmetry_operations(lattice, atoms, positions, magnetic_moments=[];
                                     tol_symmetry=SYMMETRY_TOLERANCE,
                                     check_symmetry=SYMMETRY_CHECK)
    spin_polarization = determine_spin_polarization(magnetic_moments)
    dimension   = count(!iszero, eachcol(lattice))
    if isempty(atoms) || dimension != 3
        # spglib doesn't support these cases, so we default to no symmetries
        return [one(SymOp)]
    end

    if spin_polarization == :full
        @warn("Symmetry detection not yet supported in full spin polarization. " *
              "Returning no symmetries")
        return [one(SymOp)]
    end

    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]
    cell = spglib_cell(lattice, atom_groups, positions, magnetic_moments)
    Ws, ws = try
        if spin_polarization == :none
            Spglib.get_symmetry(cell, tol_symmetry)
        elseif spin_polarization == :collinear
            Spglib.get_symmetry_with_collinear_spin(cell, tol_symmetry)
        end
    catch e
        if e isa Spglib.SpglibError
            msg = ("spglib failed to get the symmetries. Check your lattice, use a " *
                   "uniform BZ mesh or disable symmetries. Spglib reported : " * e.msg)
            throw(Spglib.SpglibError(msg))
        else
            rethrow()
        end
    end

    symmetries = [SymOp(W, w) for (W, w) in zip(Ws, ws)]
    if check_symmetry
        _check_symmetries(symmetries, lattice, atom_groups, positions; tol_symmetry)
    end
    symmetries
end
function symmetry_operations(system::AbstractSystem; kwargs...)
    parsed = parse_system(system)
    symmetry_operations(parsed.lattice, parsed.atoms, parsed.positions,
                        parsed.magnetic_moments; kwargs...)
end

@timing function _check_symmetries(symmetries::AbstractVector{<:SymOp},
                                   lattice, atom_groups, positions;
                                   tol_symmetry=SYMMETRY_TOLERANCE)
    # Check (W, w) maps atoms to equivalent atoms in the lattice
    for symop in symmetries
        W, w = symop.W, symop.w

        # Check (A W A^{-1}) is orthogonal
        Wcart = lattice * W / lattice
        if maximum(abs, Wcart'Wcart - I) > tol_symmetry
            error("Issue in symmetry determination: Non-orthogonal rotation matrix.")
        end

        for group in atom_groups
            group_positions = positions[group]
            for coord in group_positions
                # If all elements of a difference in diffs is integer, then
                # W * coord + w and pos are equivalent lattice positions
                if !any(c -> is_approx_integer(W * coord + w - c; atol=tol_symmetry), group_positions)
                    error("Issue in symmetry determination: Cannot map the atom at position " *
                          "$coord to another atom of the same element under the symmetry " *
                          "operation (W, w):\n($W, $w)")
                end
            end
        end
    end
end

# Approximate in; can be performance-critical, so we optimize in case of rationals
_is_approx_in(x::AbstractArray{<:Rational}, X)  = any(isequal(x), X)
_is_approx_in(x::AbstractArray{T}, X) where {T} = any(y -> isapprox(x, y; atol=sqrt(eps(T))), X)

"""
Filter out the symmetry operations that don't respect the symmetries of the discrete BZ grid
"""
function symmetries_preserving_kgrid(symmetries, kcoords)
    kcoords_normalized = normalize_kpoint_coordinate.(kcoords)
    function preserves_grid(symop)
        all(_is_approx_in(normalize_kpoint_coordinate(symop.S * k), kcoords_normalized)
            for k in kcoords_normalized)
    end
    filter(preserves_grid, symmetries)
end
function symmetries_preserving_kgrid(symmetries, kgrid::ExplicitKpoints)
    # First apply symmetries as the provides k-points can be arbitrary
    # (e.g. only along a line or similar)
    all_kcoords = unfold_kcoords(kgrid.kcoords, symmetries)
    symmetries_preserving_kgrid(symmetries, all_kcoords)
end
function symmetries_preserving_kgrid(symmetries, kgrid::MonkhorstPack)
    if all(isone, kgrid.kgrid_size)
        # TODO Keeping this special casing from version of the code before refactor
        [one(SymOp)]
    else
        # if k' = Rk
        # then
        #    R' = diag(kgrid) R diag(kgrid)^-1
        # should be integer where

        # TODO This can certainly be improved by knowing this is an MP grid,
        #      see symmetries_preserving_rgrid below for ideas
        symmetries_preserving_kgrid(symmetries, reducible_kcoords(kgrid).kcoords)
    end
end

"""
Filter out the symmetry operations that don't respect the symmetries of the discrete real-space grid
"""
function symmetries_preserving_rgrid(symmetries, fft_size)
    is_in_grid(r) = all(zip(r, fft_size)) do (ri, size)
        abs(ri * size - round(ri * size)) / size ≤ SYMMETRY_TOLERANCE
    end

    onehot3(i) = (x = zeros(Bool, 3); x[i] = true; Vec3(x))
    function preserves_grid(symop)
        all(is_in_grid(symop.W * onehot3(i) .// fft_size[i] + symop.w) for i=1:3)
    end

    filter(preserves_grid, symmetries)
end

@doc raw"""
Apply various standardisations to a lattice and a list of atoms. It uses spglib to detect
symmetries (within `tol_symmetry`), then cleans up the lattice according to the symmetries
(unless `correct_symmetry` is `false`) and returns the resulting standard lattice
and atoms. If `primitive` is `true` (default) the primitive unit cell is returned, else
the conventional unit cell is returned.
"""
function standardize_atoms(lattice, atoms, positions, magnetic_moments=[]; kwargs...)
    @assert length(atoms) == length(positions)
    @assert isempty(magnetic_moments) || (length(atoms) == length(magnetic_moments))
    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]
    ret = spglib_standardize_cell(lattice, atom_groups, positions, magnetic_moments; kwargs...)
    (; ret.lattice, atoms, ret.positions, ret.magnetic_moments)
end


"""
Apply a symmetry operation to eigenvectors `ψk` at a given `kpoint` to obtain an
equivalent point in [-0.5, 0.5)^3 and associated eigenvectors (expressed in the
basis of the new ``k``-point).
"""
function apply_symop(symop::SymOp, basis, kpoint, ψk::AbstractVecOrMat)
    S, τ = symop.S, symop.τ
    isone(symop) && return kpoint, ψk

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
        # Build new k-point datastructure
        Skpoint = Kpoint(basis, Sk, kpoint.spin)
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
    for iband = 1:size(ψk, 2)
        for (ig, G_full) in enumerate(Gs_full)
            igired = index_G_vectors(basis, kpoint, invS * G_full)
            @assert igired !== nothing
            ψSk[ig, iband] = cis2pi(-dot(G_full, τ)) * ψk[igired, iband]
        end
    end

    Skpoint, ψSk
end

"""
Apply a symmetry operation to a density.
"""
function apply_symop(symop::SymOp, basis, ρin; kwargs...)
    isone(symop) && return ρin
    symmetrize_ρ(basis, ρin; symmetries=[symop], kwargs...)
end

# Accumulates the symmetrized versions of the density ρin into ρout (in Fourier space).
# No normalization is performed
function accumulate_over_symmetries!(ρaccu, ρin, basis::PlaneWaveBasis{T}, symmetries) where {T}
    for symop in symmetries
        # Common special case, where ρin does not need to be processed
        if isone(symop)
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
            isnothing(igired) && continue

            if iszero(symop.τ)
                @inbounds ρaccu[ig] += ρin[igired]
            else
                factor = cis2pi(-T(dot(G, symop.τ)))
                @inbounds ρaccu[ig] += factor * ρin[igired]
            end
        end
    end  # symop
    ρaccu
end

# Low-pass filters ρ (in Fourier) so that symmetry operations acting on it stay in the grid
function lowpass_for_symmetry!(ρ::AbstractArray, basis; symmetries=basis.symmetries)
    for symop in symmetries
        isone(symop) && continue
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
@views @timing function symmetrize_ρ(basis, ρ::AbstractArray{T};
                                     symmetries=basis.symmetries, do_lowpass=true) where {T}
    ρin_fourier  = to_cpu(fft(basis, ρ))
    ρout_fourier = zero(ρin_fourier)
    for σ = 1:size(ρ, 4)
        accumulate_over_symmetries!(ρout_fourier[:, :, :, σ],
                                    ρin_fourier[:, :, :, σ], basis, symmetries)
        do_lowpass && lowpass_for_symmetry!(ρout_fourier[:, :, :, σ], basis; symmetries)
    end
    inv_fft = T <: Real ? irfft : ifft
    inv_fft(basis, to_device(basis.architecture, ρout_fourier) ./ length(symmetries))
end

"""
Symmetrize the stress tensor, given as a Matrix in Cartesian coordinates
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
array `forces[iel][α,i]`.
"""
function symmetrize_forces(model::Model, forces; symmetries)
    symmetrized_forces = zero(forces)
    for group in model.atom_groups, symop in symmetries
        positions_group = model.positions[group]
        W, w = symop.W, symop.w
        for (idx, position) in enumerate(positions_group)
            # see (A.27) of https://arxiv.org/pdf/0906.2569.pdf
            # (but careful that our symmetries are r -> Wr+w, not R(r+f))
            other_at = W \ (position - w)
            i_other_at = findfirst(a -> is_approx_integer(a - other_at), positions_group)
            # (A.27) is in Cartesian coordinates, and since Wcart is orthogonal,
            # Fsymcart = Wcart * Fcart <=> Fsymred = inv(Wred') Fred
            symmetrized_forces[idx] += inv(W') * forces[group[i_other_at]]
        end
    end
    symmetrized_forces / length(symmetries)
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
        # TODO This can be optimised much better by avoiding the recomputation
        #      of the terms wherever possible.
        use_symmetry_for_kpoint_reduction = false
        return PlaneWaveBasis(basis.model, basis.Ecut, basis.fft_size,
                              basis.variational, basis.kgrid,
                              basis.symmetries_respect_rgrid,
                              use_symmetry_for_kpoint_reduction,
                              basis.comm_kpts, basis.architecture)
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

function unfold_array(basis_irred, basis_unfolded, data, is_ψ)
    if basis_irred == basis_unfolded
        return data
    end
    if !(basis_irred.comm_kpts == basis_irred.comm_kpts == MPI.COMM_WORLD)
        error("Brillouin zone symmetry unfolding not supported with MPI yet")
    end
    data_unfolded = similar(data, length(basis_unfolded.kpoints))
    for ik_unfolded = 1:length(basis_unfolded.kpoints)
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
    ψ = unfold_array(scfres.basis, basis_unfolded, scfres.ψ, true)
    eigenvalues = unfold_array(scfres.basis, basis_unfolded, scfres.eigenvalues, false)
    occupation = unfold_array(scfres.basis, basis_unfolded, scfres.occupation, false)
    energies, ham = energy_hamiltonian(basis_unfolded, ψ, occupation;
                                       scfres.ρ, eigenvalues, scfres.εF)
    @assert energies.total ≈ scfres.energies.total
    new_scfres = (; basis=basis_unfolded, ψ, ham, eigenvalues, occupation)
    merge(scfres, new_scfres)
end

function unfold_kcoords(kcoords, symmetries)
    # unfold
    all_kcoords = [normalize_kpoint_coordinate(symop.S * kcoord)
                   for kcoord in kcoords, symop in symmetries]
    # uniquify
    digits = ceil(Int, -log10(SYMMETRY_TOLERANCE))
    unique(all_kcoords) do k
        # if x and y are both close to a round value, round(x)===round(y), except at zero
        # where 0.0 and -0.0 are considered different by unique. Add 0.0 to make both
        # -0.0 and 0.0 equal to 0.0
        normalize_kpoint_coordinate(round.(k; digits) .+ 0.0)
    end
end

"""
Ensure its real-space equivalent of passed Fourier-space representation is entirely real by
removing wavevectors `G` that don't have a `-G` counterpart in the basis.
"""
@timing function enforce_real!(fourier_coeffs, basis::PlaneWaveBasis)
    lowpass_for_symmetry!(fourier_coeffs, basis; symmetries=[SymOp(-Mat3(I), Vec3(0, 0, 0))])
end
