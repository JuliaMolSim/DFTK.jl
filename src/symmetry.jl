# This file contains functions to handle the symetries.
# The type SymOp is defined in Symop.jl

# A symmetry (W, w) (or (S, τ)) induces a symmetry in the Brillouin zone that the
# Hamiltonian at S k is unitary equivalent to that at k, which we exploit to reduce
# computations. The relationship is
#   S = W'
#   τ = -W^-1 w
# (valid both in reduced and cartesian coordinates). In our notation the rotation matrix
# W and translation w are such that, for each atom of type A at position a, W a + w is also
# an atom of type A.

# The full (reducible) Brillouin zone is implicitly represented by a set of (irreducible)
# kpoints (see explanation in docs). Each irreducible k-point k comes with a list of
# symmetry operations (S, τ) (containing at least the trivial operation (I, 0)), where S is
# a unitary matrix (/!\ in cartesian but not in reduced coordinates) and τ a translation
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

@doc raw"""
Return the ``k``-point symmetry operations associated to a lattice and atoms.
"""
function symmetry_operations(lattice, atoms, positions, magnetic_moments=[];
                             tol_symmetry=SYMMETRY_TOLERANCE)
    @assert length(atoms) == length(positions)
    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]
    Ws, ws = spglib_get_symmetry(lattice, atom_groups, positions, magnetic_moments; tol_symmetry)
    [SymOp(W, w) for (W, w) in zip(Ws, ws)]
end
function symmetry_operations(system::AbstractSystem)
    parsed = parse_system(system)
    symmetry_operations(parsed.lattice, parsed.atoms, parsed.positions, parsed.magnetic_moments)
end

# Approximate in; can be performance-critical, so we optimize in case of rationals
is_approx_in_(x::AbstractArray{<:Rational}, X)  = any(isequal(x), X)
is_approx_in_(x::AbstractArray{T}, X) where {T} = any(y -> isapprox(x, y; atol=sqrt(eps(T))), X)

"""
Filter out the symmetry operations that don't respect the symmetries of the discrete BZ grid
"""
function symmetries_preserving_kgrid(symmetries, kcoords)
    kcoords_normalized = normalize_kpoint_coordinate.(kcoords)
    function preserves_grid(symop)
        all(is_approx_in_(normalize_kpoint_coordinate(symop.S * k), kcoords_normalized)
            for k in kcoords_normalized)
    end
    filter(preserves_grid, symmetries)
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
@views @timing function symmetrize_ρ(basis, ρ; symmetries=basis.symmetries, do_lowpass=true)
    ρin_fourier  = to_cpu(fft(basis, ρ))
    ρout_fourier = zero(ρin_fourier)
    for σ = 1:size(ρ, 4)
        accumulate_over_symmetries!(ρout_fourier[:, :, :, σ],
                                    ρin_fourier[:, :, :, σ], basis, symmetries)
        do_lowpass && lowpass_for_symmetry!(ρout_fourier[:, :, :, σ], basis; symmetries)
    end
    irfft(basis, to_device(basis.architecture, ρout_fourier) ./ length(symmetries))
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
    symmetrized_forces = zero(forces)
    for group in model.atom_groups, symop in symmetries
        positions_group = model.positions[group]
        W, w = symop.W, symop.w
        for (idx, position) in enumerate(positions_group)
            # see (A.27) of https://arxiv.org/pdf/0906.2569.pdf
            # (but careful that our symmetries are r -> Wr+w, not R(r+f))
            other_at = W \ (position - w)
            i_other_at = findfirst(a -> is_approx_integer(a - other_at), positions_group)
            # (A.27) is in cartesian coordinates, and since Wcart is orthogonal,
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
        kcoords = unfold_kcoords(basis.kcoords_global, basis.symmetries)
        return PlaneWaveBasis(basis.model,
                              basis.Ecut, basis.fft_size, basis.variational,
                              kcoords, [1/length(kcoords) for _ in kcoords],
                              basis.kgrid, basis.kshift,
                              basis.symmetries_respect_rgrid, basis.comm_kpts, basis.architecture)
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
@timing function enforce_real!(basis, fourier_coeffs)
    lowpass_for_symmetry!(fourier_coeffs, basis; symmetries=[SymOp(-Mat3(I), Vec3(0, 0, 0))])
end
