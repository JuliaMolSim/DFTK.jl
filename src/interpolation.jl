"""
Interpolate a function expressed in a basis `b_in` to a basis `b_out`
This interpolation uses a very basic real-space algorithm, and makes
a DWIM-y attempt to take into account the fact that b_out can be a supercell of b_in
"""
function interpolate_density(ρ_in::RealFourierArray, b_out::PlaneWaveBasis)
    ρ_out = interpolate_density(ρ_in.real, ρ_in.basis.fft_size, b_out.fft_size,
                                ρ_in.basis.model.lattice, b_out.model.lattice)
    from_real(b_out, ρ_out)
end

# TODO Specialisation for the common case lattice_out = lattice_in
function interpolate_density(ρ_in::AbstractArray, grid_in, grid_out, lattice_in, lattice_out=lattice_in)
    T = real(eltype(ρ_in))
    @assert size(ρ_in) == grid_in

    # First, build supercell, array of 3 ints
    supercell = zeros(Int, 3)
    for i = 1:3
        if norm(lattice_in[:, i]) == 0
            @assert norm(lattice_out[:, i]) == 0
            supercell[i] = 1
        else
            supercell[i] = round(Int, norm(lattice_out[:, i]) / norm(lattice_in[:, i]))
        end
        if norm(lattice_out[:, i] - supercell[i]*lattice_in[:, i]) > .3*norm(lattice_out[:, i])
            @warn "In direction $i, the output lattice is very different from the input lattice"
        end
    end

    # ρ_in represents a periodic function, on a grid 0, 1/N, ... (N-1)/N
    grid_supercell = grid_in .* supercell
    ρ_in_supercell = similar(ρ_in, (grid_supercell...))
    for i = 1:supercell[1]
        for j = 1:supercell[2]
            for k = 1:supercell[3]
                ρ_in_supercell[
                    1 + (i-1)*grid_in[1] : i*grid_in[1],
                    1 + (j-1)*grid_in[2] : j*grid_in[2],
                    1 + (k-1)*grid_in[3] : k*grid_in[3]] = ρ_in
            end
        end
    end

    # interpolate ρ_in_supercell from grid grid_supercell to grid_out
    axes_in = (range(0, 1, length=grid_supercell[i]+1)[1:end-1] for i=1:3)
    itp = interpolate(ρ_in_supercell, BSpline(Quadratic(Periodic(OnCell()))))
    sitp = scale(itp, axes_in...)
    ρ_interp = extrapolate(sitp, Periodic())
    ρ_out = similar(ρ_in, grid_out)
    for i = 1:grid_out[1]
        for j = 1:grid_out[2]
            for k = 1:grid_out[3]
                ρ_out[i, j, k] = ρ_interp((i-1)/grid_out[1],
                                          (j-1)/grid_out[2],
                                          (k-1)/grid_out[3])
            end
        end
    end

    ρ_out
end


"""
Interpolate some data from one k-Point to another. The interpolation is fast, but not
necessarily exact or even normalised. Intended only to construct guesses for iterative
solvers
"""
function interpolate_kpoint(data_in::AbstractVecOrMat, kpoint_in::Kpoint, kpoint_out::Kpoint)
    if kpoint_in == kpoint_out
        return copy(data_in)
    end
    @assert length(G_vectors(kpoint_in)) == size(data_in, 1)

    n_bands = size(data_in, 2)
    data_out = similar(data_in, length(G_vectors(kpoint_out)), n_bands) .= 0
    for iin in 1:size(data_in, 1)
        idx_fft = kpoint_in.mapping[iin]
        idx_fft in keys(kpoint_out.mapping_inv) || continue
        iout = kpoint_out.mapping_inv[idx_fft]
        data_out[iout, :] = data_in[iin, :]
    end
    data_out
end

"""
Compute indices from basis_in that are in basis_out, alongside the complementary
set of indices of basis_out that are not in basis_in. Currently only for basis_out
bigger than basis_in.
"""
function grid_interpolation_indices(basis_in, basis_out)
    # build the indices from basis_out that are in basis_in
    idcs_out = []
    for (ik, kpt_out) in enumerate(basis_out.kpoints)
        # Get indices of the G vectors of the old basis inside the new basis.
        idcsk_out = index_G_vectors.(Ref(basis_out), G_vectors(basis_in.kpoints[ik]))

        # Linearise the indices
        idcsk_out = getindex.(Ref(LinearIndices(basis_out.fft_size)), idcsk_out)

        # Map to the indices of the corresponding G-vectors in G_vectors(kpt_out)
        idcsk_out = indexin(idcsk_out, kpt_out.mapping)
        # this array might contains some nothings if basis_out has less G_vectors
        # than basis_in at this kpoint
        push!(idcs_out, idcsk_out)
    end

    # build the complement indices of basis_out that are not in basis_in
    # create empty arrays if there is at least one nothing in idcs_out[ik]
    idcs_out_cplmt = [[id for id in 1:length(G_vectors(basis_out.kpoints[ik]))
                       if any(isnothing, idcs_out[ik]) && !(id in idcs_out[ik])]
                      for ik in 1:length(basis_in.kpoints)]

    (idcs_out, idcs_out_cplmt)
end

"""
Interpolate Bloch wave between two basis sets. Limited feature set. Currently only
interpolation to a bigger grid (larger Ecut) on the same lattice supported.
"""
function interpolate_blochwave(ψ_in, basis_in, basis_out)
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))

    ψ_out = empty(ψ_in)

    # separating indices from basis_out that are in basis_in from those that are
    # not in basis_in, eventually idcs_out[ik] contains nothings, in which case
    # idcs_out_cplmt[ik] = [] when the output grid is smaller
    idcs_out, idcs_out_cplmt = grid_interpolation_indices(basis_in, basis_out)

    # separating indices from basis_in that are in basis_out from those that are
    # not in basis_out, eventually idcs_in[ik] contains nothing, in which case
    # idcs_in_cplmt[ik] = [] when the output grid is bigger
    idcs_in, idcs_in_cplmt = grid_interpolation_indices(basis_out, basis_in)

    # Small talk on what we did here :
    # When Ecut_out > Ecut_in, only idcs_out is needed. Sometimes, we might need
    # to interpolate between two grids that have the same Ecut (hence same
    # fft_size) but slightly different lattices. In this case, both idcs_out and
    # idcs_in are needed as it is not possible to predict which basis will have most
    # G_vectors (and it might differs from the k points !). This is why we do
    # both interpolations by default, and use the interpolation of basis_out into
    # basis_in when needed.

    for (ik, kpt_out) in enumerate(basis_out.kpoints)
        n_bands = size(ψ_in[ik], 2)

        # Set values
        ψk_out = similar(ψ_in[ik], length(G_vectors(kpt_out)), n_bands)
        ψk_out .= 0
        if !any(isnothing, idcs_out[ik])
            # only this is used when Ecut_out > Ecut_in
            ψk_out[idcs_out[ik], :] .= ψ_in[ik]
        else
            # when needed, we get rid of high frequencies G_vectors
            ψk_out .= ψ_in[ik][idcs_in[ik], :]
        end
        push!(ψ_out, ψk_out)
    end

    ψ_out, idcs_out, idcs_out_cplmt
end
