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
Interpolate Bloch wave between two basis sets. Limited feature set. Currently only
interpolation to a bigger grid (larger Ecut) on the same lattice supported.
"""
function interpolate_blochwave(ψ_in, basis_in, basis_out)
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))

    ψ_out = empty(ψ_in)
    idcs_out = []
    for (ik, kpt_out) in enumerate(basis_out.kpoints)
        n_bands = size(ψ_in[ik], 2)

        # Get indices of the G vectors of the old basis inside the new basis.
        idcsk_out = index_G_vectors.(Ref(basis_out), G_vectors(basis_in.kpoints[ik]))

        # Linearise the indices
        idcsk_out = getindex.(Ref(LinearIndices(basis_out.fft_size)), idcsk_out)

        # Map to the indices of the corresponding G-vectors in G_vectors(kpt_out)
        idcsk_out = indexin(idcsk_out, kpt_out.mapping)
        @assert !any(isnothing, idcsk_out)

        # Set values
        ψk_out = similar(ψ_in[ik], length(G_vectors(kpt_out)), n_bands)
        ψk_out .= 0
        ψk_out[idcsk_out, :] .= ψ_in[ik]
        push!(ψ_out, ψk_out)
        push!(idcs_out, idcsk_out)
    end

    (ψ_out, idcs_out)
end
