using Interpolations

"""
Interpolate a function expressed in a basis `basis_in` to a basis `basis_out`
This interpolation uses a very basic real-space algorithm, and makes
a DWIM-y attempt to take into account the fact that basis_out can be a supercell of basis_in
"""
function interpolate_density(ρ_in, basis_in::PlaneWaveBasis, basis_out::PlaneWaveBasis)
    ρ_out = interpolate_density(ρ_in, basis_in.fft_size, basis_out.fft_size,
                                basis_in.model.lattice, basis_out.model.lattice)
end

# TODO Specialization for the common case lattice_out = lattice_in
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
necessarily exact or even normalized. Intended only to construct guesses for iterative
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
Compute indices from basis_in that are in basis_out. The output format is an array
idcs_out of size length(basis_in.kpoint), where idcs_out[ik] contains the indices
of the vectors in G_vectors(basis_in) that are also present in G_vectors(basis_out).

If, for some kpt ik, basis_in has less vectors than basis_out, then idcs_out[ik] is
the array of the indices of the G_vectors from basis_in in basis_out.
It is then of size G_vectors(basis_in.kpoints[ik]) and the interpolation can be done with
ψ_out[ik] .= 0
ψ_out[ik][idcs_out[ik], :] .= ψ_in[ik]

Otherwise, if, for some kpt ik, basis_in has more vectors than basis_out, then
idcs_out[ik] just keep the indices of the G_vectors from basis_in that are in basis_out.
It is then of size G_vectors(basis_out.kpoints[ik]) and the interpolation can be done with
ψ_out[ik] .= ψ_in[ik][idcs_in[ik], :]

For the moment, only PlaneWaveBasis with same lattice and kgrid are supported.
"""
function grid_interpolation_indices(basis_in::PlaneWaveBasis{T},
                                    basis_out::PlaneWaveBasis{T}) where T
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))

    idcs_out = []

    for (ik, kpt_out) in enumerate(basis_out.kpoints)
        # Get indices of the G vectors of the old basis inside the new basis.
        idcsk_out = index_G_vectors.(Ref(basis_out), G_vectors(basis_in.kpoints[ik]))
        # if basis_in is too big and contains vectors that are not in
        # basis_out.fft_size, then there are nothings in idcsk_out, which we
        # don't want so we filter them
        filter!(e -> e != nothing, idcsk_out)

        # Linearise the indices
        idcsk_out = getindex.(Ref(LinearIndices(basis_out.fft_size)), idcsk_out)

        # Map to the indices of the corresponding G-vectors in G_vectors(kpt_out)
        idcsk_out = indexin(idcsk_out, kpt_out.mapping)
        # this array might contains some nothings if basis_out has less G_vectors
        # than basis_in at this kpoint
        push!(idcs_out, idcsk_out)
    end

    idcs_out
end

"""
Interpolate Bloch wave between two basis sets. Limited feature set.
Currently, only PlaneWaveBasis with same lattice and kgrid are supported.
"""
function interpolate_blochwave(ψ_in, basis_in::PlaneWaveBasis{T},
                               basis_out::PlaneWaveBasis{T}) where T

    ψ_out = empty(ψ_in)

    # indices from basis_out that are in basis_in
    # idcs_out[ik] might contains nothings if basis_out is smaller than basis_in
    idcs_out = grid_interpolation_indices(basis_in, basis_out)

    # indices from basis_in that are in basis_out
    # idcs_in[ik] might contains nothings if basis_in is smaller than basis_out
    idcs_in  = grid_interpolation_indices(basis_out, basis_in)

    for (ik, kpt_out) in enumerate(basis_out.kpoints)
        n_bands = size(ψ_in[ik], 2)

        # Set values
        ψk_out = similar(ψ_in[ik], length(G_vectors(kpt_out)), n_bands)
        ψk_out .= 0
        if !any(isnothing, idcs_out[ik])
            # if true, then Ecut_out >= Ecut_in and we pad with zeros
            ψk_out[idcs_out[ik], :] .= ψ_in[ik]
        else
            # else, then Ecut_in > Ecut_out and we cut off high frequencies
            ψk_out .= ψ_in[ik][idcs_in[ik], :]
        end
        push!(ψ_out, ψk_out)
    end

    ψ_out
end
