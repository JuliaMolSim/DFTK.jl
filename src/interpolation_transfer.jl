import Interpolations
import Interpolations: interpolate, extrapolate, scale, BSpline, Quadratic, OnCell
using SparseArrays

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
    itp = interpolate(ρ_in_supercell, BSpline(Quadratic(Interpolations.Periodic(OnCell()))))
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
Interpolate some data from one ``k``-point to another. The interpolation is fast, but not
necessarily exact or even normalized. Intended only to construct guesses for iterative
solvers
"""
function interpolate_kpoint(data_in::AbstractVecOrMat,
                            basis_in::PlaneWaveBasis,  kpoint_in::Kpoint,
                            basis_out::PlaneWaveBasis, kpoint_out::Kpoint)
    # TODO merge with transfer_blochwave_kpt
    if kpoint_in == kpoint_out
        return copy(data_in)
    end
    @assert length(G_vectors(basis_in, kpoint_in)) == size(data_in, 1)

    n_bands  = size(data_in, 2)
    n_Gk_out = length(G_vectors(basis_out, kpoint_out))
    data_out = similar(data_in, n_Gk_out, n_bands) .= 0
    for iin in 1:size(data_in, 1)
        idx_fft = kpoint_in.mapping[iin]
        idx_fft in keys(kpoint_out.mapping_inv) || continue
        iout = kpoint_out.mapping_inv[idx_fft]
        data_out[iout, :] = data_in[iin, :]
    end
    data_out
end

"""
Compute the index mapping between two bases. Returns two arrays
`idcs_in` and `idcs_out` such that `ψkout[idcs_out] = ψkin[idcs_in]` does
the transfer from `ψkin` (defined on `basis_in` and `kpt_in`) to `ψkout`
(defined on `basis_out` and `kpt_out`).
"""
function transfer_mapping(basis_in::PlaneWaveBasis{T},  kpt_in::Kpoint,
                          basis_out::PlaneWaveBasis{T}, kpt_out::Kpoint) where T
    idcs_in  = 1:length(G_vectors(basis_in, kpt_in))  # All entries from idcs_in
    kpt_in == kpt_out && return idcs_in, idcs_in

    # Get indices of the G vectors of the old basis inside the new basis.
    idcs_out = index_G_vectors.(Ref(basis_out), G_vectors(basis_in, kpt_in))

    # In the case where G_vectors(basis_in.kpoints[ik]) are bigger than vectors
    # in the fft_size box of basis_out, we need to filter out the "nothings" to
    # make sure that the index linearization works. It is not an issue to
    # filter these vectors as this can only happen if Ecut_in > Ecut_out.
    if any(isnothing, idcs_out)
        idcs_in  = idcs_in[idcs_out .!= nothing]
        idcs_out = idcs_out[idcs_out .!= nothing]
    end
    idcs_out = getindex.(Ref(LinearIndices(basis_out.fft_size)), idcs_out)

    # Map to the indices of the corresponding G-vectors in
    # G_vectors(basis_out, kpt_out) this array might contains some nothings if
    # basis_out has less G_vectors than basis_in at this k-point
    idcs_out = indexin(idcs_out, kpt_out.mapping)
    if any(isnothing, idcs_out)
        idcs_in  = idcs_in[idcs_out .!= nothing]
        idcs_out = idcs_out[idcs_out .!= nothing]
    end

    idcs_in, idcs_out
end


"""
Return a sparse matrix that maps quantities given on `basis_in` and `kpt_in`
to quantities on `basis_out` and `kpt_out`.
"""
function compute_transfer_matrix(basis_in::PlaneWaveBasis{T}, kpt_in::Kpoint,
                                 basis_out::PlaneWaveBasis{T}, kpt_out::Kpoint) where T
    idcs_in, idcs_out = transfer_mapping(basis_in, kpt_in, basis_out, kpt_out)
    sparse(idcs_out, idcs_in, true)
end


"""
Return a list of sparse matrices (one per ``k``-point) that map quantities given in the
`basis_in` basis to quantities given in the `basis_out` basis.
"""
function compute_transfer_matrix(basis_in::PlaneWaveBasis{T}, basis_out::PlaneWaveBasis{T}) where T
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))
    [compute_transfer_matrix(basis_in, kpt_in, basis_out, kpt_out)
     for (kpt_in, kpt_out) in zip(basis_in.kpoints, basis_out.kpoints)]
end


"""
Transfer an array ψk defined on basis_in ``k``-point kpt_in to basis_out ``k``-point kpt_out.
"""
function transfer_blochwave_kpt(ψk_in, basis_in::PlaneWaveBasis{T}, kpt_in::Kpoint,
                                basis_out::PlaneWaveBasis{T}, kpt_out::Kpoint) where T
    kpt_in == kpt_out && return copy(ψk_in)
    @assert length(G_vectors(basis_in, kpt_in)) == size(ψk_in, 1)
    idcsk_in, idcsk_out = transfer_mapping(basis_in, kpt_in, basis_out, kpt_out)

    n_bands = size(ψk_in, 2)
    ψk_out  = similar(ψk_in, length(G_vectors(basis_out, kpt_out)), n_bands)
    ψk_out .= 0
    ψk_out[idcsk_out, :] .= ψk_in[idcsk_in, :]

    ψk_out
end

"""
Transfer Bloch wave between two basis sets. Limited feature set.
"""
function transfer_blochwave(ψ_in, basis_in::PlaneWaveBasis{T},
                            basis_out::PlaneWaveBasis{T}) where T
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))

    # If, for some kpt ik, basis_in has less vectors than basis_out, then idcs_out[ik] is
    # the array of the indices of the G_vectors from basis_in in basis_out.
    # It is then of size G_vectors(basis_in.kpoints[ik]) and the transfer can be done with
    # ψ_out[ik] .= 0
    # ψ_out[ik][idcs_out[ik], :] .= ψ_in[ik]

    # Otherwise, if, for some kpt ik, basis_in has more vectors than basis_out, then
    # idcs_out[ik] just keep the indices of the G_vectors from basis_in that are in basis_out.
    # It is then of size G_vectors(basis_out.kpoints[ik]) and the transfer can be done with
    # ψ_out[ik] .= ψ_in[ik][idcs_in[ik], :]

    map(enumerate(basis_out.kpoints)) do (ik, kpt_out)
        transfer_blochwave_kpt(ψ_in[ik], basis_in, basis_in.kpoints[ik], basis_out, kpt_out)
    end
end
