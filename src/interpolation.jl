import Interpolations
import Interpolations: interpolate, extrapolate, scale, BSpline, Quadratic, OnCell

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
necessarily exact. Intended only to construct guesses for iterative solvers.
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
    #TODO: use a map, or this will not be GPU compatible (scalar indexing)
    for iin in 1:size(data_in, 1)
        idx_fft = kpoint_in.mapping[iin]
        idx_fft in keys(kpoint_out.mapping_inv) || continue
        iout = kpoint_out.mapping_inv[idx_fft]
        data_out[iout, :] = data_in[iin, :]
    end
    ortho_qr(data_out)  # Re-orthogonalize and renormalize
end
