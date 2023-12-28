import Interpolations
import Interpolations: interpolate, extrapolate, scale, BSpline, Quadratic, OnCell

"""
Interpolate a density expressed in a basis `basis_in` to a basis `basis_out`.
This interpolation uses a very basic real-space algorithm, and makes a DWIM-y attempt
to take into account the fact that `basis_out` can be a supercell of `basis_in`.
"""
function interpolate_density(ρ_in::AbstractArray{T, 4},
                             basis_in::PlaneWaveBasis,
                             basis_out::PlaneWaveBasis) where {T}
    if basis_in.model.lattice == basis_out.model.lattice
        @assert size(ρ_in) == (basis_in.fft_size..., basis_in.model.n_spin_components)
        interpolate_density(ρ_in, basis_out.fft_size)
    else
        interpolate_density(ρ_in, basis_in.fft_size, basis_out.fft_size,
                            basis_in.model.lattice, basis_out.model.lattice)
    end
end

"""
Interpolate a density in real space from one FFT grid to another. Assumes the
lattice is unchanged.
"""
function interpolate_density(ρ_in::AbstractArray{T, 4}, grid_out::NTuple{3}) where {T}
    n_spin = size(ρ_in, 4)
    interpolate_density!(similar(ρ_in, grid_out..., n_spin), ρ_in)
end

"""
Interpolate a density in real space from one FFT grid to another, where
`lattice_in` and `lattice_out` may be supercells of each other.
"""
function interpolate_density(ρ_in::AbstractArray{T, 4},
                             grid_in::NTuple{3}, grid_out::NTuple{3},
                             lattice_in, lattice_out) where {T}
    # The two lattices should have the same dimension.
    @assert iszero.(eachcol(lattice_in)) == iszero.(eachcol(lattice_out))
    @assert size(ρ_in)[1:3] == grid_in

    # Build supercell, array of 3 integers
    supercell = map(eachcol(lattice_in), eachcol(lattice_out)) do col_in, col_out
        iszero(col_in) ? 1 : round(Int, norm(col_out) / norm(col_in))
    end

    # Check if some direction of lattice_in is not too big compared to lattice_out.
    supercell_in = supercell .* lattice_in
    is_suspicious_direction = map(eachcol(supercell_in), eachcol(lattice_out)) do s_in, a_out
        norm(s_in - a_out) > 0.3*norm(a_out)
    end
    for i in findall(is_suspicious_direction)
        @warn "In direction $i, the output lattice is very different from the input lattice"
    end

    # ρ_in represents a periodic function, on a grid 0, 1/N, ... (N-1)/N
    grid_supercell = grid_in .* supercell
    ρ_in_supercell = similar(ρ_in, grid_supercell..., size(ρ_in, 4))
    for i = 1:supercell[1], j = 1:supercell[2], k = 1:supercell[3]
        ρ_in_supercell[1 + (i-1)*grid_in[1] : i*grid_in[1],
                       1 + (j-1)*grid_in[2] : j*grid_in[2],
                       1 + (k-1)*grid_in[3] : k*grid_in[3], :] = ρ_in
    end

    interpolate_density(ρ_in_supercell, grid_out)
end

function interpolate_density!(ρ_out::AbstractArray{T, 3}, ρ_in::AbstractArray{T, 3}) where {T}
    size(ρ_in) == size(ρ_out) && return ρ_out .= ρ_in

    grid_in  = size(ρ_in)
    grid_out = size(ρ_out)
    axes_in = (range(0, 1, length=grid_in[i]+1)[1:end-1] for i=1:3)
    itp = interpolate(ρ_in, BSpline(Quadratic(Interpolations.Periodic(OnCell()))))
    sitp = scale(itp, axes_in...)
    interpolator = extrapolate(sitp, Periodic())
    for i = 1:grid_out[1], j = 1:grid_out[2], k = 1:grid_out[3]
        ρ_out[i, j, k] = interpolator((i-1)/grid_out[1],
                                      (j-1)/grid_out[2],
                                      (k-1)/grid_out[3])
    end
    ρ_out
end
function interpolate_density!(ρ_out::AbstractArray{T, 4}, ρ_in::AbstractArray{T, 4}) where {T}
    @assert size(ρ_in, 4) == size(ρ_out, 4)
    for (ρ_out_slice, ρ_in_slice) in zip(eachslice(ρ_out; dims=4), eachslice(ρ_in; dims=4))
        interpolate_density!(ρ_out_slice, ρ_in_slice)
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
    # TODO: use a map, or this will not be GPU compatible (scalar indexing)
    for iin = 1:size(data_in, 1)
        idx_fft = kpoint_in.mapping[iin]
        idx_fft in keys(kpoint_out.mapping_inv) || continue
        iout = kpoint_out.mapping_inv[idx_fft]
        data_out[iout, :] = data_in[iin, :]
    end
    ortho_qr(data_out)  # Re-orthogonalize and renormalize
end
