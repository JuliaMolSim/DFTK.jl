# Stores the density in both real and fourier space. Should be read-only once created.

struct Density
    basis
    _real     # Real-space component
    _fourier  # Fourier-space component
end
import Base: real
real(ρ::Density) = ρ._real
fourier(ρ::Density) = ρ._fourier

function density_from_real(basis, ρ_real)
    T = real(eltype(ρ_real))
    check_density_real(ρ_real)
    Density(basis, real(ρ_real), r_to_G(basis, ρ_real .+ 0im))
end

function density_from_fourier(basis, ρ_fourier)
    ρ_real = G_to_r(basis, ρ_fourier .+ 0im)
    T = real(eltype(ρ_real))
    check_density_real(ρ_real)
    Density(basis, real(ρ_real), ρ_fourier .+ 0im)
end

# This assumes CPU arrays
Density(basis::PlaneWaveBasis{T}) where T = density_from_real(basis, zeros(T, basis.fft_size))

check_density_real(ρ::Density) = check_density(real(ρ))
function check_density_real(ρreal)
    if norm(imag(ρreal)) > 100 * eps(real(eltype(ρreal)))
        @warn "Large imag(ρ)" norm_imag=norm(imag(ρreal))
    end
end

"""
Interpolate a density `ρ` expressed in a basis `b_in` to a basis `b_out`.
This attempts to take into account the fact that b_out can be a supercell of b_in
"""
function interpolate_density(ρ_in::Density, b_in::PlaneWaveBasis, b_out::PlaneWaveBasis)
    ρ_out = interpolate_density(real(ρ_in), b_in.fft_size, b_out.fft_size, b_in.lattice, b_out.lattice)
    density_from_real(b_out, ρ_out)
end

# ρ_in in real space
function interpolate_density(ρ_in::AbstractArray, grid_in, grid_out, lattice_in, lattice_out; allow_supercell=true)
    T = real(eltype(ρ_in))
    # First, build supercell, array of 3 ints
    supercell = zeros(3)
    for i = 1:3
        if norm(lattice_in[:, i]) == 0
            @assert norm(lattice_out[:, i]) == 0
            supercell[i] = 1
        else
            supercell[i] = lattice_out[:, i] ./ lattice_in[:, i]
        end
        if abs(supercell[i] - round(supercell[i])) > .3
            @warn "In direction $i, the output lattice is very different from the input lattice"
        end
    end
    supercell = round.(Int, supercell)
    # TODO allow_supercell

    # Compute ρ in a real space grid
    grid_supercell = grid_in .* supercell
    ρ_in_supercell = similar(ρ_in, grid_supercell)
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

    # ρ_in_supercell is in grid_supercell. Now we interpolate to grid_out
    # TODO figure out if this boundary condition is the correct one
    itp = interpolate(ρ_in_supercell, BSpline(Quadratic(Periodic())))
    sitp = scale(itp, (range(0, 1, length=grid_supercell[i]+1)[1:end-1] for i=1:3)...)
    ρ_out = similar(ρ_in, grid_out)
    for i = 1:grid_out[1]
        for j = 1:grid_out[2]
            for k = 1:grid_out[3]
                ρ_out[i, j, k] = sitp((i-1)/grid_out[1],
                                      (j-1)/grid_out[2],
                                      (k-1)/grid_out[3])
            end
        end
    end

    ρ_out
end
