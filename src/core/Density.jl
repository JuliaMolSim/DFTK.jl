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
function copy!(dst::Density, src::Density)
    @assert dst.basis == src.basis
    dst._real .= src._real
    dst._fourier .= src._fourier
end
