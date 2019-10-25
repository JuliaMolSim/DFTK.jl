struct Density
    basis
    real     # Real-space component
    fourier  # Fourier-space component
end

function density_from_real(basis, ρ_real)
    T = real(eltype(ρ_real))
    if maximum(imag(ρ_real)) > 100 * eps(T)
        @warn "Large norm(imag(ρ))" norm_imag=maximum(imag(ρ_real))
    end

    Density(basis, real(ρ_real), r_to_G(basis, ρ_real .+ 0im))
end

function density_from_fourier(basis, ρ_fourier)
    ρ_real = G_to_r(basis, ρ_fourier .+ 0im)
    T = real(eltype(ρ_real))
    if maximum(imag(ρ_real)) > 100 * eps(T)
        @warn "Large norm(imag(ρ))" norm_imag=maximum(imag(ρ_real))
    end

    Density(basis, real(ρ_real), ρ_fourier .+ 0im)
end

# This assumes CPU arrays
density_zero(basis::PlaneWaveModel{T}) where T = density_from_real(basis, zeros(T, basis.fft_size))

import Base: real
real(ρ::Density) = ρ.real
fourier(ρ::Density) = ρ.fourier
