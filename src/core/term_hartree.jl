# Functionality for building the Hartree potential term


"""
TODO docme (kind of internal though)
"""
function term_hartree_(basis::PlaneWaveModel, energy::Union{Ref,Nothing}, potential;
                       ρ=nothing, kwargs...)
    @assert ρ !== nothing
    T = real(eltype(ρ))
    model = basis.model

    function ifft(x)
        tmp = G_to_r(basis, x .+ 0im)
        @assert(maximum(abs.(imag(tmp))) < 100 * eps(eltype(real(x))),
                "Imaginary part too large $(maximum(imag(tmp)))")
        real(tmp)
    end

    # Solve the Poisson equation ΔV = -4π ρ in Fourier space,
    # i.e. Multiply elementwise by 4π / |G|^2.
    # 0im to force a complex array
    values = 0im .+ T(4π) * ρ ./ [sum(abs2, model.recip_lattice * G) for G in basis_Cρ(basis)]
    # TODO The above assumes CPU arrays

    # Zero the DC component (i.e. assume a compensating charge background)
    values[1] = 0

    # Fourier-transform values and store in Vh
    Vh = (potential === nothing) ? G_to_r(basis, values) : G_to_r!(potential, basis, values)
    if energy !== nothing
        # TODO Maybe one could compute the energy directly in Fourier space
        #      and in this way save one FFT
        dVol = model.unit_cell_volume / prod(basis.fft_size)
        energy[] = 2 * real(sum(ifft(ρ) .* Vh)) / 2 * dVol / 2
        # One factor (1/2) to avoid double counting of electrons
        # One factor (1/2) because ρ is the sum of alpha and beta density.
        # Factor 2 because α and β operators are identical for case without spin polarisation
    end

    energy, potential
end


"""
TODO docme
"""
term_hartree() = term_hartree_
