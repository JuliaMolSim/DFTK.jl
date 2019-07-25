struct PotHartree
    basis::PlaneWaveBasis
end

function update_energies_potential!(energies, potential, op::PotHartree, ρ)
    T = real(eltype(ρ))
    pw = op.basis

    # Solve the Poisson equation ΔV = -4π ρ in Fourier space,
    # i.e. Multiply elementwise by 4π / |G|^2.
    values = [4π * ρ[ig] / sum(abs2, pw.recip_lattice * G)
              for (ig, G) in enumerate(basis_ρ(pw))]

    # Zero the DC component (i.e. assume a compensating charge background)
    values[pw.idx_DC] = 0

    # Fourier-transform values and store in values_real
    values_real = similar(potential, Complex{T})
    G_to_r!(pw, values, values_real)

    # TODO Maybe one could compute the energy directly in Fourier space
    #      and in this way save one FFT
    ρ_real = real(G_to_r!(pw, ρ, similar(ρ, Complex{T}, size(pw.FFT)...)))
    dVol = pw.unit_cell_volume / prod(size(pw.FFT))
    energies[:PotHartree] = 2 * real(sum(ρ_real .* values_real) / 2 * dVol) / 2

    if maximum(imag(values_real)) > 100 * eps(T)
        throw(ArgumentError("Expected potential on the real-space grid B_ρ to be entirely" *
                            " real-valued, but the present density gives rise to a " *
                            "maximal imaginary entry of $(maximum(imag(values_real)))."))
    end
    potential .= real(values_real)

    (energies=energies, potential=potential)
end
