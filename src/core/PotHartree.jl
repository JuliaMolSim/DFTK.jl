struct PotHartree
    basis::PlaneWaveBasis
end

function compute_potential!(precomp, pot::PotHartree, ρ)
    pw = pot.basis

    # Solve the Poisson equation ΔV = -4π ρ in Fourier space,
    # i.e. Multiply elementwise by 4π / |G|^2.
    values = [4π * ρ[ig] / sum(abs2, pw.recip_lattice * G)
              for (ig, G) in enumerate(basis_ρ(pw))]

    # Zero the DC component (i.e. assume a compensating charge background)
    values[pw.idx_DC] = 0

    # Fourier-transform and store in values_real
    T = eltype(pw.lattice)
    values_real = similar(precomp, Complex{T})
    G_to_r!(pw, values, values_real)

    if maximum(imag(values_real)) > 100 * eps(T)
        throw(ArgumentError("Expected potential on the real-space grid B_ρ to be entirely" *
                            " real-valued, but the present density gives rise to a " *
                            "maximal imaginary entry of $(maximum(imag(values_real)))."))
    end
    precomp .= real(values_real)
end
