struct PotXc
    basis::PlaneWaveBasis
    supersampling::Int  # Supersampling for the XC grid
    func                # Functional or functionals to be used
end
PotXc(basis::PlaneWaveBasis, func; supersampling=2) = PotXc(basis, supersampling, func)


function compute_potential!(precomp, pot::PotXc, ρ)
    T = eltype(precomp)

    # TODO Consider supersampling
    @assert pot.supersampling == 2
    ρ_real = similar(precomp, Complex{T})
    G_to_r!(pot.basis, ρ, ρ_real)
    @assert(maximum(abs.(imag(ρ_real))) < 100 * eps(T),
            "Imaginary part too large $(maximum(imag(ρ_real)))")
    ρ_real = real(ρ_real)

    precomp .= 0
    for xc in pot.func
        # TODO Implement GGA
        @assert xc.family == family_lda

        V = similar(ρ_real)
        evaluate_lda!(xc, ρ_real, Vρ=V)
        precomp .+= V
    end

    precomp
end
