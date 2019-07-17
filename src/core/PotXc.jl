include("xc_evaluate.jl")

struct PotXc
    basis::PlaneWaveBasis
    supersampling::Int  # Supersampling for the XC grid
    functional          # Functional or functionals to be used
end


"""
    PotXc(basis, functional, supersampling=2)

Construct an exchange-correlation term. `functional` is the Libxc.jl `Functional` to be used
or a list of such objects. `supersampling` specifies the supersampling factor for the
exchang-correlation integration grid.
"""
function PotXc(basis::PlaneWaveBasis, functional::Array; supersampling=2)
    @assert(supersampling == 2, "Only the case supersampling == 2 is implemented")
    # TODO Actually not even that ... we assume the grid to use is the density grid
    PotXc(basis, supersampling, functional)
end
function PotXc(basis::PlaneWaveBasis, functional::Functional; supersampling=2)
    PotXc(basis, [functional]; supersampling=supersampling)
end


function compute_potential!(precomp, pot::PotXc, ρ)
    T = real(eltype(ρ))

    ρ_real = similar(precomp, Complex{T})
    G_to_r!(pot.basis, ρ, ρ_real)
    @assert(maximum(abs.(imag(ρ_real))) < 100 * eps(T),
            "Imaginary part too large $(maximum(imag(ρ_real)))")
    ρ_real = real(ρ_real)

    precomp .= 0
    for xc in pot.functional
        # TODO Implement GGA
        @assert xc.family == Libxc.family_lda

        V = similar(ρ_real)
        evaluate_lda!(xc, ρ_real, Vρ=V)
        precomp .+= V
    end

    precomp
end
