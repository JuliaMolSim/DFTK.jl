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


function update_energies_potential!(energies, potential, op::PotXc, ρ)
    T = real(eltype(ρ))
    pw = op.basis

    ρ_real = similar(potential, Complex{T})
    G_to_r!(pw, ρ, ρ_real)
    @assert(maximum(abs.(imag(ρ_real))) < 100 * eps(T),
            "Imaginary part too large $(maximum(imag(ρ_real)))")
    ρ_real = real(ρ_real)

    energies[:XC] = T(0)
    potential .= 0
    V = similar(ρ_real)
    E = similar(ρ_real)

    for xc in op.functional
        # TODO Implement GGA
        @assert xc.family == Libxc.family_lda

        evaluate_lda!(xc, ρ_real, Vρ=V, E=E)
        potential .+= V

        # TODO This is a fat hack until the Libxc changes are in master
        identifier = Meta.parse(xc.name)
        # end TODO

        # Factor (1/2) to avoid double counting of electrons (see energy expression)
        # Factor 2 because α and β operators are identical for spin-restricted case
        dVol = pw.unit_cell_volume / prod(size(pw.FFT))
        energies[identifier] = 2 * sum(E .* ρ_real) * dVol / 2
        energies[:XC] += energies[xc.identifier]
    end

    (energies=energies, potential=potential)
end
