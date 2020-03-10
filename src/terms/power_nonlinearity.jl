"""
Power nonlinearity, with energy C ∫ρ^α.
"""
struct PowerNonlinearity
    C::Real
    α::Real
end
(P::PowerNonlinearity)(basis) = TermPowerNonlinearity(basis, P.C, P.α)

struct TermPowerNonlinearity <: Term
    basis::PlaneWaveBasis
    C::Real
    α::Real
end

function ene_ops(term::TermPowerNonlinearity, ψ, occ; ρ, kwargs...)
    basis = term.basis
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

    E = term.C * sum(ρ.real .^ term.α) * dVol
    potential = @. term.C * term.α * ρ.real^(term.α-1)

    ops = [RealSpaceMultiplication(basis, kpoint, potential) for kpoint in basis.kpoints]
    (E=E, ops=ops)
end
