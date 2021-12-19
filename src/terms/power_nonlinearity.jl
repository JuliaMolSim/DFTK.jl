"""
Power nonlinearity, with energy C ∫ρ^α where ρ is the density
"""
struct PowerNonlinearity
    C::Real
    α::Real
end
(P::PowerNonlinearity)(::AbstractBasis{T}) where {T} = TermPowerNonlinearity{T}(P.C, P.α)

struct TermPowerNonlinearity{T <: Real}<: TermNonlinear
    C::T
    α::T
end

function ene_ops(term::TermPowerNonlinearity, basis::PlaneWaveBasis, ψ, occ; ρ, kwargs...)
    E = term.C * sum(ρ .^ term.α) * basis.dvol
    potential = @. term.C * term.α * ρ^(term.α-1)

    # In the case of collinear spin, the potential is spin-dependent
    ops = [RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin])
           for kpt in basis.kpoints]
    (E=E, ops=ops)
end


_pnl_kernel(C, α, ρ) = @. C * α * (α-1) * ρ^(α-2)

function compute_kernel(term::TermPowerNonlinearity, ::AbstractBasis; ρ, kwargs...)
    Diagonal(vec(_pnl_kernel(term.C, term.α, ρ)))
end

function apply_kernel(term::TermPowerNonlinearity, ::AbstractBasis, δρ; ρ, kwargs...)
    _pnl_kernel(term.C, term.α, ρ) .* δρ
end
