using ForwardDiff
"""
Local nonlinearity, with energy ∫f(ρ) where ρ is the density
"""
struct LocalNonlinearity
    f
end
@deprecate PowerNonlinearity(C, α) LocalNonlinearity(ρ -> C * ρ^α)
struct TermLocalNonlinearity{TF} <: TermNonlinear
    f::TF
end
(L::LocalNonlinearity)(::AbstractBasis) = TermLocalNonlinearity(L.f)

function ene_ops(term::TermLocalNonlinearity, basis::PlaneWaveBasis, ψ, occ; ρ, kwargs...)
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    E = sum(term.f.(ρ)) * basis.dvol
    potential = fp.(ρ)

    # In the case of collinear spin, the potential is spin-dependent
    ops = [RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin])
           for kpt in basis.kpoints]
    (E=E, ops=ops)
end


function compute_kernel(term::TermLocalNonlinearity, ::AbstractBasis; ρ, kwargs...)
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    fpp(ρ) = ForwardDiff.derivative(fp, ρ)
    Diagonal(vec(fpp.(ρ)))
end

function apply_kernel(term::TermLocalNonlinearity, ::AbstractBasis, δρ; ρ, kwargs...)
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    fpp(ρ) = ForwardDiff.derivative(fp, ρ)
    fpp.(ρ) .* δρ
end
