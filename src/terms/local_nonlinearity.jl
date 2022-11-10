"""
Local nonlinearity, with energy ∫f(ρ) where ρ is the density
"""
struct LocalNonlinearity
    f
end
struct TermLocalNonlinearity{TF} <: TermNonlinear
    f::TF
end
(L::LocalNonlinearity)(::AbstractBasis) = TermLocalNonlinearity(L.f)

function ene_ops(term::TermLocalNonlinearity, basis::PlaneWaveBasis{T}, ψ, occupation;
                 ρ, kwargs...) where {T}
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    E = sum(term.f.(ρ)) * basis.dvol
    E = convert_dual(T, E)
    potential = convert_dual.(T, fp.(ρ))

    # In the case of collinear spin, the potential is spin-dependent
    ops = [RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin])
           for kpt in basis.kpoints]
    (; E, ops)
end


function compute_kernel(term::TermLocalNonlinearity, ::AbstractBasis{T}; ρ, kwargs...) where {T}
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    fpp(ρ) = ForwardDiff.derivative(fp, ρ)
    Diagonal(vec(convert_dual.(T, fpp.(ρ))))
end

function apply_kernel(term::TermLocalNonlinearity, ::AbstractBasis{T}, δρ; ρ, kwargs...) where {T}
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    fpp(ρ) = ForwardDiff.derivative(fp, ρ)
    convert_dual.(T, fpp.(ρ) .* δρ)
end
