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

# FD on the GPU, when T<:Dual causes all sorts of troubles, at least on AMD. TODO: also on NVIDIA?
# TODO: only transfer to CPU when T <: Dual ?, or only if ROCArray?
function ene_ops(term::TermLocalNonlinearity, basis::PlaneWaveBasis{T}, ψ, occupation;
                 ρ, kwargs...) where {T}
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    E = sum(fρ -> convert_dual(T, fρ), term.f.(ρ)) * basis.dvol
    potential = to_device(basis.architecture, convert_dual.(T, fp.(to_cpu(ρ))))

    # In the case of collinear spin, the potential is spin-dependent
    ops = [RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin])
           for kpt in basis.kpoints]
    (; E, ops)
end


function compute_kernel(term::TermLocalNonlinearity, basis::AbstractBasis{T}; ρ, kwargs...) where {T}
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    fpp(ρ) = ForwardDiff.derivative(fp, ρ)
    Diagonal(to_device(basis.architecture, vec(convert_dual.(T, fpp.(to_cpu(ρ))))))
end

function apply_kernel(term::TermLocalNonlinearity, basis::AbstractBasis{T},
                      δρ::AbstractArray{Tδρ}; ρ, kwargs...) where {T, Tδρ}
    S = promote_type(T, Tδρ)
    fp(ρ) = ForwardDiff.derivative(term.f, ρ)
    fpp(ρ) = ForwardDiff.derivative(fp, ρ)
    to_device(basis.architecture, convert_dual.(S, fpp.(to_cpu(ρ)) .* to_cpu(δρ)))
end
