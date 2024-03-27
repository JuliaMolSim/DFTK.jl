### Linear operators operating on quantities in real or fourier space
# This is the optimized low-level interface. Use the functions in Hamiltonian for high-level usage

"""
Linear operators that act on tuples (real, fourier)
The main entry point is `apply!(out, op, in)` which performs the operation `out += op*in`
where `out` and `in` are named tuples `(; real, fourier)`.
They also implement `mul!` and `Matrix(op)` for exploratory use.
"""
abstract type RealFourierOperator end
# RealFourierOperator contain fields `basis` and `kpoint`

# Unoptimized fallback, intended for exploratory use only.
# For performance, call through Hamiltonian which saves FFTs.
function LinearAlgebra.mul!(Hψ::AbstractMatrix, op::RealFourierOperator, ψ::AbstractMatrix)
    ψ_real = ifft(op.basis, op.kpoint, ψ)
    Hψ_fourier = similar(ψ)
    Hψ_real = similar(ψ_real)
    Hψ_fourier .= 0
    Hψ_real .= 0
    apply!((; real=Hψ_real, fourier=Hψ_fourier),
           op,
           (; real=ψ_real, fourier=ψ))
    Hψ .= Hψ_fourier .+ fft(op.basis, op.kpoint, Hψ_real)
    Hψ
end
function LinearAlgebra.mul!(Hψ::AbstractArray3, op::RealFourierOperator, ψ::AbstractArray3)
    @views for n = 1:size(ψ, 3)
        mul!(Hψ[:, :, n], op, ψ[:, :, n])
    end
    Hψ
end
Base.:*(op::RealFourierOperator, ψ) = mul!(similar(ψ), op, ψ)
# Default transformation: from two components tensors to diagonal matrices for each
# reciprocal vector.
function Matrix(op::RealFourierOperator)
    n_Gk = length(G_vectors(op.basis, op.kpoint))
    n_components = op.basis.model.n_components
    reshape(Array(op), n_components*n_Gk, n_components*n_Gk)
end

"""
Noop operation: don't do anything.
Useful for energy terms that don't depend on the orbitals at all (eg nuclei-nuclei interaction).
"""
struct NoopOperator{T <: Real} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
end
apply!(Hψ, op::NoopOperator, ψ) = nothing
function Array(op::NoopOperator)
    n_Gk = length(G_vectors(op.basis, op.kpoint))
    n_components = op.basis.model.n_components
    zeros_like(op.basis.G_vectors, eltype(op.basis), n_components, n_Gk, n_components, n_Gk)
end

"""
Real space multiplication by a potential:
```math
(Hψ)(r) = V(r) ψ(r).
```
"""
struct RealSpaceMultiplication{T <: Real, AT <: AbstractArray} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    potential::AT
end
function apply!(Hψ, op::RealSpaceMultiplication, ψ)
    @views for σ = 1:op.basis.model.n_components
        Hψ.real[σ, :, :, :] .+= op.potential .* ψ.real[σ, :, :, :]
    end
    Hψ.real
end
function Array(op::RealSpaceMultiplication)
    # V(G, G') = <eG|V|eG'> = 1/sqrt(Ω) <e_{G-G'}|V>
    pot_fourier = fft(op.basis, op.potential)
    n_G = length(G_vectors(op.basis, op.kpoint))
    n_components = op.basis.model.n_components
    H = zeros(complex(eltype(op.basis)), n_components, n_G, n_components, n_G)
    for σ = 1:n_components
        for (j, G′) in enumerate(G_vectors(op.basis, op.kpoint))
            for (i, G) in enumerate(G_vectors(op.basis, op.kpoint))
                # G_vectors(basis)[ind_ΔG] = G - G'
                ind_ΔG = index_G_vectors(op.basis, G - G′)
                if isnothing(ind_ΔG)
                    error("For full matrix construction, the FFT size must be " *
                          "large enough so that Hamiltonian applications are exact")
                end
                H[σ, i, σ, j] = pot_fourier[ind_ΔG] / sqrt(op.basis.model.unit_cell_volume)
            end
        end
    end
    H
end

@doc raw"""
Fourier space multiplication, like a kinetic energy term:
```math
(Hψ)(G) = {\rm multiplier}(G) ψ(G).
```
"""
struct FourierMultiplication{T <: Real, AT <: AbstractArray} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    multiplier::AT
end
function apply!(Hψ, op::FourierMultiplication, ψ)
    @views for σ = 1:op.basis.model.n_components
        Hψ.fourier[σ, :] .+= op.multiplier .* ψ.fourier[σ, :]
    end
    Hψ.fourier
end
function Array(op::FourierMultiplication)
    n_Gk = length(G_vectors(op.basis, op.kpoint))
    n_components = op.basis.model.n_components
    D = Diagonal(op.multiplier)
    reshape(kron(D, I(n_components)), n_components, n_Gk, n_components, n_Gk)
end

"""
Nonlocal operator in Fourier space in Kleinman-Bylander format,
defined by its projectors P matrix and coupling terms D:
Hψ = PDP' ψ.
"""
struct NonlocalOperator{T <: Real, PT, DT} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    # not typed, can be anything that supports PDP'ψ
    P::PT
    D::DT
end
function apply!(Hψ, op::NonlocalOperator, ψ)
    @views for σ = 1:op.basis.model.n_components
        Hψ.fourier[σ, :, :] .+= op.P * (op.D * (op.P' * ψ.fourier[σ, :, :]))
    end
    Hψ.fourier
end
function Array(op::NonlocalOperator)
    n_Gk = length(G_vectors(op.basis, op.kpoint))
    n_components = op.basis.model.n_components
    H = zeros(complex(eltype(op.basis)), n_components, n_Gk, n_components, n_Gk)
    for σ = 1:op.basis.model.n_components
        H[σ, :, σ, :] = op.P * op.D * op.P'
    end
    H
end

"""
Magnetic field operator A⋅(-i∇).
"""
struct MagneticFieldOperator{T <: Real, AT} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    Apot::AT  # Apot[α][i,j,k] is the A field in (Cartesian) direction α
end
function apply!(Hψ, op::MagneticFieldOperator, ψ)
    # TODO this could probably be better optimized
    @views for α = 1:3
        iszero(op.Apot[α]) && continue
        pα = [p[α] for p in Gplusk_vectors_cart(op.basis, op.kpoint)]
        ∂αψ_fourier = similar(ψ.fourier)
        for σ = 1:op.basis.model.n_components
            ∂αψ_fourier[σ, :] = pα .* ψ.fourier[σ, :]
        end
        ∂αψ_real = ifft(op.basis, op.kpoint, ∂αψ_fourier)
        for σ = 1:op.basis.model.n_components
            Hψ.real[σ, :, :, :] .+= op.Apot[α] .* ∂αψ_real[σ, :, :, :]
        end
    end
end
# TODO Implement  Array(op::MagneticFieldOperator)

@doc raw"""
Nonlocal "divAgrad" operator ``-½ ∇ ⋅ (A ∇)`` where ``A`` is a scalar field on the
real-space grid. The ``-½`` is included, such that this operator is a generalisation of the
kinetic energy operator (which is obtained for ``A=1``).
"""
struct DivAgradOperator{T <: Real, AT} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    A::AT
end
function apply!(Hψ, op::DivAgradOperator, ψ;
                ψ_scratch=zeros(complex(eltype(op.basis)),
                                op.basis.model.n_components, op.basis.fft_size...))
    # TODO: Performance improvements: Unscaled plans, avoid remaining allocations
    #       (which are only on the small k-point-specific Fourier grid
    G_plus_k = [[p[α] for p in Gplusk_vectors_cart(op.basis, op.kpoint)] for α = 1:3]
    @views for α = 1:3
        ∂αψ_fourier = similar(ψ.fourier)
        for σ = 1:op.basis.model.n_components
            ∂αψ_fourier[σ, :] = im .* G_plus_k[α] .* ψ.fourier[σ, :]
        end
        ∂αψ_real = ifft!(ψ_scratch, op.basis, op.kpoint, ∂αψ_fourier)
        for σ = 1:op.basis.model.n_components
            ∂αψ_real[σ, :, :, :] .*= op.A
        end
        A∇ψ = fft(op.basis, op.kpoint, ∂αψ_real)
        for σ = 1:op.basis.model.n_components
            Hψ.fourier[σ, :] .-= im .* G_plus_k[α] .* A∇ψ[σ, :] ./ 2
        end
    end
end
# TODO Implement  Array(op::DivAgrad)


# Optimize RFOs by combining terms that can be combined
function optimize_operators(ops)
    ops = [op for op in ops if !(op isa NoopOperator)]
    RSmults = [op for op in ops if op isa RealSpaceMultiplication]
    isempty(RSmults) && return ops
    nonRSmults = [op for op in ops if !(op isa RealSpaceMultiplication)]
    combined_RSmults = RealSpaceMultiplication(RSmults[1].basis,
                                               RSmults[1].kpoint,
                                               sum([op.potential for op in RSmults]))
    [nonRSmults..., combined_RSmults]
end
