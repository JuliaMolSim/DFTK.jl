### Linear operators operating on quantities in real or fourier space
# This is the optimized low-level interface. Use the functions in Hamiltonian for high-level usage

"""
Linear operators that act on tuples (real, fourier)
The main entry point is `apply!(out, op, in)` which performs the operation out += op*in
where out and in are named tuples (real, fourier)
They also implement mul! and Matrix(op) for exploratory use.
"""
abstract type RealFourierOperator end
# RealFourierOperator contain fields `basis` and `kpoint`

# Unoptimized fallback, intended for exploratory use only.
# For performance, call through Hamiltonian which saves FFTs.
function LinearAlgebra.mul!(Hψ::AbstractVector, op::RealFourierOperator, ψ::AbstractVector)
    ψ_real = G_to_r(op.basis, op.kpoint, ψ)
    Hψ_fourier = similar(ψ)
    Hψ_real = similar(ψ_real)
    Hψ_fourier .= 0
    Hψ_real .= 0
    apply!((real=Hψ_real, fourier=Hψ_fourier), op, (real=ψ_real, fourier=ψ))
    Hψ .= Hψ_fourier .+ r_to_G(op.basis, op.kpoint, Hψ_real)
    Hψ
end
function LinearAlgebra.mul!(Hψ::AbstractMatrix, op::RealFourierOperator, ψ::AbstractMatrix)
    @views for i = 1:size(ψ, 2)
        mul!(Hψ[:, i], op, ψ[:, i])
    end
    Hψ
end
Base.:*(op::RealFourierOperator, ψ) = mul!(similar(ψ), op, ψ)

"""
Noop operation: don't do anything.
Useful for energy terms that don't depend on the orbitals at all (eg nuclei-nuclei interaction).
"""
struct NoopOperator <: RealFourierOperator
    basis
    kpoint
end
apply!(Hψ, op::NoopOperator, ψ) = nothing
function Matrix(op::NoopOperator)
    zeros(eltype(op.basis), length(G_vectors(op.kpoint)), length(G_vectors(op.kpoint)))
end

"""
Real space multiplication by a potential:
(Hψ)(r) V(r) ψ(r)
"""
struct RealSpaceMultiplication <: RealFourierOperator
    basis
    kpoint
    potential::AbstractArray
end
@timing_seq "apply RealSpaceMultiplication" function apply!(Hψ, op::RealSpaceMultiplication, ψ)
    Hψ.real .+= op.potential .* ψ.real
end
function Matrix(op::RealSpaceMultiplication)
    # V(G,G') = <eG|V|eG'> = 1/sqrt(Ω) <e_{G-G'}|V>
    pot_fourier = r_to_G(op.basis, complex.(op.potential))
    npw = length(G_vectors(op.kpoint))
    H = zeros(complex(eltype(op.basis)), npw, npw)
    for i = 1:npw
        G = G_vectors(op.kpoint)[i]
        for j = 1:npw
            Gp = G_vectors(op.kpoint)[j]
            ΔG = G-Gp
            # G_vectors(basis)[ind] = ΔG
            ind = index_G_vectors(op.basis, ΔG)
            if ind === nothing
                error("For full matrix construction, the FFT size must be " *
                      "large enough so that Hamiltonian applications are exact")
            end
            H[i, j] = pot_fourier[ind] / sqrt(op.basis.model.unit_cell_volume)
        end
    end
    H
end

"""
Fourier space multiplication, like a kinetic energy term:
(Hψ)(G) = multiplier(G) ψ(G)
"""
struct FourierMultiplication <: RealFourierOperator
    basis
    kpoint
    multiplier::AbstractArray
end
@timing_seq "apply FourierMultiplication" function apply!(Hψ, op::FourierMultiplication, ψ)
    Hψ.fourier .+= op.multiplier .* ψ.fourier
end
Matrix(op::FourierMultiplication) = Array(Diagonal(op.multiplier))

"""
Nonlocal operator in Fourier space in Kleinman-Bylander format,
defined by its projectors P matrix and coupling terms D:
Hψ = PDP' ψ
"""
struct NonlocalOperator <: RealFourierOperator
    basis
    kpoint
    # not typed, can be anything that supports PDP'ψ
    P
    D
end
@timing_seq "apply NonlocalOperator" function apply!(Hψ, op::NonlocalOperator, ψ)
    Hψ.fourier .+= op.P * (op.D * (op.P' * ψ.fourier))
end
Matrix(op::NonlocalOperator) = op.P * op.D * op.P'

"""
Magnetic field operator A⋅(-i∇).
"""
struct MagneticFieldOperator{T <: Real} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    Apot  # Apot[α][i,j,k] is the A field in (cartesian) direction α
end
@timing_seq "apply MagneticFieldOperator" function apply!(Hψ, op::MagneticFieldOperator, ψ)
    # TODO this could probably be better optimized
    for α = 1:3
        all(op.Apot[α] .== 0) && continue
        pα = [(op.basis.model.recip_lattice*(G + op.kpoint.coordinate))[α] for G in G_vectors(op.kpoint)]
        ∂αψ_fourier = pα .* ψ.fourier
        ∂αψ_real = G_to_r(op.basis, op.kpoint, ∂αψ_fourier)
        Hψ.real .+= op.Apot[α] .* ∂αψ_real
    end
end
Matrix(op::MagneticFieldOperator) = error("Not implemented")


# Optimize RFOs by combining terms that can be combined
function optimize_operators_(ops)
    RSmults = [op for op in ops if op isa RealSpaceMultiplication]
    isempty(RSmults) && return ops
    nonRSmults = [op for op in ops if !(op isa RealSpaceMultiplication)]
    combined_RSmults = RealSpaceMultiplication(RSmults[1].basis,
                                               RSmults[1].kpoint,
                                               sum([op.potential for op in RSmults]))
    [nonRSmults..., combined_RSmults]
end
