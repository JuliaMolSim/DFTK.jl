@doc raw"""
Kinetic energy:
```math
1/2 ∑_n f_n ∫ |∇ψ_n|^2 * {\rm blowup}(-i∇Ψ_n).
```
"""
Base.@kwdef struct Kinetic{F}
    scaling_factor::Real = 1
    blowup::F = BlowupIdentity()  # Blow-up to smooth energy bands.
end

(kin::Kinetic)(basis) = TermKinetic(basis, kin.scaling_factor, kin.blowup)
function Base.show(io::IO, kin::Kinetic)
    bup = kin.blowup isa BlowupIdentity ? "" : ", blowup=$(kin.blowup)"
    fac = isone(kin.scaling_factor) ? "" : ", scaling_factor=$(kin.scaling_factor)"
    print(io, "Kinetic($bup$fac)")
end

struct TermKinetic <: Term
    scaling_factor::Real  # scaling factor, absorbed into kinetic_energies
    # kinetic energies 1/2(k+G)^2 *blowup(|k+G|, Ecut) for each k-point.
    kinetic_energies::Vector{<:AbstractVector}
end
function TermKinetic(basis::PlaneWaveBasis{T}, scaling_factor, blowup) where {T}
    kinetic_energies = [kinetic_energy(blowup, scaling_factor, basis.Ecut,
                                       Gplusk_vectors_cart(basis, kpt))
                        for kpt in basis.kpoints]
    TermKinetic(T(scaling_factor), kinetic_energies)
end

function kinetic_energy(blowup, scaling_factor, Ecut, p::AbstractArray{Vec3{T}}) where {T}
    map(p) do pk
        T(scaling_factor) * norm2(pk) / 2 * blowup(norm(pk), Ecut)
    end
end
function kinetic_energy(kin::Kinetic, Ecut, p)
    kinetic_energy(kin.blowup, kin.scaling_factor, Ecut, p)
end

@timing "ene_ops: kinetic" function ene_ops(term::TermKinetic, basis::PlaneWaveBasis{T},
                                            ψ, occupation; kwargs...) where {T}
    ops = [FourierMultiplication(basis, kpoint, term.kinetic_energies[ik])
           for (ik, kpoint) in enumerate(basis.kpoints)]
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(Inf), ops)
    end
    occupation = [to_cpu(occk) for occk in occupation]

    E = zero(T)
    for (ik, ψk) in enumerate(ψ)
        for iband = 1:size(ψk, 2)
            ψnk = @views ψk[:, iband]
            E += (basis.kweights[ik] * occupation[ik][iband]
                  * real(dot(ψnk, Diagonal(term.kinetic_energies[ik]), ψnk)))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (; E, ops)
end


"""
Default blow-up corresponding to the standard kinetic energies.
"""
struct BlowupIdentity end
(blowup::BlowupIdentity)(x, Ecut) = one(x)


"""
Blow-up function as proposed in [arXiv:2210.00442](https://arxiv.org/abs/2210.00442).
The blow-up order of the function is fixed to ensure C^2 regularity of the energies bands
away from crossings and Lipschitz continuity at crossings.
"""
struct BlowupCHV end
function (blowup::BlowupCHV)(y::T, Ecut) where {T}
    Ekin = y^2 / 2
    x = y / √(2Ecut)  # x in [0,1]
    x1, x2 = T(0.85), T(0.90)  # Interval of interpolation

    # Define blow-up part
    Ca_opt = 0.013952310177257383  # optimized to best match the x->x^2 curve
    blowup_part(x) = Ca_opt / (1-x)^2

    if (0 ≤ x < x1)
        return one(T)
    elseif (x1 ≤ x < x2) # smooth interpolation between 1 and blowup_part
        f(x::T) = iszero(x) ? zero(T) : exp(-1 / x)
        smooth_step(x) = f((x-x1)/(x2-x1)) / (f((x-x1)/(x2-x1)) + f(1 - (x-x1)/(x2-x1)))
        return (Ecut/Ekin) * ((1-smooth_step(x)) * x^2 + smooth_step(x) * blowup_part(x))
    else
        return (Ecut/Ekin) * blowup_part(x)
    end
end


"""
Blow-up function as used in Abinit.
"""
Base.@kwdef struct BlowupAbinit
    Ecutsm::Float64 = 0.5  # Recommended value in Abinit documentation
end
function (blowup::BlowupAbinit)(y::T, Ecut) where {T}
    Ekin   = y^2/2
    Ecutsm = Ecut * blowup.Ecutsm
    @assert Ecutsm < Ecut

    if y ≤ sqrt(2 * (Ecut - Ecutsm))
        return one(T)
    else
        x = (Ecut - Ekin) / Ecutsm
        1/(x^2 * (3 + x - 6x^2 + 3x^2))
    end
end
