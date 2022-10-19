"""
Kinetic energy: 1/2 sum_n f_n ∫ |∇ψn|^2 * blowup(-i∇Ψ).
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
    kinetic_energies = [[T(scaling_factor) * sum(abs2, Gk)/2 * blowup(norm(Gk), basis.Ecut)
                         for Gk in Gplusk_vectors_cart(basis, kpt)] for kpt in basis.kpoints]
    TermKinetic(T(scaling_factor), kinetic_energies)
end

@timing "ene_ops: kinetic" function ene_ops(term::TermKinetic, basis::PlaneWaveBasis{T},
                                            ψ, occ; kwargs...) where {T}
    ops = [FourierMultiplication(basis, kpoint, term.kinetic_energies[ik])
           for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(Inf), ops=ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[ik], 2)
            ψnk = @views ψ[ik][:, iband]
            E += (basis.kweights[ik] * occ[ik][iband]
                  * real(dot(ψnk, Diagonal(term.kinetic_energies[ik]), ψnk)))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (E=E, ops=ops)
end


"""
Default blow-up corresponding to the standard kinetic energies.
"""
struct BlowupIdentity end
(blowup::BlowupIdentity)(x, Ecut) = one(x)


"""
Blow-up function as proposed in [https://arxiv.org/abs/2210.00442]
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
