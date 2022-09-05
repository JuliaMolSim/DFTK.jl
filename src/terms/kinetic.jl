"""
Kinetic energy: 1/2 sum_n f_n ∫ |∇ψn|^2 (standard)
             or 1/2 sum_n f_n ∫ blowup(-i∇Ψ)^2 (modified kinetic term).
"""
Base.@kwdef struct Kinetic{F}
    scaling_factor::Real = 1
    blowup::F = BlowupKineticEnergy() # blow-up function to smooth energy bands. Defaut is x↦x^2.
end

(kin::Kinetic)(basis) = TermKinetic(basis, kin.scaling_factor, kin.blowup)
function Base.show(io::IO, kin::Kinetic)
    fac = isone(kin.scaling_factor) ? "" : ", scaling_factor=$scaling_factor"
    print(io, "Kinetic($fac)")
end

"""
Define different energy cutoff strategies.
ADD REF when the paper is published.
"""
struct BlowupKineticEnergy end
struct BlowupCHV end

"""
Default blow-up, x->x^2, corresponding to the standard kinetic energies.
"""
(blowup::BlowupKineticEnergy)(x) = x^2

"""
Blow-up function as proposed in [REF paper Cancès, Hassan, Vidal to be submitted]
The blow-up order of the function is fixed to ensure C^2 regularity of the energies bands
away from crossings and Lipschitz continuity at crossings.
"""
@inline function (blowup::BlowupCHV)(x)
    @assert (0 ≤ x < 1) "The blow-up function is defined on [0,1)"
    x1, x2 = 0.85, 0.90 # Interval of interpolation

    # Define blow-up part
    Ca_opt = 0.013952310177257383 # optimized to match the x->x^2 curve the most
    blowup_part(x) = Ca_opt/( (1-x)^(2) )

    if (0 ≤ x < x1)
        return x^2
    elseif (x1 ≤ x < x2)
        f(x) = (x==0) ? 0 : exp(-1/x)
        step(x) = f((x-x1)/(x2-x1)) / (f((x-x1)/(x2-x1)) + f(1-(x-x1)/(x2-x1)))
        return (1-step(x))*x^2 + step(x)*blowup_part(x)
    else
        return blowup_part(x)
    end
    Inf # Handle |G+k|^2 = E_cut case
end

struct TermKinetic <: Term
    scaling_factor::Real  # scaling factor, absorbed into kinetic_energies
    # kinetic energies 1/2(k+G)^2 (or Ecut*blowup(|k+G|/√(2Ecut)) for energy
    # cutoff smearing methods) for each k-point.
    kinetic_energies::Vector{<:AbstractVector}
end
function TermKinetic(basis::PlaneWaveBasis{T}, scaling_factor, blowup) where {T}
    Ecut = basis.Ecut
    kinetic_energies = [[T(scaling_factor) * Ecut * blowup(norm(Gk)/√(2Ecut))
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
