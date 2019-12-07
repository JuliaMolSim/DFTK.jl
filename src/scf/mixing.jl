# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the Hamiltonian at ρin
# These define the basic fix-point iteration, that are then combined with acceleration methods (eg anderson)
# All these methods attempt to approximate the inverse jacobian of the SCF step, J^-1 = (1 - χ0 vc)^-1
# Note that "mixing" is sometimes used to refer to the combined process of formulating the fixed-point
# and solving it; we call "mixing" only the first part

# The interface is `mix(m, basis, ρin, ρout) -> ρnext`

"""
Kerker mixing: J^-1 ≈ α*G^2/(G0^2 + G^2)
"""
struct KerkerMixing{T <: Real}
    α::T
    G0::T
end
KerkerMixing() = KerkerMixing(1, 1)
function mix(m::KerkerMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray)
    Gsq = [sum(abs2, basis.model.recip_lattice * G)
           for G in basis_Cρ(basis)]
    ρin = ρin.fourier
    ρout = ρout.fourier
    ρnext = @. ρin + m.α * (ρout - ρin) * Gsq / (m.G0^2 + Gsq)
    from_fourier(basis, ρnext; isreal=true)
end

"""
Simple mixing: J^-1 ≈ α
"""
struct SimpleMixing{T <: Real}
    α::T
end
SimpleMixing() = SimpleMixing(1)
function mix(m::SimpleMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray)
    if m.α == 1
        return ρout # optimization
    else
        ρin + m.α * (ρout - ρin)
    end
end
