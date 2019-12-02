# Mixing rules: ρin, ρout => ρnext
# These define the basic fix-point iteration, that are then combined with acceleration methods (eg anderson)

struct KerkerMixing{T <: Real}
    q0::T
end
KerkerMixing() = KerkerMixing(1)
function mix(m::KerkerMixing, basis, ρin::Density, ρout::Density)
    Gsq = [sum(abs2, basis.model.recip_lattice * G)
           for G in basis_Cρ(basis)]
    ρin = fourier(ρin)
    ρout = fourier(ρout)
    ρnext = @. ρin + (ρout - ρin) * Gsq / (m.q0 + Gsq)
    density_from_fourier(basis, ρnext)
end

struct SimpleMixing{T <: Real}
    α::T
end
SimpleMixing() = SimpleMixing(1)
function mix(m::SimpleMixing, basis, ρin::Density, ρout::Density)
    if m.α == 1
        return ρout # optimization
    else
        ## TODO optimize this by defining broadcasting directly on density objects?
        density_from_real(basis, real(ρin) .+ m.α .* (real(ρout) .- real(ρin)))
    end
end

