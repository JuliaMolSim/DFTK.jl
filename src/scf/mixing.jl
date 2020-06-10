# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the Hamiltonian at ρin
# These define the basic fix-point iteration, that are then combined with acceleration methods (eg anderson)
# All these methods attempt to approximate the inverse jacobian of the SCF step, J^-1 = (1 - χ0 vc)^-1
# Note that "mixing" is sometimes used to refer to the combined process of formulating the fixed-point
# and solving it; we call "mixing" only the first part

# The interface is `mix(m, basis, ρin, ρout) -> ρnext`

using LinearMaps
using IterativeSolvers

@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{α G^2}{(k_F^2 + G^2}``
where ``k_F`` is the Thomas-Fermi screening constant.

Notes:
  - Abinit calls ``1/k_F`` the dielectric screening length (parameter *dielng*)
"""
struct KerkerMixing
    α::Real
    kF::Real
end
# Default parameters suggested by Kresse, Furthmüller 1996 (α=0.8, kF=1.5 Ǎ^{-1})
# DOI 10.1103/PhysRevB.54.11169
KerkerMixing(;α=0.8, kF=0.8) = KerkerMixing(α, kF)
function mix(mixing::KerkerMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    ρin = ρin.fourier
    ρout = ρout.fourier
    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * Gsq / (T(mixing.kF)^2 + Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext; assume_real=true)
end

"""
Simple mixing: J^-1 ≈ α
"""
struct SimpleMixing
    α::Real
end
SimpleMixing(;α=1) = SimpleMixing(α)
function mix(mixing::SimpleMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    if mixing.α == 1
        return ρout
    else
        T = eltype(basis)
        ρin + T(mixing.α) * (ρout - ρin)
    end
end
