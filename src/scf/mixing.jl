using LinearMaps
using IterativeSolvers

# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the
# Hamiltonian at ρin These define the basic fix-point iteration, that are then combined with
# acceleration methods (eg anderson).
# All these methods attempt to approximate the inverse Jacobian of the SCF step,
# ``J^-1 = (1 - χ0 (vc + fxc))^-1``, where vc is the Coulomb and fxc the
# exchange-correlation kernel. Note that "mixing" is sometimes used to refer to the combined
# process of formulating the fixed-point and solving it; we call "mixing" only the first part
#
# The interface is `mix(m; basis, ρin, ρout, kwargs...) -> ρnext`

@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{α G^2}{(k_F^2 + G^2}``
where ``k_F`` is the Thomas-Fermi wave vector.

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

@doc raw"""
Simple mixing: ``J^{-1} ≈ α``
"""
struct SimpleMixing
    α::Real
end
SimpleMixing(;α=1) = SimpleMixing(α)
function mix(mixing::SimpleMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    if mixing.α == 1
        return ρout
    else
        ρin + T(mixing.α) * (ρout - ρin)
    end
end

@doc raw"""
We use a simplification of the Resta model DOI 10.1103/physrevb.16.2717 and set
``χ_0(q) = \frac{C G^2}{4π (1 - C G^2 / k_F^2)}
where ``C = 1 - ε_r`` with ``ε_r`` being the macroscopic relative permittivity.
We neglect ``f_\text{xc}``, such that
``J^{-1} ≈ α \frac{k_F^2 - C G^2}{ε_r k_F^2 - C G^2}``

By default it assumes a relative permittivity of 10 (similar to Silicon).
`εr == 1` is equal to `SimpleMixing` and `εr == Inf` to `KerkerMixing`.
"""
struct RestaMixing
    α::Real
    εr::Real
    kF::Real
end
RestaMixing(;α=0.8, kF=0.8, εr=10) = RestaMixing(α, εr, kF)
function mix(mixing::RestaMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kF = T(mixing.kF)
    εr == 1               && return mix(SimpleMixing(α=α), basis, ρin, ρout)
    εr > 1 / sqrt(eps(T)) && return mix(KerkerMixing(α=α, kF=kF), basis, ρin, ρout)

    ρin = ρin.fourier
    ρout = ρout.fourier
    C = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]

    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * (kF^2 - C * Gsq) / (εr * kF^2 - C * Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext; assume_real=true)
end
