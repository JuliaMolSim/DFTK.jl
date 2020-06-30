using LinearMaps
using IterativeSolvers
import Base: @kwdef

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
Kerker mixing: ``J^{-1} ≈ \frac{α G^2}{k_F^2 + G^2}``
where ``k_F`` is the Thomas-Fermi wave vector.

Notes:
  - Abinit calls ``1/k_F`` the dielectric screening length (parameter *dielng*)
"""
@kwdef struct KerkerMixing
    # Default parameters suggested by Kresse, Furthmüller 1996 (α=0.8, kF=1.5 Ǎ^{-1})
    # DOI 10.1103/PhysRevB.54.11169
    α::Real = 0.8
    kF::Real = 0.8
end
function mix(mixing::KerkerMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    ρin = ρin.fourier
    ρout = ρout.fourier
    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * Gsq / (T(mixing.kF)^2 + Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext)
end

@doc raw"""
Simple mixing: ``J^{-1} ≈ α``
"""
@kwdef struct SimpleMixing
    α::Real = 0.8
end
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
``χ_0(q) = \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_F^2)}
where ``C_0 = 1 - ε_r`` with ``ε_r`` being the macroscopic relative permittivity.
We neglect ``f_\text{xc}``, such that
``J^{-1} ≈ α \frac{k_F^2 - C_0 G^2}{ε_r k_F^2 - C_0 G^2}``

By default it assumes a relative permittivity of 10 (similar to Silicon).
`εr == 1` is equal to `SimpleMixing` and `εr == Inf` to `KerkerMixing`.
"""
@kwdef struct RestaMixing
    α::Real = 0.8
    εr::Real = 0.8
    kF::Real = 10
end
function mix(mixing::RestaMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kF = T(mixing.kF)
    εr == 1               && return mix(SimpleMixing(α=α), basis, ρin, ρout)
    εr > 1 / sqrt(eps(T)) && return mix(KerkerMixing(α=α, kF=kF), basis, ρin, ρout)

    ρin = ρin.fourier
    ρout = ρout.fourier
    C0 = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]

    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * (kF^2 - C0 * Gsq) / (εr * kF^2 - C0 * Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext)
end


@doc raw"""
The model for the susceptibility is
```math
χ_0(r, r') = (-LDOS(εF, r) δ(r, r') + LDOS(εF, r) LDOS(εF, r') / DOS(εF))
           + \sqrt{L(x)} IFFT \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_F^2)} FFT \sqrt{L(x)}
```
where ``C_0 = 1 - ε_r`` and the same convention for parameters is used as in `RestaMixing`.
Additionally there is the real-space localisation function `L(r)`.
"""
@kwdef struct HybridMixing
    α::Real = 0.8
    εr::Real = 10
    kF::Real = 0.8
    localisation::Function = identity  # `L(r)` with `r` in fractional real-space coordinates
    rpa::Bool = true       # Use RPA, i.e. only apply the Hartree and not the XC Kernel
    verbose::Bool = false  # Run the GMRES verbosely
end

function mix(mixing::HybridMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray;
             εF, eigenvalues, ψ, kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kF = T(mixing.kF)
    C0 = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]

    apply_sqrtL = identity
    if mixing.localisation != identity
        sqrtL = sqrt.(mixing.localisation.(r_vectors(basis)))
        apply_sqrtL = x -> from_real(basis, sqrtL .* x.real)
    end

    # Compute the LDOS if required
    ldos = nothing
    if basis.model.temperature > 0
        ldos = LDOS(εF, basis, eigenvalues, ψ)
    end

    # Solve J Δρ = ΔF with J = (1 - χ0 vc) and χ_0 given as in the docstring of the class
    ΔF = ρout.real - ρin.real
    devec(x) = reshape(x, size(ρin))
    function Jop(x)
        δρ = devec(x)
        Jδρ = copy(δρ)

        # Apply Kernel (just vc for RPA and (vc + fxc) if not RPA)
        δV = apply_kernel(basis, from_real(basis, δρ), ρ=ρin, rpa=mixing.rpa)

        # Apply Resta term of χ0
        loc_δV = apply_sqrtL(δV).fourier
        resta_loc_δV =  @. C0 * kF^2 * Gsq / 4T(π) / (kF^2 - C0 * Gsq) * loc_δV
        Jδρ .-= apply_sqrtL(from_fourier(basis, resta_loc_δV)).real

        # Apply LDOS term of χ0
        if ldos !== nothing
            Jδρ .-= (-ldos .* δV.real .+ sum(ldos .* δV.real) .* ldos ./ sum(ldos))
        end

        # Poor man's zero DC component before return
        vec(Jδρ .-= sum(Jδρ) / length(Jδρ))
    end
    J = LinearMap(Jop, length(ρin))
    x = gmres(J, ΔF, verbose=mixing.verbose)
    Δρ = devec(x)
    ρnext = real(@. ρin.real + T(mixing.α) * Δρ)

    # Set DC from ρout
    ρnext .+= (sum(ρout.real) - sum(ρnext)) / length(ρnext)
    from_real(basis, ρnext)
end
