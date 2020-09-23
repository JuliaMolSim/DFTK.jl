using LinearMaps
using IterativeSolvers
import Base: @kwdef

# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the
# Hamiltonian at ρin These define the basic fix-point iteration, that are then combined with
# acceleration methods (eg anderson).
# All these methods attempt to approximate the inverse Jacobian of the SCF step,
# ``J^-1 = (1 - χ0 (vc + K_{xc}))^-1``, where vc is the Coulomb and ``K_{xc}`` the
# exchange-correlation kernel. Note that "mixing" is sometimes used to refer to the combined
# process of formulating the fixed-point and solving it; we call "mixing" only the first part
# The notation in this file follows Herbst, Levitt arXiv:2009.01665
#
# The interface is `mix(m; basis, ρin, ρout, kwargs...) -> ρnext`

@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{α |G|^2}{k_{TF}^2 + |G|^2}``
where ``k_{TF}`` is the Thomas-Fermi wave vector.

Notes:
- Abinit calls ``1/k_{TF}`` the dielectric screening length (parameter *dielng*)
"""
@kwdef struct KerkerMixing
    # Default parameters suggested by Kresse, Furthmüller 1996 (α=0.8, kTF=1.5 Ǎ^{-1})
    # DOI 10.1103/PhysRevB.54.11169
    α::Real = 0.8
    kTF::Real = 0.8
end
@timing "mixing Kerker" function mix(mixing::KerkerMixing, basis, ρin::RealFourierArray,
                                     ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    ρin = ρin.fourier
    ρout = ρout.fourier
    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * Gsq / (T(mixing.kTF)^2 + Gsq)
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
@timing "mixing Simple" function mix(mixing::SimpleMixing, basis, ρin::RealFourierArray,
                                     ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    if mixing.α == 1
        return ρout
    else
        ρin + T(mixing.α) * (ρout - ρin)
    end
end

@doc raw"""
We use a simplification of the Resta model DOI 10.1103/physrevb.16.2717 and set
``χ_0(q) = \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_{TF}^2)}``
where ``C_0 = 1 - ε_r`` with ``ε_r`` being the macroscopic relative permittivity.
We neglect ``K_\text{xc}``, such that
``J^{-1} ≈ α \frac{k_{TF}^2 - C_0 G^2}{ε_r k_{TF}^2 - C_0 G^2}``

By default it assumes a relative permittivity of 10 (similar to Silicon).
`εr == 1` is equal to `SimpleMixing` and `εr == Inf` to `KerkerMixing`.
"""
@kwdef struct DielectricMixing
    α::Real = 0.8
    kTF::Real = 0.8
    εr::Real = 10
end
@timing "mixing Dielectric" function mix(mixing::DielectricMixing, basis, ρin::RealFourierArray,
                                         ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kTF = T(mixing.kTF)
    εr == 1               && return mix(SimpleMixing(α=mixing.α), basis, ρin, ρout)
    εr > 1 / sqrt(eps(T)) && return mix(KerkerMixing(α=mixing.α, kTF=kTF), basis, ρin, ρout)

    ρin = ρin.fourier
    ρout = ρout.fourier
    C0 = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]

    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * (kTF^2 - C0 * Gsq) / (εr * kTF^2 - C0 * Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext)
end


@doc raw"""
The model for the susceptibility is
```math
\begin{aligned}
    χ_0(r, r') &= (-D_\text{loc}(r) δ(r, r') + D_\text{loc}(r) D_\text{loc}(r') / D) \\
    &+ \sqrt{L(x)} \text{IFFT} \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_{TF}^2)} \text{FFT} \sqrt{L(x)}
\end{aligned}
```
where ``C_0 = 1 - ε_r``, ``D_\text{loc}`` is the local density of states,
``D`` is the density of states and
the same convention for parameters are used
as in `DielectricMixing`.
Additionally there is the real-space localisation function `L(r)`.
For details see Herbst, Levitt 2020 arXiv:2009.01665
"""
@kwdef struct HybridMixing
    α::Real = 0.8
    εr::Real = 1.0
    kTF::Real = 0.8
    localisation::Function = identity  # `L(r)` with `r` in fractional real-space coordinates
    RPA::Bool = true       # Use RPA, i.e. only apply the Hartree and not the XC Kernel
    verbose::Bool = false  # Run the GMRES verbosely
end

@timing "mixing Hybrid" function mix(mixing::HybridMixing, basis, ρin::RealFourierArray,
                                     ρout::RealFourierArray; εF, eigenvalues, ψ, kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kTF = T(mixing.kTF)
    C0 = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

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

        # Apply Kernel (just vc for RPA and (vc + K_{xc}) if not RPA)
        δV = apply_kernel(basis, from_real(basis, δρ), ρ=ρin, RPA=mixing.RPA)
        δV.real .-= sum(δV.real) / length(δV.real)  # set DC to zero

        # Apply Dielectric term of χ0
        if !iszero(C0)
            loc_δV = apply_sqrtL(δV).fourier
            dielectric_loc_δV =  @. C0 * kTF^2 * Gsq / 4T(π) / (kTF^2 - C0 * Gsq) * loc_δV
            Jδρ .-= apply_sqrtL(from_fourier(basis, dielectric_loc_δV)).real
        end

        # Apply LDOS term of χ0
        if ldos !== nothing && maximum(abs, ldos) > eps(real(eltype(ldos)))
            Jδρ .-= (-ldos .* δV.real
                     .+ sum(ldos .* δV.real) .* dVol .* ldos ./ (sum(ldos) .* dVol))
        end

        # Zero DC component and return
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
