using LinearMaps
using IterativeSolvers
using Statistics
import Base: @kwdef

# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the
# Hamiltonian at ρin These define the basic fix-point iteration, that are then combined with
# acceleration methods (eg anderson). For the mixing interface we use `δF = ρout - ρin` and
# `δρ = ρnext - ρin`, such that the mixing interface is
# `mix_density(mixing, basis, δF; kwargs...) -> δρ` with the user being assumed to add this
# to ρin to get ρnext. All these methods attempt to approximate the inverse Jacobian of the
# SCF step, ``J^-1 = (1 - χ0 (vc + K_{xc}))^-1``, where vc is the Coulomb and ``K_{xc}`` the
# exchange-correlation kernel. Note that "mixing" is sometimes used to refer to the combined
# process of formulating the fixed-point and solving it; we call "mixing" only the first part
# The notation in this file follows Herbst, Levitt arXiv:2009.01665

# Mixing can be done in the potential or the density. By default we assume
# the dielectric model is so simple that both types of mixing are identical.
# If mixing is done in the potential, the interface is
# `mix_potential(mixing, basis, δF; kwargs...) -> δV`
abstract type Mixing end
mix_potential(args...; kwargs...) = mix_density(args...; kwargs...)

@doc raw"""
Simple mixing: ``J^{-1} ≈ 1``
"""
struct SimpleMixing <: Mixing; end
mix_density(::SimpleMixing, ::PlaneWaveBasis, δF; kwargs...) = δF


@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{|G|^2}{k_{TF}^2 + |G|^2}``
where ``k_{TF}`` is the Thomas-Fermi wave vector. For spin-polarized calculations
by default the spin density is not preconditioned. Unless a non-default value
for ``ΔDOS_Ω`` is specified. This value should roughly be the expected difference in density
of states (per unit volume) between spin-up and spin-down.

Notes:
- Abinit calls ``1/k_{TF}`` the dielectric screening length (parameter *dielng*)
"""
@kwdef struct KerkerMixing <: Mixing
    # Default kTF parameter suggested by Kresse, Furthmüller 1996 (kTF=1.5Å⁻¹)
    # DOI 10.1103/PhysRevB.54.11169
    kTF::Real    = 0.8  # == sqrt(4π (DOS_α + DOS_β)) / Ω
    ΔDOS_Ω::Real = 0.0  # == (DOS_α - DOS_β) / Ω
end

@timing "KerkerMixing" function mix_density(mixing::KerkerMixing, basis::PlaneWaveBasis,
                                            δF; kwargs...)
    T      = eltype(δF)
    G²     = [sum(abs2, G) for G in G_vectors_cart(basis)]
    kTF    = T.(mixing.kTF)
    ΔDOS_Ω = T.(mixing.ΔDOS_Ω)

    # TODO This can be improved to use less copies for the new (α, β) interface

    # For Kerker the model dielectric written as a 2×2 matrix in spin components is
    #     1 - [-DOSα      0] * [1 1]
    #         [    0  -DOSβ]   [1 1] * (4π/G²)
    # which maps (δρα, δρβ)ᵀ to (δFα, δFβ)ᵀ and where DOSα and DOSβ is the density
    # of states per unit volume in the spin-up and spin-down channels. After basis
    # transformation to a mapping (δρtot, δρspin)ᵀ to (δFtot, δFspin)ᵀ this becomes
    #     [(G² + kTF²)    0]
    #     [ 4π * ΔDOS    G²] / G²
    # where we defined kTF² = 4π * (DOSα + DOSβ) and ΔDOS = DOSα - DOSβ.
    # Gaussian elimination on this matrix yields for the linear system ε δρ = δF
    #     δρtot  = G² δFtot / (G² + kTF²)
    #     δρspin = δFspin - 4π * ΔDOS / (G² + kTF²) δFtot

    δF_fourier     = r_to_G(basis, δF)
    δFtot_fourier  = total_density(δF_fourier)
    δFspin_fourier = spin_density(δF_fourier)

    δρtot_fourier = δFtot_fourier .* G² ./ (kTF.^2 .+ G²)
    δρtot = G_to_r(basis, δρtot_fourier)

    # Copy DC component, otherwise it never gets updated
    δρtot .+= mean(total_density(δF)) .- mean(δρtot)

    if basis.model.n_spin_components == 1
        ρ_from_total_and_spin(δρtot, nothing)
    else
        δρspin_fourier = @. δFspin_fourier - δFtot_fourier * (4π * ΔDOS_Ω) / (kTF^2 + G²)
        δρspin = G_to_r(basis, δρspin_fourier)
        ρ_from_total_and_spin(δρtot, δρspin)
    end
end


@doc raw"""
The same as [`KerkerMixing`](@ref), but the Thomas-Fermi wavevector is computed
from the current density of states at the Fermi level.
"""
@kwdef struct KerkerDosMixing <: Mixing
    adjust_temperature = IncreaseMixingTemperature()
end
@timing "KerkerDosMixing" function mix_density(mixing::KerkerDosMixing, basis::PlaneWaveBasis,
                                               δF; εF, eigenvalues, kwargs...)
    if iszero(basis.model.temperature)
        return mix_density(SimpleMixing(), basis, δF)
    else
        n_spin = basis.model.n_spin_components
        Ω = basis.model.unit_cell_volume
        temperature = mixing.adjust_temperature(basis.model.temperature; kwargs...)
        dos_per_vol  = compute_dos(εF, basis, eigenvalues; temperature) ./ Ω
        kTF  = sqrt(4π * sum(dos_per_vol))
        ΔDOS_Ω = n_spin == 2 ? dos_per_vol[1] - dos_per_vol[2] : 0.0
        mix_density(KerkerMixing(kTF=kTF, ΔDOS_Ω=ΔDOS_Ω), basis, δF)
    end
end

@doc raw"""
We use a simplification of the Resta model DOI 10.1103/physrevb.16.2717 and set
``χ_0(q) = \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_{TF}^2)}``
where ``C_0 = 1 - ε_r`` with ``ε_r`` being the macroscopic relative permittivity.
We neglect ``K_\text{xc}``, such that
``J^{-1} ≈ \frac{k_{TF}^2 - C_0 G^2}{ε_r k_{TF}^2 - C_0 G^2}``

By default it assumes a relative permittivity of 10 (similar to Silicon).
`εr == 1` is equal to `SimpleMixing` and `εr == Inf` to `KerkerMixing`.
The mixing is applied to ``ρ`` and ``ρ_\text{spin}`` in the same way.
"""
@kwdef struct DielectricMixing <: Mixing
    kTF::Real = 0.8
    εr::Real  = 10
end
@timing "DielectricMixing" function mix_density(mixing::DielectricMixing, basis::PlaneWaveBasis,
                                                δF; kwargs...)
    T = eltype(δF)
    εr = T(mixing.εr)
    kTF = T(mixing.kTF)
    εr == 1               && return mix_density(SimpleMixing(), basis, δF)
    εr > 1 / sqrt(eps(T)) && return mix_density(KerkerMixing(kTF=kTF), basis, δF)

    C0 = 1 - εr
    Gsq = [sum(abs2, G) for G in G_vectors_cart(basis)]
    δF_fourier = r_to_G(basis, δF)
    δρ = @. δF_fourier * (kTF^2 - C0 * Gsq) / (εr * kTF^2 - C0 * Gsq)
    δρ = G_to_r(basis, δρ)
    δρ .+= mean(δF) .- mean(δρ)
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
the same convention for parameters are used as in [`DielectricMixing`](@ref).
Additionally there is the real-space localization function `L(r)`.
For details see Herbst, Levitt 2020 arXiv:2009.01665

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `RPA`: Is the random-phase approximation used for the kernel (i.e. only Hartree kernel is
  used and not XC kernel)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES
"""
function HybridMixing(;εr=1.0, kTF=0.8, localization=identity,
                      adjust_temperature=IncreaseMixingTemperature(), kwargs...)
    χ0terms = [DielectricModel(εr=εr, kTF=kTF, localization=localization),
               LdosModel(;adjust_temperature)]
    χ0Mixing(; χ0terms=χ0terms, kwargs...)
end


@doc raw"""
The model for the susceptibility is
```math
\begin{aligned}
    χ_0(r, r') &= (-D_\text{loc}(r) δ(r, r') + D_\text{loc}(r) D_\text{loc}(r') / D)
\end{aligned}
```
where ``D_\text{loc}`` is the local density of states,
``D`` is the density of states. For details see Herbst, Levitt 2020 arXiv:2009.01665.

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `RPA`: Is the random-phase approximation used for the kernel (i.e. only Hartree kernel is
  used and not XC kernel)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES
"""
function LdosMixing(; adjust_temperature=IncreaseMixingTemperature(), kwargs...)
    χ0Mixing(; χ0terms=[LdosModel(;adjust_temperature)], kwargs...)
end


@doc raw"""
Generic mixing function using a model for the susceptibility composed of the sum of the `χ0terms`.
For valid `χ0terms` See the subtypes of `χ0Model`. The dielectric model is solved in
real space using a GMRES. Either the full kernel (`RPA=false`) or only the Hartree kernel
(`RPA=true`) are employed. `verbose=true` lets the GMRES run in verbose mode
(useful for debugging).
"""
@kwdef struct χ0Mixing <: Mixing
    RPA::Bool = true       # Use RPA, i.e. only apply the Hartree and not the XC Kernel
    χ0terms   = χ0Model[Applyχ0Model()]  # The terms to use as the model for χ0
    verbose::Bool = false   # Run the GMRES verbosely
    reltol::Float64 = 0.01  # Relative tolerance for GMRES
end

@views @timing "χ0Mixing" function mix_density(mixing::χ0Mixing, basis, δF; ρin, kwargs...)
    # Initialise χ0terms and remove nothings (terms that don't yield a contribution)
    χ0applies = filter(!isnothing, [χ₀(basis; ρin=ρin, kwargs...) for χ₀ in mixing.χ0terms])

    # If no applies left, do not bother running GMRES and directly do simple mixing
    isempty(χ0applies) && return mix_density(SimpleMixing(), basis, δF)

    # Solve (ε^†) δρ = δF with ε^† = (1 - χ₀ vc) and χ₀ given as the sum of the χ0terms
    devec(x) = reshape(x, size(δF))
    function dielectric_adjoint(δF)
        δF = devec(δF)
        # Apply Kernel (just vc for RPA and (vc + K_{xc}) if not RPA)
        δV = apply_kernel(basis, δF; ρ=ρin, RPA=mixing.RPA)
        δV .-= mean(δV)
        εδF = copy(δF)
        for apply_term! in χ0applies
            apply_term!(εδF, δV, -1)  # εδF .-= χ₀ * δV
        end
        εδF .-= mean(εδF)
        vec(εδF)
    end

    DC_δF = mean(δF)
    δF .-= DC_δF
    ε  = LinearMap(dielectric_adjoint, length(δF))
    δρ = devec(gmres(ε, vec(δF), verbose=mixing.verbose, reltol=mixing.reltol))
    δρ .+= DC_δF  # Set DC from δF
    δρ
end

@timing "χ0Mixing" function mix_potential(mixing::χ0Mixing, basis, δF::AbstractArray; ρin, kwargs...)
    # Initialise χ0terms and remove nothings (terms that don't yield a contribution)
    χ0applies = filter(!isnothing, [χ0(basis; ρin=ρin, kwargs...) for χ0 in mixing.χ0terms])

    # If no applies left, do not bother running GMRES and directly do simple mixing
    isempty(χ0applies) && return mix_potential(SimpleMixing(), basis, δF)

    # Note: Since in the potential-mixing version the χ₀ model is directly applied to δF
    #       (instead of first being "low-pass" filtered by the 1/G² in the Hartree kernel
    #       like in the density-mixing version), mix_potential is much more susceptible
    #       to having a good model. For example when using LdosMixing this means one needs
    #       to choose a good enough k-Point sampling / high enough smearing temperature.
    #       I also tried experimenting with some low-pass filtering in the LdosModel, but
    #       so far without a fully satisfactory result. As of now LdosMixing should be avoided
    #       with potential mixing.
    @warn("LdosMixing / χ0Mixing not yet fine-tuned for potential mixing. You're on your own. " *
          "Make sure to use sufficient k-Point sampling and maybe low-pass filtering.", maxlog=1)

    # Solve ε δV = δF with ε = (1 - vc χ₀) and χ₀ given as the sum of the χ0terms
    devec(x) = reshape(x, size(δF))
    function dielectric(δF)
        δF = devec(δF)

        δρ = zero(δF)
        for apply_term! in χ0applies
            apply_term!(δρ, δF)  # δρ .+= χ₀ * δF
        end
        δρ .-= mean(δρ)
        εδF = δF .- apply_kernel(basis, δρ; ρ=ρin, RPA=mixing.RPA)
        εδF .-= mean(εδF)
        vec(εδF)
    end

    DC_δF = mean(δF)
    δF .-= DC_δF
    ε  = LinearMap(dielectric, length(δF))
    δV = devec(gmres(ε, vec(δF), verbose=mixing.verbose, reltol=mixing.reltol))
    δV .+= DC_δF  # Set DC from δF
    δV
end


"""
Increase the temperature used for computing the SCF preconditioners. Initially the temperature
is increased by a `factor`, which is then smoothly lowered towards the temperature used
within the model as the SCF converges. Once the density change is below `above_ρdiff` the
mixing temperature is equal to the model temperature.
"""
function IncreaseMixingTemperature(;factor=25, above_ρdiff=1e-2, temperature_max=0.5)
    function callback(temperature; n_iter, ρin=nothing, ρout=nothing, info...)
        if iszero(temperature) || temperature > temperature_max
            return temperature
        elseif isnothing(ρin) || isnothing(ρout)
            return temperature
        elseif n_iter ≤ 1
            return factor * temperature
        end

        # Continuous piecewise linear function on a logarithmic scale
        # In [log(above_ρdiff), log(above_ρdiff) + switch_slope] it switches from 1 to factor
        switch_slope = 1
        ρdiff = norm(ρout .- ρin)
        enhancement = clamp(1 + (factor - 1) / switch_slope * log10(ρdiff / above_ρdiff), 1, factor)

        # Between SCF iterations temperature may never grow
        temperature = clamp(enhancement * temperature, temperature, temperature_max)
        temperature_max = temperature
        return temperature
    end
end
