using LinearMaps
using IterativeSolvers
using Statistics
import Base: @kwdef

# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the
# Hamiltonian at ρin These define the basic fix-point iteration, that are then combined with
# acceleration methods (eg anderson). For the mixing interface we use `δF = ρout - ρin` and
# `δρ = ρnext - ρin`, such that the mixing interface is `mix(mixing, basis, δF; kwargs...) -> δρ`
# with the user being assumed to add this to ρin to get ρnext.
# All these methods attempt to approximate the inverse Jacobian of the SCF step,
# ``J^-1 = (1 - χ0 (vc + K_{xc}))^-1``, where vc is the Coulomb and ``K_{xc}`` the
# exchange-correlation kernel. Note that "mixing" is sometimes used to refer to the combined
# process of formulating the fixed-point and solving it; we call "mixing" only the first part
# The notation in this file follows Herbst, Levitt arXiv:2009.01665

@doc raw"""
Simple mixing: ``J^{-1} ≈ α``
"""
@kwdef struct SimpleMixing
    α::Real = 0.8
end
@timing "SimpleMixing" function mix(mixing::SimpleMixing, ::PlaneWaveBasis, δFs...; kwargs...)
    T = eltype(δFs[1])
    map(δFs) do δF  # Apply to both δF{total} and δF{spin} in the same way 
        isnothing(δF) && return nothing
        T(mixing.α) * δF
    end
end


@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{α |G|^2}{k_{TF}^2 + |G|^2}``
where ``k_{TF}`` is the Thomas-Fermi wave vector. For spin-polarized calculations
by default the spin density is not preconditioned. Unless a non-default value
for ``ΔDOS`` is specified. This value should roughly be the expected difference in density
of states (per unit volume) between spin-up and spin-down.

Notes:
- Abinit calls ``1/k_{TF}`` the dielectric screening length (parameter *dielng*)
"""
@kwdef struct KerkerMixing
    # Default parameters suggested by Kresse, Furthmüller 1996 (α=0.8, kTF=1.5Å⁻¹)
    # DOI 10.1103/PhysRevB.54.11169
    α::Real    = 0.8
    kTF::Real  = 0.8  # == sqrt(4π (DOS_α + DOS_β) / Ω)
    ΔDOS::Real = 0.0  # == (DOS_α - DOS_β) / Ω
end

@timing "KerkerMixing" function mix(mixing::KerkerMixing, basis::PlaneWaveBasis,
                                    δF::RealFourierArray, δFspin=nothing; kwargs...)
    T    = eltype(δF)
    G²   = [sum(abs2, G) for G in G_vectors_cart(basis)]
    kTF  = T.(mixing.kTF)
    ΔDOS = T.(mixing.ΔDOS)

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

    δρtot    = @. δF.fourier * G² / (kTF^2 + G²)
    δρtot[1] = δF.fourier[1]  # Copy DC component, otherwise it never gets updated
    if basis.model.n_spin_components == 1
        from_fourier(basis, T(mixing.α) * δρtot), nothing
    else
        δρspin = @. δFspin.fourier - δF.fourier * (4π * ΔDOS) / (kTF^2 + G²)
        from_fourier(basis, T(mixing.α) * δρtot), from_fourier(basis, T(mixing.α) * δρspin)
    end
end

@doc raw"""
The same as [`KerkerMixing`](@ref), but the Thomas-Fermi wavevector is computed
from the current density of states at the Fermi level.
"""
@kwdef struct KerkerDosMixing
    α::Real = 0.8
end
@timing "KerkerDosMixing" function mix(mixing::KerkerDosMixing, basis::PlaneWaveBasis,
                                       δF, δFspin=nothing; εF, ψ, eigenvalues, kwargs...)
    if basis.model.temperature == 0
        return mix(SimpleMixing(α=mixing.α), basis, δF, δFspin)
    else
        n_spin = basis.model.n_spin_components
        Ω = basis.model.unit_cell_volume
        dos  = [DOS(εF, basis, eigenvalues, spins=[σ]) / Ω for σ in 1:n_spin]
        kTF  = sqrt(4π * sum(dos))
        ΔDOS = n_spin == 2 ? dos[1] - dos[2] : 0.0
        mix(KerkerMixing(α=mixing.α, kTF=kTF, ΔDOS=ΔDOS), basis, δF, δFspin)
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
The mixing is applied to ``ρ`` and ``ρ_\text{spin}`` in the same way.
"""
@kwdef struct DielectricMixing
    α::Real   = 0.8
    kTF::Real = 0.8
    εr::Real  = 10
end
@timing "DielectricMixing" function mix(mixing::DielectricMixing, basis::PlaneWaveBasis,
                                        δFs...; kwargs...)
    T = eltype(δFs[1])
    εr = T(mixing.εr)
    kTF = T(mixing.kTF)
    εr == 1               && return mix(SimpleMixing(α=mixing.α), basis, δFs...)
    εr > 1 / sqrt(eps(T)) && return mix(KerkerMixing(α=mixing.α, kTF=kTF), basis, δFs...)

    C0 = 1 - εr
    Gsq = [sum(abs2, G) for G in G_vectors_cart(basis)]
    map(δFs) do δF  # Apply to both δF{total} and δF{spin} in the same way
        isnothing(δF) && return nothing
        δρ    = @. T(mixing.α) * δF.fourier * (kTF^2 - C0 * Gsq) / (εr * kTF^2 - C0 * Gsq)
        δρ[1] = δF.fourier[1]  # Copy DC component, otherwise it never gets updated
        from_fourier(basis, δρ)
    end
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
Additionally there is the real-space localisation function `L(r)`.
For details see Herbst, Levitt 2020 arXiv:2009.01665

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `α`: Damping parameter
- `RPA`: Is the random-phase approximation used for the kernel (i.e. only Hartree kernel is
  used and not XC kernel)
- `verbose`: Run the GMRES in verbose mode.
"""
function HybridMixing(;εr=1.0, kTF=0.8, localisation=identity, kwargs...)
    χ0terms = [DielectricModel(εr=εr, kTF=kTF, localisation=localisation), LdosModel()]
    χ0Mixing(; χ0terms=χ0terms, kwargs...)
end


@doc raw"""
Generic mixing function using a model for the susceptibility composed of the sum of the `χ0terms`.
For valid `χ0terms` See the subtypes of `χ0Model`. The dielectric model is solved in
real space using a GMRES. Either the full kernel (`RPA=false`) or only the Hartree kernel
(`RPA=true`) are employed. `verbose=true` lets the GMRES run in verbose mode
(useful for debugging).
"""
@kwdef struct χ0Mixing
    α::Real   = 0.8
    RPA::Bool = true       # Use RPA, i.e. only apply the Hartree and not the XC Kernel
    χ0terms   = χ0Model[Applyχ0Model()]  # The terms to use as the model for χ0
    verbose::Bool = false  # Run the GMRES verbosely
end

@timing "χ0Mixing" function mix(mixing::χ0Mixing, basis, δF_tot::RealFourierArray,
                                δF_spin=nothing; ρin, ρ_spin_in, kwargs...)
    T = eltype(δF_tot)
    n_spin = basis.model.n_spin_components
    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)

    # Initialise χ0terms and remove nothings (terms that don't yield a contribution)
    χ0applies = [χ0(basis; ρin=ρin, ρ_spin_in=ρ_spin_in, kwargs...) for χ0 in mixing.χ0terms]
    χ0applies = [apply for apply in χ0applies if !isnothing(apply)]

    # If no applies left, do not bother running GMRES and directly do simple mixing
    isempty(χ0applies) && return mix(SimpleMixing(α=mixing.α), basis, δF_tot, δF_spin)

    # Solve J δρ = δF with J = (1 - χ0 vc) and χ_0 given as the sum of the χ0terms
    devec(x) = reshape(x, size(δF_tot)..., n_spin)
    function Jop(x)
        δF = devec(x)  # [:, :, :, 1] is spin-up (or total), [:, :, :, 2] is spin-down

        # Apply Kernel (just vc for RPA and (vc + K_{xc}) if not RPA)
        @views if n_spin == 1
            x_tot  = from_real(basis, δF[:, :, :, 1])
            x_spin = nothing
        else
            x_tot  = from_real(basis, δF[:, :, :, 1] .+ δF[:, :, :, 2])
            x_spin = from_real(basis, δF[:, :, :, 1] .- δF[:, :, :, 2])
        end
        δV = apply_kernel(basis, x_tot, x_spin; ρ=ρin, ρspin=ρ_spin_in, RPA=mixing.RPA)

        # set DC of δV to zero (δV[1] is spin-up or total, δV[2] is spin-down)
        δV_DC = mean(mean(δV[σ].real) for σ in 1:n_spin)
        δV[1].real .-= δV_DC
        n_spin == 2 && (δV[2].real .-= δV_DC)

        JδF = copy(δF)
        for apply_term! in χ0applies
            apply_term!(JδF, δV, -1)  # JδF .-= χ0 * δV
        end
        vec(JδF .-= mean(JδF))  # Zero DC component in total density response
    end

    if n_spin == 1
        δF_updown = δF_tot.real
    else
        δF_updown = cat((δF_tot.real .+ δF_spin.real) ./ 2,  # spin-up
                        (δF_tot.real .- δF_spin.real) ./ 2,  # spin-down
                        dims=4)
    end
    δF_updown .-= mean(δF_updown)  # Zero DC of δF_updown
    J = LinearMap(Jop, length(δF_updown))
    x = gmres(J, vec(δF_updown), verbose=mixing.verbose)
    # TODO Further improvement: Adapt tolerance of gmres to norm(ρ_out - ρ_in)

    δρ = T(mixing.α) .* devec(x)  # Apply damping
    δρ .+= (sum(δF_tot.real) - sum(δρ)) / length(δF_tot)  # Set DC from δF

    @views if n_spin == 1
        from_real(basis, δρ[:, :, :, 1]), nothing
    else
        (from_real(basis, δρ[:, :, :, 1] .+ δρ[:, :, :, 2]),
         from_real(basis, δρ[:, :, :, 1] .- δρ[:, :, :, 2]))
    end
end
