using KrylovKit
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
function mix_potential(args...; kwargs...)
    mix_density(args...; kwargs...)
end


# Mixing in the generalised density (essentially an adapted tuple of ρ and τ;
# see pack_gdensity in densities.jl): For now just fall back to ρ-only mixing
function mix_gdensity(mixing, basis, ΔD; kwargs...)
    Δρ, Δτ  = split_gdensity_flat_(basis, ΔD)
    Pinv_Δρ = mix_density(mixing, basis, Δρ; kwargs...)
    pack_gdensity_flat_(basis, Pinv_Δρ, Δτ)
end


@doc raw"""
Simple mixing: ``J^{-1} ≈ 1``
"""
struct SimpleMixing <: Mixing; end
mix_density(::SimpleMixing, ::PlaneWaveBasis, δF; kwargs...) = δF


@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{|G|^2}{k_{TF}^2 + |G|^2}``
where ``k_{TF}`` is the Thomas-Fermi wave vector. For spin-polarized calculations
by default the spin density is not preconditioned unless a non-default value
for `ΔDOS_Ω` is specified. This value should roughly be the expected difference in density
of states (per unit volume) between spin-up and spin-down. Notably setting
`ΔDOS_Ω = kTF^2 / 4π` disables acting on the ``β`` spin channel completely (as if the
DOS on ``β`` spin was zero).

Notes:
- Abinit calls ``1/k_{TF}`` the dielectric screening length (parameter *dielng*)
"""
@kwdef struct KerkerMixing <: Mixing
    # Default kTF parameter suggested by Kresse, Furthmüller 1996 (kTF=1.5Å⁻¹)
    # DOI 10.1103/PhysRevB.54.11169
    kTF::Real    = 0.8  # == sqrt(4π (DOS_α + DOS_β) / Ω)
    ΔDOS_Ω::Real = 0.0  # == (DOS_α - DOS_β) / Ω; set == kTF^2/4π to disable acting on β density
end

@timing "KerkerMixing" function mix_density(mixing::KerkerMixing, basis::PlaneWaveBasis,
                                            δF; kwargs...)
    T      = eltype(δF)
    G²     = norm2.(G_vectors_cart(basis))
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

    δF_fourier    = fft(basis, δF)
    δFtot_fourier = total_density(δF_fourier)
    δρtot_fourier = δFtot_fourier .* G² ./ (kTF.^2 .+ G²)
    enforce_real!(δρtot_fourier, basis)
    δρtot = irfft(basis, δρtot_fourier)

    # Copy DC component, otherwise it never gets updated
    δρtot .+= mean(total_density(δF)) .- mean(δρtot)

    if basis.model.n_spin_components == 1
        ρ_from_total_and_spin(δρtot, nothing)
    elseif abs(ΔDOS_Ω) < eps(real(T))
        ρ_from_total_and_spin(δρtot, spin_density(δF))
    else
        δFspin_fourier = spin_density(δF_fourier)
        δρspin_fourier = @. δFspin_fourier - δFtot_fourier * (4π * ΔDOS_Ω) / (kTF^2 + G²)
        enforce_real!(δρspin_fourier, basis)
        δρspin = irfft(basis, δρspin_fourier)
        ρ_from_total_and_spin(δρtot, δρspin)
    end
end


@doc raw"""
The same as [`KerkerMixing`](@ref), but the Thomas-Fermi wavevector is computed
from the current density of states at the Fermi level. To determine the DOS
by default a temperature of `min(50basis.model.temperature, 0.1)` and `Smearing.Gaussian`
smearing is employed (irrespective of the SCF smearing), but this may be changed using the
`smearing` and `temperature` arguments. Note, that using a non-monotonous smearing at
temperatures much above the SCF temperature can lead to artefacts (e.g. negative LDOS)
and is thus not recommended.
"""
@kwdef struct KerkerDosMixing <: Mixing
    smearing::Union{Nothing,Smearing.SmearingFunction} = nothing
    temperature::Union{Nothing,Float64} = nothing
end
Base.show(io::IO, ::KerkerDosMixing) = print(io, "KerkerDosMixing()")
@timing "KerkerDosMixing" function mix_density(mixing::KerkerDosMixing, basis::PlaneWaveBasis,
                                               δF; εF, eigenvalues, kwargs...)
    defaults = default_smearing_temperature(basis.model)
    temperature = @something(mixing.temperature, defaults.temperature)
    smearing    = @something(mixing.smearing,    defaults.smearing)
    @debug "Mixing smearing and temperature: $smearing $temperature"

    if iszero(temperature)
        return mix_density(SimpleMixing(), basis, δF)
    else
        n_spin = basis.model.n_spin_components
        Ω = basis.model.unit_cell_volume
        dos_per_vol  = compute_dos(εF, basis, eigenvalues; temperature, smearing) ./ Ω
        kTF  = sqrt(4π * sum(dos_per_vol))
        ΔDOS_Ω = n_spin == 2 ? dos_per_vol[1] - dos_per_vol[2] : zero(kTF)
        mix_density(KerkerMixing(; kTF, ΔDOS_Ω), basis, δF)
    end
end

@doc raw"""
We use a simplification of the [Resta model](https://doi.org/10.1103/physrevb.16.2717) and set
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
    εr > 1 / sqrt(eps(T)) && return mix_density(KerkerMixing(; kTF), basis, δF)

    C0 = 1 - εr
    Gsq = map(G -> norm2(G), G_vectors_cart(basis))
    δF_fourier = fft(basis, δF)
    δρ = @. δF_fourier * (kTF^2 - C0 * Gsq) / (εr * kTF^2 - C0 * Gsq)
    δρ = irfft(basis, δρ)
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
For details see  [Herbst, Levitt 2020](https://arxiv.org/abs/2009.01665).

By default the LdosModel is constructed using a temperature of
`min(50basis.model.temperature, 0.1)` and `Smearing.Gaussian` smearing (irrespective of the
`model.smearing`), but this may be changed using the `smearing` and `temperature` arguments.
Note, that using a non-monotonous smearing at temperatures much above the SCF temperature
can lead to artefacts (e.g. negative LDOS) and is thus not recommended.

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `RPA`: Is the random-phase approximation used for the kernel (i.e. only Hartree kernel is
  used and not XC kernel)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES
"""
function HybridMixing(; εr=10.0, kTF=0.8, localization=identity,
                        smearing=nothing, temperature=nothing, kwargs...)
    # TODO: switch to non-adaptive version above
    χ0terms = [DielectricModel(; εr, kTF, localization),
               LdosModel(; smearing, temperature)]
    χ0Mixing(; χ0terms, kwargs...)
end


@doc raw"""
The model for the susceptibility is
```math
\begin{aligned}
    χ_0(r, r') &= (-D_\text{loc}(r) δ(r, r') + D_\text{loc}(r) D_\text{loc}(r') / D)
\end{aligned}
```
where ``D_\text{loc}`` is the local density of states,
``D`` is the density of states.
For details see [Herbst, Levitt 2020](https://arxiv.org/abs/2009.01665).

By default the LdosModel is constructed using a temperature of
`min(50basis.model.temperature, 0.1)` and `Smearing.Gaussian` smearing (irrespective of the
`model.smearing`), but this may be changed using the `smearing` and `temperature` arguments.
Note, that using a non-monotonous smearing at temperatures much above the SCF temperature
can lead to artefacts (e.g. negative LDOS) and is thus not recommended.

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `RPA`: Is the random-phase approximation used for the kernel (i.e. only Hartree kernel is
  used and not XC kernel)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES
"""
function LdosMixing(; smearing=nothing, temperature=nothing, kwargs...)
    # TODO: switch to non-adaptive version above
    χ0Mixing(; χ0terms=[LdosModel(; smearing, temperature)], kwargs...)
end

@doc raw"""
Hybrid mixing for ferromagnetic systems, that uses the LDOS χ0-model for the Hartree 
kernel and a diagonal χ0-model for the exchange-correlation kernel: 
```math
\begin{aligned}
    & χ_0^\text{diag} = \sum_{i=1}^\infty f'(\varepsilon_i-\varepsilon_F) |\psi_i|^2(r)+|\psi_i|^2(r') + \frac1D D_\text{loc}(r) D_\text{loc}(r') \\
    & \varepsilon^\dagger = I - \chi_0^text{LDOS} K_H - \chi_0^\text{diag} K_\text{XC}
\end{aligned}
```
"""
function HybridDiagonalMixing(; verbose=false, maxiter=20, reltol=1e-6, kwargs...)
    mixedχ0Mixing(; χ0terms_Kxc=[ΔεModel(; kwargs...), ΔεFModel()], 
                    χ0terms_vc=[DFTK.LdosModel()], 
                    verbose, maxiter, reltol)
end


@doc raw"""
Generic mixing function using a model for the susceptibility composed of the sum of the `χ0terms`.
For valid `χ0terms` See the subtypes of `χ0Model`. The dielectric model is solved in
real space using a GMRES. Either the full kernel (`RPA=false`) or only the Hartree kernel
(`RPA=true`) are employed. `verbose=true` lets the GMRES run in verbose mode
(useful for debugging).
"""
@kwdef struct χ0Mixing <: Mixing
    χ0terms   = χ0Model[Applyχ0Model()]  # The terms to use as the model for χ0
    RPA::Bool = true        # Use RPA, i.e. only apply the Hartree and not the XC Kernel
    verbose::Bool = false   # Run the GMRES verbosely
    reltol::Float64 = 0.01  # Relative tolerance for GMRES
    maxiter::Int = 100      # Maximum number of GMRES iterations
end
function Base.show(io::IO, mixing::χ0Mixing)
    χ0terms = mixing.χ0terms
    if length(χ0terms) == 1 && χ0terms[1] isa Applyχ0Model
        print(io, "χ0Mixing([Applyχ0Model()], ")
    elseif length(χ0terms) == 1 && χ0terms[1] isa LdosModel
        print(io, "LdosMixing(")
    elseif length(χ0terms) == 2 && χ0terms[2] isa LdosModel && χ0terms[1] isa DielectricModel
        print(io, "HybridMixing(")
    else
        print(io, "χ0Mixing([$(length(mixing.χ0terms)) terms], ")
    end
    print(io, "RPA=$(mixing.RPA), reltol=$(mixing.reltol))")
end

"""
Get the model adjoint dielectric operator used for this mixing.
"""
function get_ε_adj_op(mixing::χ0Mixing, basis::PlaneWaveBasis; ρin, kwargs...)
    
    χ0applies = filter(!isnothing, [χ₀(basis; ρin, kwargs...) for χ₀ in mixing.χ0terms])
    # If no applies left, do not bother running GMRES and directly do simple mixing
    isempty(χ0applies) && return identity

    # Solve (ε^†) δρ = δF with ε^† = (1 - χ₀ vc) and χ₀ given as the sum of the χ0terms
    function dielectric_adjoint(δF)
        # Apply Kernel (just vc for RPA and (vc + K_{xc}) if not RPA)
        δV = apply_kernel(basis, δF; ρ=ρin, mixing.RPA)
        δV .-= mean(δV)
        εδF = copy(δF)
        for apply_term! in χ0applies
            apply_term!(εδF, δV, -1)  # εδF .-= χ₀ * δV
        end
        εδF .-= mean(εδF)
        εδF
    end
end

"""
Generic mixing function using several χ0-models, adapted to each kernel.
    ε† = I - [ χ0_Kh vc + χ0_Kxc Kxc + χ0_Khxc (vc+Kxc) ]
"""
@kwdef struct mixedχ0Mixing <: DFTK.Mixing
    χ0terms_vc::Array{DFTK.χ0Model} = DFTK.χ0Model[]    # Terms for the χ0_Kh model
    χ0terms_Kxc::Array{DFTK.χ0Model} = DFTK.χ0Model[]   # Terms for the χ0_Kxc model
    χ0terms_Khxc::Array{DFTK.χ0Model} = DFTK.χ0Model[]  # Terms for the χ0_Khxc model
    verbose::Bool = false   # Run the GMRES verbosely
    reltol::Float64 = 1e-10 # Relative tolerance for the GMRES.
    maxiter::Int = 20       # Maximum number of iterations for the GMRES
end
function Base.show(io::IO, mixing::mixedχ0Mixing)
    χ0terms_vc   = mixing.χ0terms_vc
    χ0terms_Kxc  = mixing.χ0terms_Kxc
    χ0terms_Khxc = mixing.χ0terms_Khxc

    if isempty(χ0terms_Khxc) && length(χ0terms_vc) == 1 && χ0terms_vc[1] isa LdosModel &&
       length(χ0terms_Kxc) == 2 && χ0terms_Kxc[1] isa ΔεModel && χ0terms_Kxc[2] isa ΔεFModel
        print(io, "HybridDiagonalMixing(temperature=$(χ0terms_Kxc[1].temperature), ")
    else
        print(io, "mixedχ0Mixing([$(length(χ0terms_vc)) Kh terms], ",
                  "[$(length(χ0terms_Kxc)) Kxc terms], ",
                  "[$(length(χ0terms_Khxc)) Khxc terms], ")
    end
    print(io, "reltol=$(mixing.reltol), maxiter=$(mixing.maxiter))")
end

"""
Get the model adjoint dielectric operator used for this mixing.
"""
function get_ε_adj_op(mixing::mixedχ0Mixing, basis::PlaneWaveBasis; ρin, kwargs...)
    
    χ0applies_vc = filter(!isnothing, [χ₀(basis; ρin, kwargs...) for χ₀ in mixing.χ0terms_vc])
    χ0applies_Kxc = filter(!isnothing, [χ₀(basis; ρin, kwargs...) for χ₀ in mixing.χ0terms_Kxc])
    χ0applies_Khxc = filter(!isnothing, [χ₀(basis; ρin, kwargs...) for χ₀ in mixing.χ0terms_Khxc])
    need_vc = !isempty(χ0applies_vc) || !isempty(χ0applies_Khxc)
    need_Kxc = !isempty(χ0applies_Kxc) || !isempty(χ0applies_Khxc)

    !need_vc && !need_Kxc && return identity

    function ε_adj(δρ)
        # Apply kernel
        vc_δρ = nothing
        Kxc_δρ = nothing
        if need_vc
            vc_δρ = apply_kernel(basis, δρ; ρ=ρin, RPA=true)
        end
        if need_Kxc
            Kxc_δρ = zero(δρ)
            for term in basis.terms
                if !(term isa DFTK.TermHartree)
                    δV_term = apply_kernel(term, basis, δρ;  ρ=ρin)
                    if !isnothing(δV_term)
                        Kxc_δρ .+= δV_term
                    end
                end
            end
        end
        εδρ = copy(δρ)
        # Apply χ0 :
        for apply_term! in χ0applies_vc
            apply_term!(εδρ, vc_δρ, -1)     # εδρ .-= χ₀ * vc_δρ
        end
        for apply_term! in χ0applies_Kxc
            apply_term!(εδρ, Kxc_δρ, -1)    # εδρ .-= χ₀ * Kxc_δρ
        end
        for apply_term! in χ0applies_Khxc
            apply_term!(εδρ, vc_δρ .+ Kxc_δρ, -1)  # εδρ .-= χ₀ * (vc_δρ + Kxc_δρ)
        end
        return εδρ
    end
end

@views @timing "χ0Mixing" function DFTK.mix_density(mixing::Union{χ0Mixing, mixedχ0Mixing}, 
        basis, Δρ::AbstractArray{T};
        ρin, kwargs...) where {T}

    ε_adj_op = get_ε_adj_op(mixing, basis; ρin, kwargs...)
    ε_adj_op == identity && return mix_density(SimpleMixing(), basis, Δρ)
    
    mixed_Δρ = similar(Δρ)
    mixed_Δρ, info = linsolve(ε_adj_op, Δρ;
        verbosity=(mixing.verbose ? 3 : 0),
        rtol=T(mixing.reltol),
        krylovdim=mixing.maxiter,
        maxiter=1,
        ishermitian=false,
        isposdef=false,
    )
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        info.converged == 0 && @warn "χ0-mixing GMRES not converged"
    end

    MPI.Bcast!(mixed_Δρ, 0, MPI.COMM_WORLD) 

    # Ensuring that the mean value of Δρ is unchanged 
    # (conservation of electron number).
    return mixed_Δρ .+ DFTK.mean(Δρ) .- DFTK.mean(mixed_Δρ)
end


@timing "χ0Mixing" function mix_potential(mixing::Mixing, basis::χ0Mixing, δF::AbstractArray; kwargs...)
    error("Not yet implemented.")
end

function default_smearing_temperature(model::Model)
    # Set temperature to be 100 times the model temperature, but make sure
    # to never overshoot 0.1 and never under-shoot the model.temperature
    temperature = max(model.temperature, min(0.1, 100model.temperature))
    (; smearing=Smearing.Gaussian(), temperature)
end
