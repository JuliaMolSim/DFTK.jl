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
    Δρ, Δτ  = split_gdensity(basis, ΔD)
    Pinv_Δρ = mix_density(mixing, basis, Δρ; kwargs...)
    pack_gdensity(basis, Pinv_Δρ, Δτ)
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
        δρspin = irfft(basis, δρspin_fourier)
        ρ_from_total_and_spin(δρtot, δρspin)
    end
end


@doc raw"""
The same as [`KerkerMixing`](@ref), but the Thomas-Fermi wavevector is computed
from the current density of states at the Fermi level. To determine the DOS
by default a temperature of `min(100basis.model.temperature, 0.1)` and `Smearing.Gaussian`
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
`min(100basis.model.temperature, 0.1)` and `Smearing.Gaussian` smearing (irrespective of the
`model.smearing`), but this may be changed using the `smearing` and `temperature` arguments.
Note, that using a non-monotonous smearing at temperatures much above the SCF temperature
can lead to artefacts (e.g. negative LDOS) and is thus not recommended.

The `RPA` keyword argument controls whether or not the random-phase approximation used for 
the kernel (i.e. only Hartree kernel is used and not XC kernel).

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES.
- `maxiter`: Maximum number of iterations for GMRES.
"""
function LdosDielectricMixing(; εr=10.0, kTF=0.8, localization=identity,
                                smearing=nothing, temperature=nothing, RPA=true, kwargs...)
    # TODO: switch to non-adaptive version above
    kernel_types = RPA ? [TermHartree, ] : [TermHartree, TermXc]
    χ0s = [DielectricModel(; εr, kTF, localization),
           LdosModel(; smearing, temperature)]
    model_name = "LdosDielectricMixing(εr=$εr, kTF=$(kTF), RPA=$(RPA))"
    χ0Mixing(; terms=[(kernel_types, χ0s)], model_name,  kwargs...)
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
`min(100basis.model.temperature, 0.1)` and `Smearing.Gaussian` smearing (irrespective of the
`model.smearing`), but this may be changed using the `smearing` and `temperature` arguments.
Note, that using a non-monotonous smearing at temperatures much above the SCF temperature
can lead to artefacts (e.g. negative LDOS) and is thus not recommended.

The `RPA` keyword argument controls whether or not the random-phase approximation used for 
the kernel (i.e. only Hartree kernel is used and not XC kernel).

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES.
- `maxiter`: Maximum number of iterations for GMRES.
"""
function LdosMixing(; smearing=nothing, temperature=nothing, RPA=true, kwargs...)
    # TODO: switch to non-adaptive version above
    kernel_types = RPA ? [TermHartree, ] : [TermHartree, TermXc]
    χ0s = [LdosModel(; smearing, temperature), ]
    model_name="LdosMixing(RPA=$(RPA))"
    χ0Mixing(; terms=[(kernel_types, χ0s)], model_name, kwargs...)
end

@doc raw"""
Hybrid mixing for ferromagnetic systems, that uses the LDOS χ0-model [`LdosModel`](@ref) 
for the Hartree  kernel and a diagonal χ0-model for the exchange-correlation kernel: 
```math
\begin{aligned}
    & χ_0^\text{diag} = \sum_{i=1}^\infty f'(\varepsilon_i-\varepsilon_F)
      |\psi_i|^2(r)+|\psi_i|^2(r') + \frac1D D_\text{loc}(r) D_\text{loc}(r') \\
    & \varepsilon^\dagger = I - \chi_0^text{LDOS} K_H - \chi_0^\text{diag} K_\text{XC}
\end{aligned}
```
For details see [Barat, Levitt, Torrent 2026](https://hal.science/hal-05658631).
ABINIT calls this the Hybrid preconditioner.

The smearing temperature and smearing functions used in the LDOS and diagonal χ0-models can
be set with the `smearing` and `temperature` keyword arguments. The default is 
`Smearing.Gaussian` smearing with a temperature of `min(100basis.model.temperature, 0.1)`.

Important `kwargs` passed on to [`χ0Mixing`](@ref)
- `verbose`: Run the GMRES in verbose mode.
- `reltol`: Relative tolerance for GMRES.
- `maxiter`: Maximum number of iterations for GMRES.
"""
function LdosXcDiagonalMixing(; reltol=1e-6, smearing=nothing, temperature=nothing, kwargs...)
    terms = [
        ([TermHartree], [LdosModel(;     smearing, temperature)]),
        ([TermXc],      [DiagonalModel(; smearing, temperature)]),
    ]
    # TODO: Maybe reltol can be raised to similar levels (1e-2) as in LdosMixing ?
    χ0Mixing(; terms, model_name="LdosXcDiagonalMixing()", reltol, kwargs...)
end

@doc raw"""
Generic mixing function using model susceptibilities. `terms` links which kernels are
combined with which approximations of the independent particle susceptibility to make
up the approximation of the dielectric model, see the implementations of [`LdosMixing`](@ref)
and [`LdosXcDiagonalMixing`](@ref) for examples. The datastructure used to represent
the `terms` may change in future versions of DFTK. Don't rely on it in your work
and instead use [`LdosMixing`](@ref), [`LdosDielectricMixing`](@ref) and
[`LdosXcDiagonalMixing`](@ref) to construct mixing methods.

The dielectric model is solved in real space using a GMRES, whose
convergence is controlled by `reltol` and `maxiter`.
`verbose=true` lets the GMRES run in verbose mode (useful for debugging).
"""
@kwdef struct χ0Mixing <: Mixing
    terms::Vector = [([TermHartree, TermXc], [Applyχ0Model()])]
    model_name = "χ0Mixing(custom)"  # Human-readable model name (used for printing)
    verbose::Bool = false            # Run the GMRES verbosely
    reltol::Float64 = 1e-2           # Relative tolerance for the GMRES.
    maxiter::Int = 20                # Maximum number of iterations for the GMRES
end
function Base.show(io::IO, mixing::χ0Mixing)
    extra_string = "reltol=$(mixing.reltol)"
    if endswith(mixing.model_name, "()")
        print(io, mixing.model_name[1:end-2], "($extra_string)")
    else
        print(io, replace(mixing.model_name, ")" => ", $extra_string)"))
    end
end


@views @timing "χ0Mixing" function mix_density(mixing::χ0Mixing, basis, Δρ::AbstractArray{T};
                                               kwargs...) where {T}
    mix_dielectric_model(mixing, basis, Δρ; act_on_density=true, kwargs...)
end

@timing "χ0Mixing" function mix_potential(mixing::χ0Mixing, basis, ΔV::AbstractArray{T};
                                          kwargs...) where {T}
    mix_dielectric_model(mixing, basis, ΔV; act_on_density=false, kwargs...)
end

function build_dielectric_terms_(mixing::χ0Mixing, basis::PlaneWaveBasis; ρin, kwargs...)
    dielectric_terms = map(mixing.terms) do (kernel_types, χ0terms)
        # Filter out χ0terms that give nothing apply functions and kernel terms
        # that are not present in the basis; also prepare the χ0applies and select
        # the actual instantiated terms instead of the term types.
        kernel_terms  = [term for term in basis.terms if any(term isa t for t in kernel_types)]
        χ0applies     = filter(!isnothing, [χ₀(basis; ρin, kwargs...) for χ₀ in χ0terms])
        (kernel_terms, χ0applies)
    end
    filter(t -> !isempty(t[1]) && !isempty(t[2]), dielectric_terms)
end

"""
Apply χ0Mixing to a density difference (act_on_density=true) or to a potential difference
(act_on_density=false).
"""
function mix_dielectric_model(mixing::χ0Mixing, basis::PlaneWaveBasis, Δx::AbstractArray{T};
                              act_on_density::Bool, ρin, kwargs...) where {T}
    dielectric_terms = build_dielectric_terms_(mixing, basis; ρin, kwargs...)
    if isempty(dielectric_terms)
        return Δx  # No preconditioning
    end

    # Either ε = 1 - Kχ0 (adjoint=false) or ε^† = 1 - χ0K (adjoint=true)
    dielectric = (act_on_density ? get_ε_adj_op(dielectric_terms, basis; ρ=ρin)
                                 : get_ε_op(dielectric_terms, basis;     ρ=ρin))

    mixed_Δx, info = linsolve(dielectric, Δx;
        verbosity=(mixing.verbose ? 3 : 0),
        rtol=T(mixing.reltol),
        krylovdim=mixing.maxiter,
        maxiter=1,
        ishermitian=false,
        isposdef=false,
    )
    if mpi_master(MPI.COMM_WORLD)
        cond_not = info.converged == 0 ? "NOT " : ""
        cond_eps = act_on_density ? "ε_adj" : "ε"
        @debug "χ0Mixing GMRES$(cond_not) on $(cond_eps) converged in $(info.numops) ε-vec-products."
        info.converged == 0 && @warn "χ0-mixing GMRES not converged."
    end
    MPI.Bcast!(mixed_Δx, 0, MPI.COMM_WORLD)

    if act_on_density
        # Ensure that the mean value of Δρ is unchanged (conservation of electron count)
        return mixed_Δx .+ mean(Δx) .- mean(mixed_Δx)
    else
        return mixed_Δx
    end
end


function get_ε_adj_op(dielectric_terms::AbstractVector, basis; ρ)
    # TODO : Combine this with the DielectricAdjoint struct
    function ε_adj(δρ)
        εδρ = copy(δρ)
        for (kernel_terms, χ0applies) in dielectric_terms
            Kδρ = zero(δρ)
            for term in kernel_terms
                Kδρ .+= apply_kernel(term, basis, δρ; ρ)
            end

            # Apply χ0 model
            for apply_χ0! in χ0applies
                apply_χ0!(εδρ, Kδρ, -1)  # εδρ .-= χ₀ * Kδρ
            end
        end

        εδρ
    end
end
function get_ε_op(dielectric_terms::AbstractVector, basis; ρ)
    function ε(δV)
        εδV = copy(δV)
        for (kernel_terms, χ0applies) in dielectric_terms
            # Apply χ0 model
            χ0δV = zero(δV)
            for apply_χ0! in χ0applies
                apply_χ0!(χ0δV, δV, 1)  # χ0δV .+= χ₀ * δV
            end

            for term in kernel_terms
                εδV .-= apply_kernel(term, basis, χ0δV; ρ)
            end
        end
        εδV
    end
end

function default_smearing_temperature(model::Model)
    # Set temperature to be 100 times the model temperature, but make sure
    # to never overshoot 0.1 and never under-shoot the model.temperature
    temperature = max(model.temperature, min(0.1, 100model.temperature))
    (; smearing=Smearing.Gaussian(), temperature)
end
