# This file provides fixed-pointmping
#
# The callables subtyping `ScfSolver` must accept being called like
# `fp_solver(f, x0, info0; maxiter, damping)`, where `f` is the fixed-point map.
#
# The fixed-point map `f` is expected to be called as such:
#    `(fx, info) = f(x, info)`
# The`info` contains auxiliary state, including two boolean flags:
#    `info.converged`: flagged inside `f` if the convergence criterion is achieved.
#    `info.timedout`: flagged inside `f` if a timeout is achieved.
# The fixed-point function `f` is just passive in the sense that it flags these,
# but still can be called further. The decision-making is left to the solver,
# with the default convention being that either of these flags leads to termination.
#
# The solver must return an object supporting res.fixpoint and res.info
 
## `x` is always the *packed generalised density* (see `pack_gdensity`/`split_gdensity`
## in densities.jl), i.e. a plain array combining `ρ` and (if present) `τ`. Solvers that
## need more than the density (e.g. PCDIIS, which needs reference orbitals) read that
## extra state off `info` (in particular `info.ψ` and `info.basis`), which is passed
## through on every fixed-point call regardless of which solver is used. This keeps a
## single state representation for all solvers instead of a solver-specific `StateType`.
 
abstract type ScfSolver end

"""
Create a simple solver based on damped fixed-point iterations. It updates the
generalised density `x` (i.e. the concatenation of `ρ` and `τ`) as
`x = damping * x_new + (1 - damping) * x`. For choosing the damping value
or applying some kind of mixing, see the other keyword arguments of
[`self_consistent_field`](@ref).
"""
struct ScfDampingSolver <: ScfSolver end
function (scf::ScfDampingSolver)(f, x0, info0; maxiter, damping)
    β = convert(eltype(x0), damping)
    x = x0
    info = info0
    for _ = 1:maxiter
        fx, info = f(x, info)
        if info.converged || info.timedout
            break
        end
        x = @. β * fx + (1 - β) * x
    end
    (; fixpoint=x, info)
end

@doc raw"""
Create an anderson-accelerated SCF solver for the [`self_consistent_field`](@ref) solver.

This solver only performs damping and anderson acceleration in the density, but
not in the kinetic energy density. For functionals not involving the kinetic energy
density `τ` this algorithm is equivalent to [`ScfAndersonSolver`](@ref).
For meta-GGA functionals this algorithms can be faster than [`ScfAndersonSolver`](@ref),
but generally [`ScfAndersonDensitySolver`](@ref) is more reliable. 

## Keyword arguments
- `m::Integer`       (default: `10`) Maximal Anderson history size
- `m_start::Integer` (default: `1`)  Start collecting history in the `m_start`th SCF iteration
- `maxcond::Real` (default: `1e6`)
  Maximal condition number in Anderson matrix; a larger value triggers truncation of the
  older entries in the Anderson history.
- `errorfactor::Real` (default: `1e5`): We follow [^CDLS21] (adaptive Anderson acceleration)
  and drop iterates, which do not satisfy
  ```math
      \|r(ρᵢ)\| < \text{errorfactor} minᵢ \|r(ρᵢ)\|
  ```
  where $r(ρ)$ denotes the preconditioned SCF residual corresponding to density $ρ$.
  This means the best way to save memory is to reduce `errorfactor` to `1e3` or `100`,
  which reduces the effective history size.

[^CDLS21]: Chupin, Dupuy, Legendre, Séré. Math. Model. Num. Anal. **55**, 2785 (2021) dDOI [10.1051/m2an/2021069](https://doi.org/10.1051/m2an/2021069)
"""
struct ScfAndersonDensitySolver{Targs} <: ScfSolver
    m_start::Int
    anderson_kwargs::Targs
end

function ScfAndersonDensitySolver(; m_start::Integer=1, kwargs...)
    ScfAndersonDensitySolver(m_start, kwargs)
end
# For the show function, see below
function (scf::ScfAndersonDensitySolver)(f, x0, info0; maxiter, damping)
    T = eltype(x0)
    α = convert(T, damping)
    x = x0
    ρ, _ = split_gdensity(info0.basis, x0)
    info = info0
    #acceleration = AndersonAcceleration(; scf.anderson_kwargs...)
    acceleration = Acceleration(AndersonType(); α, scf.anderson_kwargs...)
    for i = 1:maxiter
        fx, info = f(x, info)
        if info.converged || info.timedout
            break
        end

        fρ, fτ = split_gdensity(info.basis, fx)
        if i < scf.m_start
            @debug "Skipping Anderson acceleration in iteration $i"
            ρ = @. ρ + α * (fρ - ρ)
        else
            @debug "Using Anderson acceleration in iteration $i"
            # Damp ρ and send it to anderson; τ is just patched through without any changes
            #residual_ρ = fρ - ρ
            ρ = acceleration(ρ, fρ, nothing) #this is a bit ugly, since info is not necessary here, but is needed for Pcdiis
        end
        x = pack_gdensity(info.basis, ρ, fτ)
    end
    (; fixpoint=x, info)
end

@doc raw"""
Create an anderson-accelerated SCF solver for the [`self_consistent_field`](@ref) solver.

This solver performs damping and anderson acceleration on the tuple `(ρ, τUEG^{-1}(τ - τW))`,
where `τUEG` is the uniform electron gas kinetic energy densiy and `τW` is the von
Weizsäcker kinetic energy density.

An alternative is [`ScfAndersonDensitySolver`](@ref), which only performs damping and
anderson acceleration in the density, but not in the kinetic energy density. For functionals
not involving the kinetic energy density `τ` both algorithms are equivalent; for meta-GGA
functionals this solver tends to be more reliable, but [`ScfAndersonDensitySolver`](@ref)
is sometimes faster if both work.

See the documentation of [`ScfAndersonDensitySolver`](@ref) for documentation on
keyword arguments and their default values.
"""
struct ScfAndersonSolver{Repr, Targs} <: ScfSolver
    representation::Repr
    m_start::Int
    anderson_kwargs::Targs
end
function ScfAndersonSolver(; representation=TauVwScaled(), m_start::Integer=1, kwargs...)
    ScfAndersonSolver(representation, m_start, kwargs)
end
function (scf::ScfAndersonSolver)(f, x0, info0; maxiter, damping)
    T = eltype(x0)
    α = convert(T, damping)
    x = x0
    info = info0
    #acceleration = AndersonAcceleration(; scf.anderson_kwargs...)
    acceleration = Acceleration(AndersonType(); α, scf.anderson_kwargs...)
    for i = 1:maxiter
        fx, info = f(x, info)
        if info.converged || info.timedout
            break
        end

        x  = to_representation!(scf.representation, info.basis,  x)
        fx = to_representation!(scf.representation, info.basis, fx)
        if i < scf.m_start
            @debug "Skipping Anderson acceleration in iteration $i"
            x = @. x + α * (fx - x)
        else
            @debug "Using Anderson acceleration in iteration $i"
            # Damp ρ and send it to anderson; τ is just patched through without any changes
            #residual = fx - x
            x = acceleration(x, fx, nothing) #this is a bit ugly, since info is not necessary here, but is needed for Pcdiis
        end
        x = from_representation!(scf.representation, info.basis, x)
    end
    (; fixpoint=x, info)
end

@doc raw"""
Create a Pcdiis-accelerated solver for the [`self_consistent_field`](@ref) solver.

This solver performs acceleration on a packed SCF state of type GdensityOrbitals that
also contains the ψ and occupation in addition to the density. 

Internally, acceleration is performed on ψ, then ρ is recomputed with the updated ψ.

## Keyword arguments
- `depth::Integer`       (default: `10`) Maximal Pcdiis history size
- `m_start::Integer`     (default: `1`)  Start collecting history in the `m_start`th SCF iteration
- `maxcond::Real`        (default: `1e6`)
  Maximal condition number in Pcdiis matrix; a larger value triggers truncation of the
  older entries in the Anderson history.
- `errorfactor::Real`    (default: `1e5`): Drop iterates that to don satisfy
  ```math
      \|r(ψᵢ)\| < \text{errorfactor} minᵢ \|r(ψᵢ)\|
  ```
  where $r(ψ)$ denotes the commutator of the density matrix Ρ(ψ) and the Fock matrix F(ψ), 
  and the norm is the Frobenius norm.
- `reference`            (default: `nothing`) Reference for gauge fixing. Has to be provided 
manually. If not provided, fall back on start guess. If this is also not provided, stop.
"""
struct ScfPcdiisSolver{Targs} <: ScfSolver
    m_start::Int
    pcdiis_kwargs::Targs
end
function ScfPcdiisSolver(; m_start::Integer=2, kwargs...)
    m_start < 1 && throw(ArgumentError("m_start must be ≥ 1"))
    ScfPcdiisSolver(m_start, NamedTuple(kwargs)) 
end
function (scf::ScfPcdiisSolver)(f, x0, info0; maxiter, damping)
    basis = info0.basis
    x = pack_gdensity(basis, split_gdensity(basis, x0)..., info0.ψ, info0.occupation)
    info = info0

    m_start        = scf.m_start
    pcdiis_kwargs  = scf.pcdiis_kwargs
    has_reference  = haskey(pcdiis_kwargs, :reference)

    if has_reference
        # Case 1: Reference provided. 
    elseif !isnothing(info.ψ) && !isnothing(info.occupation)
        # Case 2: no explicit ψ_ref, but a start guess was passed in. Use it as
        # the reference — but only safe with an adaptive (growable) band algorithm.
        @warn "Using the provided initial guess wavefunction as the Pcdiis " *
              "reference. Ensure it has a sufficient number of bands!"
        pcdiis_kwargs = merge(pcdiis_kwargs, (; reference=info.ψ))
    else
        # Case 3: nothing provided at all.
        @error "Neither reference nor inital guess provided. Stop."
    end

    acceleration = Acceleration(PcdiisType(); pcdiis_kwargs...)

    for i = 1:maxiter
        (ρ, τ, _) = split_gdensity(basis, x)

        fx, info = f(ρ, info) #this is not nice. Maybe solve like above with to/from_representation.
        if info.converged || info.timedout
            break
        end
        fx = pack_gdensity(basis, split_gdensity(basis, fx)..., info.ψ, info.occupation)
        if i < m_start
            @debug "Skipping Pcdiis acceleration in iteration $i"
            x = fx
        else
            @debug "Using Pcdiis acceleration in iteration $i"
            x = acceleration(x, fx, info)
        end
    end

    (ρ, τ, _) = split_gdensity(basis, x)
    (; fixpoint=pack_gdensity(basis, ρ, τ), info)
end

struct TauVwScaled; end
function to_representation!(::TauVwScaled, basis, x)
    inv_τUEG(τ::AbstractArray) = (10/3 * (3π^2)^(-2/3) * max.(0, τ)) .^ (3/5)
    ρ, τ = split_gdensity(basis, x)
    if !isnothing(τ)
        τ .= inv_τUEG(τ .- von_weizsaecker_kinetic_energy_density(basis, ρ))
    end
    x
end
function from_representation!(::TauVwScaled, basis, x)
    τUEG(ρ::AbstractArray) =  3/10 * (3π^2)^(2/3)  * max.(0, ρ)  .^ (5/3)
    ρ, τ = split_gdensity(basis, x)
    if !isnothing(τ)
        τ .= τUEG(τ) .+ von_weizsaecker_kinetic_energy_density(basis, ρ)
    end
    x
end

function Base.show(io::IO, scf::Union{ScfAndersonSolver,ScfAndersonDensitySolver})
    if scf isa ScfAndersonSolver
        print(io, "ScfAndersonSolver")
    else
        print(io, "ScfAndersonDensitySolver")
    end
    print(io, "(; m_start=$(scf.m_start)")
    anderson = AndersonAcceleration(; scf.anderson_kwargs...)
    for arg in (:m, :maxcond, :errorfactor)
        print(io, ", $arg=$(getproperty(anderson, arg))")
    end
    print(io, ")")
end

#TODO Base.show(ScfPcdiisSolver)

@deprecate scf_damping_solver(; damping=1.0)           ScfDampingSolver()
@deprecate scf_anderson_solver(; m_start=1, kwargs...) ScfAndersonDensitySolver(; m_start, kwargs...)
