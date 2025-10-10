module DFTKManifoldsManoptRATExt
    using DFTK
    using Manopt
    using Manifolds
    using RecursiveArrayTools
    using LinearAlgebra

"""
    ManoptPreconditionersWrapper!!{T,S}

A wrapper for a DFTK preconditioner to be used within Manopt,
which implements the `Manopt.Preconditioner` interface.

This wrapper provides both an allocating `(M, p, X) -> Y` method to precondition
`X`, as well as the non-allocating `(M, Y, p, X) -> Y` that works in-place of `Y`.

# Fields

* `Nk::Int`: Number of k-points
* `Pks::Vector{T}`: (DFTK) Preconditioners, one for each k-point
* `kweights::Vector{S}`: weights for each k-point

# Constructor

mpw = ManoptPreconditionersWrapper!!(Nk, Pks, kweights)
"""
struct ManoptPreconditionersWrapper!!{T,S}
    Nk::Int
    Pks::Vector{T}
    kweights::Vector{S}
end
function (mp::ManoptPreconditionersWrapper!!)(
    M::ProductManifold,Y,p,X
)
    # Update preconditioner
    for ik = 1:mp.Nk
        DFTK.precondprep!(mp.Pks[ik], p[M, ik])
    end
    # Precondition the gradient in-place
    for ik = 1:mp.Nk
        DFTK.ldiv!(Y[M, ik], mp.Pks[ik], X[M, ik])
        DFTK.ldiv!(mp.kweights[ik], Y[M, ik]) # maybe remove local_Y
    end
    return Y
end
function (mp::ManoptPreconditionersWrapper!!)(M::ProductManifold, p, X)
    Y = copy(M, p, X)
    mp(M, Y, p, X)
    return Y
end
"""
    InsulatorEnergy{T,S,P,X,R,E,H}

A functor to represent both the energy cost function from direct minimization and its gradient.
To call the cost function, use `cgf(M,p)`, the gradient can be evaluated in-place of a tangent vector `X`
with `cgf(M, X, p)`.

For a derivation, see Sec. 7.2 and 7.5.2 of Cancès, Friesecke: [Density Functional Theory](https://doi.org/10.1007/978-3-031-22340-2), Springer.

This functor is designed to be used within `Manopt.jl` and caches the last computed cost,
gradient and the interim values to spare calls to [`compute_density`](@ref) and [`energy_hamiltonian`](@ref).

# Fields

* `basis::PlaneWaveBasis{T}`: The plane wave basis used for the DFT calculation.
* `occupation::Vector{S}`: The occupation numbers for each k-point.
* `Nk::Int`: The number of k-points.
* `filled_occ::Int`: The number of filled occupation states.
* `ψ::P`: The last iterate, stored as a DFTK wave representation.
* `X::X`: The last gradient, stored as a `Manopt.jl` tangent vector, i.e. using an `ArrayPartition`
* `ρ::R`: The last density, computed from the wave functions.
* `energies::E`: The last vector of energies, computed from the wave functions.
* `ham::H`: The last Hamiltonian, computed from the wave functions.
"""
mutable struct InsulatorEnergy{T,S,P,X,R,E,H}
    basis::PlaneWaveBasis{T}
    occupation::Vector{S}
    Nk::Int
    filled_occ::Int
    ψ::P        # last iterate, While we usually have an  iterate `ArrayPartition` `p` iterate in Manopt, we store it as a vector, i.e. in the ψ format for DFTK
    X::X        # last gradient
    ρ::R        # the last density
    energies::E # the last vector of energies
    ham::H      # the last Hamiltonian
    count::Int  # Count hamiltonian calls.
end
# Function shared by both cost and gradient of cost:
function _compute_density_energy_hamiltonian!(cgf::InsulatorEnergy, M::ProductManifold, p)
    # Can we improve this by copying elementwise?
    # copyto!(cgf.ψ, (copy(x) for x in p.x)) # deepcopyto!
    # Maybe like
    for ik in eachindex(cgf.ψ)
        copyto!(cgf.ψ[ik], p[M,ik])
    end
    copyto!(cgf.ρ, compute_density(cgf.basis, cgf.ψ, cgf.occupation))
    # Below not inplace, but probably not that important.
    cgf.energies, cgf.ham = energy_hamiltonian(cgf.basis, cgf.ψ, cgf.occupation; cgf.ρ)
    cgf.count += 1
    return cgf
end
# The cost function:
function (cgf::InsulatorEnergy)(M::ProductManifold,p)
    # Memoization check: Are we still at the same point?
    if all(cgf.ψ[i] == p[M, i] for i in eachindex(cgf.ψ))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    return cgf.energies.total
end
# The gradient of cost function:
function (cgf::InsulatorEnergy)(M::ProductManifold, X, p)
    # Memoization check: Is this X allready been computed?
    if all(cgf.X[M, i] == X[M, i] for i in eachindex(cgf.ψ))
        # Are we still at the same point?
        if all(cgf.ψ[i] == p[M, i] for i in eachindex(cgf.ψ))
            return X
        end
    end
    # Memoization check: Are we still at the same point?
    if all(cgf.ψ[i] == p[M, i] for i in eachindex(cgf.ψ))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    # Compute the Euclidean gradient in-place
    for ik = 1:cgf.Nk
        DFTK.mul!(X[M, ik], cgf.ham.blocks[ik], p[M, ik]) # mul! overload in DFTK
         # Using get_component(), as "X[M, ik] .*=" is not yet supported in ManifoldsBase.jl
        Manifolds.get_component(M, X, ik) .*= 2 * cgf.filled_occ * cgf.basis.kweights[ik]
    end
    riemannian_gradient!(M, X, p, X) # Convert to Riemannian gradient
    copyto!(cgf.X, X) # Memoization
    return X
end
# Access nested fields of the cost function
function Manopt.get_parameter(objective::Manopt.AbstractManifoldCostObjective, s::Symbol)
    return Manopt.get_parameter(Manopt.get_cost_function(objective), s)
end
function Manopt.get_parameter(energy_costgrad::InsulatorEnergy, s::Symbol)
    return Manopt.get_parameter(energy_costgrad, Val(s))
end
Manopt.get_parameter(energy_costgrad::InsulatorEnergy, ::Val{:ρ}) = energy_costgrad.ρ
function Manopt.get_parameter(energy_costgrad::InsulatorEnergy, ::Val{:Hamiltonian})
    return energy_costgrad.ham
end
function Manopt.get_parameter(energy_costgrad::InsulatorEnergy, ::Val{:Energies})
    return energy_costgrad.energies
end
function Manopt.get_parameter(
        energy_costgrad::InsulatorEnergy, ::Val{:HamiltonianEvaluations}
    )
    return energy_costgrad.count
end

#
#
# Debugs
# Δρ
mutable struct DebugDensityChange{F} <: DebugAction
    io::IO
    last_ρ::F
    prefix::String
end
function DFTK.DebugDensityChange(ρ::T; prefix = "Δρ:", io::IO=stdout) where {T}
    return DebugDensityChange{T}(io, ρ, prefix)
end
function (d::DebugDensityChange)(
    problem::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int
)
    current_ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ)
    ch = norm(current_ρ - d.last_ρ)
    d.last_ρ .= current_ρ
    (k >= 0) && print(d.io, "$(d.prefix)$(ch)")
end
# ‖ρ‖
mutable struct DebugDensityNorm <: DebugAction
    io::IO
    prefix::String
end
DFTK.DebugDensityNorm(; io::IO=stdout, prefix="‖ρ‖: ") = DebugDensityNorm(io, prefix)
function (::DebugDensityNorm)(
    problem::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int
)
    current_ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ)
    return (k >= 0) && print(d.io, "$(d.prefix)$(norm(current_ρ))")
end
Base.show(io::IO, ::DebugDensityNorm) = print(io, "DebugDensityNorm()")
#
#
# Records
# ρ
mutable struct RecordDensity{F} <: RecordAction
    recorded_values::Array{F,1}
end
DFTK.RecordDensity(ρ::F) where {F} = RecordDensity{F}(Array{F,1}())
function (r::RecordDensity)(
    problem::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int
)
    current_ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ)
    return Manopt.record_or_reset!(r, copy(current_ρ), k)
end
Base.show(io::IO, ::RecordDensity) = print(io, "RecordDensity()")
# Δρ
mutable struct RecordDensityChange{F,T} <: RecordAction
    recorded_values::Array{F,1}
    last_ρ::T
end
function DFTK.RecordDensityChange(ρ::T) where {T}
    return RecordDensityChange{typeof(norm(ρ)),typeof(ρ)}(Array{typeof(norm(ρ)),1}(), ρ)
end
function (r::RecordDensityChange)(
    problem::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    current_ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ)
    last_change = norm(current_ρ - r.last_ρ)
    r.last_ρ .= current_ρ
    return Manopt.record_or_reset!(r, last_change, k)
end
Base.show(io::IO, ::RecordDensityChange) = print(io, "RecordDensityChange()")
#
# norm ρ
mutable struct RecordDensityNorm{F} <: RecordAction
    recorded_values::Array{F,1}
end
DFTK.RecordDensityNorm(ρ::F) where {F} = RecordDensityNorm(typeof(norm(ρ)))
DFTK.RecordDensityNorm(T::Type=Float64) = RecordDensityNorm{T}(Array{T,1}())
function (r::RecordDensityNorm)(
    problem::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int
)
    current_ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ)
    return Manopt.record_or_reset!(r, norm(current_ρ), k)
end
Base.show(io::IO, ::RecordDensityNorm) = print(io, "RecordDensityNorm()")

#
#
# Stopping Criteria

"""
    StopWhenDensityChangeLess{T}

A `Manopt.jl` stopping criterion that indicates to stop then the change in the density `ρ`
is less than a given tolerance `tol`.

The stopping criterion assuemes that the density is either stored the objective, like the
`InsulatorEnergy` or is set as a parameter vie `get_parameter(objective, :ρ)`

# Fields
* `tolerance::F`: The tolerance for the change in density.
* `at_iteration::Int`: The iteration at which the stopping criterion was met.
* `last_ρ::T`: The last value of the density.
* `last_change::F`: The last change in the density in order to generate the reason for stopping.

# Constructor

```
StopWhenDensityChangeLess(tol::F, ρ::T) where {T,F<:Real}
```

Create the stopping criterion with a given tolerance `tol`. The provided density `ρ`
is only required to intialize the internal state.
"""
DFTK.StopWhenDensityChangeLess(tol::F, ρ::T) where {T,F<:Real}

mutable struct StopWhenDensityChangeLess{T,F<:Real} <: Manopt.StoppingCriterion
    tolerance::F
    at_iteration::Int
    last_ρ::T
    last_change::F
end
function DFTK.StopWhenDensityChangeLess(tol::F, ρ::T) where {T,F<:Real}
    StopWhenDensityChangeLess{T,F}(tol, -1, ρ, 2 * tol)
end
function (c::StopWhenDensityChangeLess)(
        problem::P, state::S, k::Int
    ) where {P<:Manopt.AbstractManoptProblem,S<:Manopt.AbstractManoptSolverState}
    current_ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ)
    if k == 0 # reset on init
        c.at_iteration = -1
        c.last_ρ .= 0
        c.last_change = 2 * c.tolerance
        return false
    end
    c.last_change = norm(current_ρ - c.last_ρ)
    c.last_ρ .= current_ρ
    if c.last_change < c.tolerance
        c.at_iteration = k
        return true
    end
    return false
end
function Manopt.get_reason(c::StopWhenDensityChangeLess)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the algorithm performed a step with a Density change ($(c.last_change)) less than $(c.tolerance)."
    end
    return ""
end
function Manopt.status_summary(c::StopWhenDensityChangeLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|Δρ| = $(c.last_change) < $(c.tolerance):\t$s"
end
function Base.show(io::IO, c::StopWhenDensityChangeLess)
    return print(io, "StopWhenDensityChangeLess with threshold $(c.tolerance).\n    $(Manopt.status_summary(c))")
end

#
#
# The direct minimization interface – the simple case
"""
    direct_minimization(basis::DFTK.PlaneWaveBasis{T}; kwargs...)

Compute a minimizer of the density energy function energy functional using the direct minimization method.

# Argument

* `basis::DFTK.PlaneWaveBasis{T}`: The plane wave basis to use for the DFT calculation.

# Keyword Arguments

* `maxiter=1000`: The maximum number of iterations for the optimization. If you set the `stopping_criterion=` directly, this keyword has no effect.
* `ψ=nothing`: The initial guess for the wave functions. If not provided, random orbitals will be generated.
* `ρ=guess_density(basis), # would be consistent with other scf solvers
* `tol=1e-6`: stop when the change in the density is less than this tolerance. If you set the `stopping_criterion=` directly, this keyword has no effect.
* `manifold=`[`Stiefel`](@extref `Manifolds.Stiefel`): the manifold the optimisation is running on.
  The current default cost function allows to also use ∞`Grassmann`](@extref `Manifolds.Grassmann`),
  both in their complex form.
* `record=[Manopt.RecordCost(), DFTK.RecordDensityChange(ρ), Manopt.RecordTime(:total)]`:
  specify what to record during the Iterations. If present, these are included in the returned named tuple

This uses several defaults from [`Manopt.jl`](@extref), for example it always uses
the [`quasi_newton`](@ref) solver.

To change this solver, use [`direct_minimizartion`](@ref direct_minimization(::PlaneWaveBasis, ::Manopt.AbstractManoptSolverState))`(basis, solver_state)`
"""
function DFTK.direct_minimization(basis::PlaneWaveBasis;
    ψ=nothing,
    ρ=guess_density(basis), # would be consistent with other scf solvers
    tol=1e-6,
    maxiter=1_000,
    manifold=Manifolds.Stiefel,
    record=[Manopt.RecordCost() => :Etot, DFTK.RecordDensityChange(ρ) => :Δρ, Manopt.RecordTime(; mode=:total) => :time]
)
    DFTK.direct_minimization(
        basis, Manopt.QuasiNewtonState;
        ψ=ψ, ρ=ρ, tol=tol, maxiter=maxiter, manifold=manifold, record=record
    )
end

"""
    direct_minimization(basis, state; kwargs...)

Compute a minimizer of the Hartree-Fock energy functional using the direct minimization method.

# Argument

* `basis::DFTK.PlaneWaveBasis{T}`: The plane wave basis to use for the DFT calculation.
* `state_type::Type{<:Manopt.AbstractManoptSolverState}`: The type of the Manopt solver to use.
    recommended: [`QuasiNewtonState`](@extref `Manopt.QuasiNewtonState`).

# Keyword Arguments

Similar to the simpler [`direct_minimizartion`](@ref direct_minimization(::PlaneWaveBasis))`(basis)`

* `maxiter=1_000`: The maximum number of iterations for the optimization. If you set the `stopping_criterion=` directly, this keyword has no effect.
    If you set the `stopping_criterion=` directly, this keyword has no effect.
* `ψ=nothing`: The initial guess for the wave functions. If not provided, random orbitals will be generated.
* `ρ=guess_density(basis), # would be consistent with other scf solvers
* `tol=1e-6`: stop when the change in the density is less than this tolerance. If you set the `stopping_criterion=` directly, this keyword has no effect.
    If you set the `stopping_criterion=` directly, this keyword has no effect.
* `cost=nothing`: provide an individual cost function. You then also have to provide `gradient=`
* `gradient=nothing`: provide an individual gradient function. You then also have to provide `cost=`
    If you provide both of these, for the access to `:Energies`, `:Hamiltonian`, `:HamiltonianEvaluation`, and `:ρ`
    you also have to implement `Manopt.get_parameter` for these.
* `evaluation`=[`InplaceEvaluation`](@extref `Manopt.InplaceEvaluation`) whether the gradient of the objective is allocating or in-place.,
* `manifold=`[`Stiefel`](@extref `Manifolds.Stiefel`): the manifold the optimisation is running on.
  The default cost function allows to also use [`Grassmann`](@extref `Manifolds.Grassmann`).
    If you set the `manifold_constructor=` directly, e.g. to switch to a real manifold, this keyword is ignored
* `preconditioner=`[`PreconditionerTPA`](@ref)` a preconditioner to use for the Newton equation or the gradient.
* `manifold_constructor=(n, k) -> manifold(n, k, ℂ)` the complete constructor for a single manifold within the
     product manifold the optimization is defined on.
* `stopping_criterion=[`StopAfterIteration`](@extref `Manopt.StopAfterIteration`)` `[`|`](@extref Manopt.StopWhenAny)` `[`StopWhenDensityChangeLess`](@ref)`(tol,deepcopy(ρ))`:
    a stopping criterion for the algorithm when to stop. Uses `maxiter=` and `tol=` as defaults, requires that `ρ` a density.
* `record=[[`RecordCost`](@extref `Manopt.RecordCost`)`() => :Etot, `[`RecordDensityChange`](@ref)`(ρ) => :Δρ`, `[`RecordTime`](@extref `Manopt.RecordTime`)`(; mode=:total) => :time]`
    specify values to record diring the iterations, where in a `pair` the second determines how to access the values; the three default ones are included in the returned named tuple
* `retraction_method = `[`ProjectionRetraction`](@extref `Manifolds.ProjectionRetraction`) the retration to use on the manifold to more into a certain direction.
* `vector_transport_method=`[`ProjectionTransport`](@extref `Manifolds.ProjectionTransport`)`()` the vector transport to use to move tangent vectors between tangent spaces.
* `stepsize = `[`ArmijoLinesearch`](@extref `Manopt.ArmijoLinesearch`)`(; retraction_method=retraction_method)`
   specify a step size rule.

All other keyword arguments are passed to the solver state constructor as well as to
both a decorate for the objective and the state.
This allows both to set other setting for a solver but also to add `debug=` funtionality
or even a `cache=` to the objective, though the default objective already caches parts
that both a cost and a gradient at the same point would require.

!!! note "Technical Note"
    This interface is still work in progress and might change in the furute even with changes
    that break compatibility.
"""
function DFTK.direct_minimization(
    basis::PlaneWaveBasis{T}, state_type::Type{<:Manopt.AbstractManoptSolverState};
    ψ=nothing,
    ρ=guess_density(basis), # would be consistent with other scf solvers
    tol=1e-6,
    maxiter=1_000,
    cost=nothing,
    gradient=nothing,
    preconditioner=DFTK.PreconditionerTPA,
    manifold=Manifolds.Stiefel,
    manifold_constructor=(n, k) -> manifold(n, k, ℂ),
    stopping_criterion = Manopt.StopAfterIteration(maxiter) | DFTK.StopWhenDensityChangeLess(tol,deepcopy(ρ)),
    evaluation = Manopt.InplaceEvaluation(),
    record=[Manopt.RecordCost() => :Etot, DFTK.RecordDensityChange(ρ) => :Δρ, Manopt.RecordTime(; mode=:total) => :time],
    retraction_method = Manifolds.ProjectionRetraction(),
    vector_transport_method=Manifolds.ProjectionTransport(),
    stepsize=Manopt.ArmijoLinesearch(; retraction_method=retraction_method),
    kwargs...
) where {T}
    # Part 1: Get DFTK variables
    #
    #
    model = basis.model
    @assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed
    filled_occ = DFTK.filled_occupation(model)
    Nk = length(basis.kpoints)
    n_spin = model.n_spin_components # Int. 2 if :collinear, 1 otherwise
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp) # Int
    ψ = isnothing(ψ) ? [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints] : ψ
    occupation = [filled_occ * ones(T, n_bands) for _ = 1:Nk]
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
    # Part II: Setup solver
    #
    #
    # Initialize the product manifold
    dimensions = size.(ψ) # Vector of touples of ψ dimensions
    manifold_array = map(dim -> manifold_constructor(dim[1], dim[2]), dimensions)
    product_manifold = ProductManifold(manifold_array...)
    Pks = [preconditioner(basis, kpt) for kpt in basis.kpoints]
    Preconditioner = ManoptPreconditionersWrapper!!(Nk, Pks, basis.kweights)
    # Repackage the ψ into a more efficient structure
    recursive_ψ = ArrayPartition(ψ...)
    if isnothing(cost) && isnothing(gradient)
        cost_rgrad! = InsulatorEnergy(basis,
            occupation,
            Nk,
            filled_occ,
            zero.(ψ),  #init different from ψ to avoid caching errors
            rand(product_manifold; vector_at=recursive_ψ), # init different from X to avoid caching errors
            deepcopy(ρ),
            deepcopy(energies),
            deepcopy(ham),
            0, # count of hamiltonian calls
    )
        cost = cost_rgrad! # local cost function
        if evaluation == InplaceEvaluation()
            grad = (M,X,p) -> cost_rgrad!(M, X, p)
        else
            grad = (M,p) -> cost_rgrad!(M, zero_vector(M,p), p)
        end
    else
        if isnothing(cost) | isnothing(gradient)
            error("Providing a cost or gradient function directly also requires the other one to be provided")
        end
    end
    # Build Objective & Problem
    objective = Manopt.ManifoldGradientObjective(cost, grad; evaluation=evaluation)
    deco_obj = Manopt.decorate_objective!(product_manifold, objective; kwargs...)
    problem = Manopt.DefaultManoptProblem(product_manifold, deco_obj)
    _stepsize = Manopt._produce_type(stepsize, product_manifold)
    state = state_type(
        product_manifold;
        p=recursive_ψ,
        stopping_criterion=stopping_criterion,
        preconditioner=QuasiNewtonPreconditioner(
            (M, Y, p, X) -> Preconditioner(M, Y, p, X); evaluation=InplaceEvaluation()
        ),
        direction=Manopt.PreconditionedDirectionRule(product_manifold,
            (M, Y, p, X) -> Preconditioner(M, Y, p, X);
            evaluation=InplaceEvaluation()
        ),
        # Set default, can still be overwritten by kwargs...
        stepsize=_stepsize,
        vector_transport_method=vector_transport_method,
        retraction_method=retraction_method,
        memory_size=10,
        X=cost_rgrad!( # Initial gradient
            product_manifold,
            zero_vector(product_manifold, recursive_ψ),
            recursive_ψ
        ),
        kwargs...
    )
    deco_state = Manopt.decorate_state!(state; record=record, kwargs...)
    Manopt.solve!(problem, deco_state)
    # Parti III: Collect result in a struct and return that
    #
    #
    # Bundle the variables in a NamedTuple for debugging:
    p = Manopt.get_solver_result(deco_state)
    recorded_values = Manopt.get_record_action(deco_state)
    t = recorded_values[:time]

    # The NamedTuple that is returned, collecting all results
    (
        info             = "This object is summarizing variables for debugging purposes",
        algorithm        = "$(state)",
        converged        = Manopt.has_converged(Manopt.get_state(deco_state, true)),
        product_manifold = product_manifold,
        basis            = basis,
        history_Δρ       = recorded_values[:Δρ],
        history_Etot     = recorded_values[:Etot],
        runtime_ns       = length(t) > 0 ? last(t) : zero(eltype(t)), # take the last recorded time if any recorded
        ψ                = collect(p.x), # reformat ArrayPartition back to a vector of matrices
        model       	 = model,
        n_bands_converge = n_bands,
        n_count          = Manopt.get_count(Manopt.get_state(deco_state, true), :Iterations),
        Nk          	 = Nk,
        occupation  	 = occupation,
        cost_value=get_cost(problem, p),
        cost_grad_value=get_gradient(problem, p),
        ρ = Manopt.get_parameter(Manopt.get_objective(problem), :ρ),
        energies = Manopt.get_parameter(Manopt.get_objective(problem), :Energies),
        ham = Manopt.get_parameter(Manopt.get_objective(problem), :Hamiltonian),
        n_matvec = Manopt.get_parameter(
            Manopt.get_objective(problem),
            :HamiltonianEvaluations
        ),
        # The “pure” solver state without debug/records.
        solver_state=Manopt.get_state(deco_state,true),
    )
end
#=
TODO
we could adapt to has_converged to report on convergence as well
=#
end