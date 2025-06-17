module DFTKManifoldsManoptRATExt
    using DFTK
    using Manopt
    using Manifolds
    using RecursiveArrayTools

function __init__()
    # A small trick to make the stopping criterion available globally
    setglobal!(DFTK, :StopWhenDensityChangeLess, StopWhenDensityChangeLess)
    return nothing
end

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
function (mp::ManoptPreconditionersWrapper!!)(M::ProductManifold,
    Y,
    p,
    X)
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
    HartreeFockEnergyCostGrad{T,S,P,X,R,E,H}

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
mutable struct HartreeFockEnergyCostGrad{T,S,P,X,R,E,H}
    basis::PlaneWaveBasis{T}
    occupation::Vector{S}
    Nk::Int
    filled_occ::Int
    ψ::P        # last iterate, While we usually have an  iterate `ArrayPartition` `p` iterate in Manopt, we store it as a vector, i.e. in the ψ format for DFTK
    X::X        # last gradient
    ρ::R        # the last density
    energies::E # the last vector of energies
    ham::H      # the last Hamiltonian
end
# Function shared by both cost and gradient of cost:
function _compute_density_energy_hamiltonian!(cgf::HartreeFockEnergyCostGrad, M::ProductManifold, p)
    # Can we improve this by copying elementwise?
    # copyto!(cgf.ψ, (copy(x) for x in p.x)) # deepcopyto!
    # Maybe like
    for i in eachindex(cgf.ψ)
        copyto!(cgf.ψ[i], p[M,i])
    end
    copyto!(cgf.ρ, compute_density(cgf.basis, cgf.ψ, cgf.occupation))
    # Below not inplace, but probably not that important.
    cgf.energies, cgf.ham = energy_hamiltonian(cgf.basis, cgf.ψ, cgf.occupation; cgf.ρ)
    return cgf
end
# The cost function:
function (cgf::HartreeFockEnergyCostGrad)(M::ProductManifold,p)
    # Memoization check: Are we still at the same point?
    if all(cgf.ψ[i] == p[M, i] for i in eachindex(cgf.ψ))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    return cgf.energies.total
end
# The gradient of cost function:
function (cgf::HartreeFockEnergyCostGrad)(M::ProductManifold, X, p)
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
        Manifolds.get_component(M, X, ik) .*= 2 * cgf.filled_occ * cgf.basis.kweights[ik] # Using get_component(), as "X[M, ik] .*=" is not yet supported in ManifoldsBase.jl
    end
    riemannian_gradient!(M, X, p, X) # Convert to Riemannian gradient
    copyto!(cgf.X, X) # Memoization
    return X
end

#
#
# Stopping Criteria
get_parameter(objective::Manopt.AbstractManifoldCostObjective, s) = get_parameter(Manopt.get_cost_function(objective), s)
get_parameter(energy_costgrad::HartreeFockEnergyCostGrad, s::Symbol) = get_parameter(energy_costgrad, Val(s))
get_parameter(energy_costgrad::HartreeFockEnergyCostGrad, ::Val{:ρ}) = energy_costgrad.ρ

# TODO/DISCUSS:
# *  Can we also accept a manifold here and conclude the `ρ` from there?
"""
    StopWhenDensityChangeLess{T}

A `Manopt.jl` stopping criterion that indicates to stop then the change in the density `ρ`
is less than a given tolerance `tol`.

The stopping criterion assuemes that the density is either stored the objective, like the
`HartreeFockEnergyCostGrad` or is set as a parameter vie `get_parameter(objective, :ρ)`

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
mutable struct StopWhenDensityChangeLess{T,F<:Real} <: Manopt.StoppingCriterion
    tolerance::F
    at_iteration::Int
    last_ρ::T
    last_change::F
end
function StopWhenDensityChangeLess(tol::F, ρ::T) where {T,F<:Real}
    return StopWhenDensityChangeLess{T,F}(tol, -1, ρ, 2 * tol)
end
function (c::StopWhenDensityChangeLess)(problem::P, state::S, k::Int) where {P<:Manopt.AbstractManoptProblem,S<:Manopt.AbstractManoptSolverState}
    current_ρ = get_parameter(Manopt.get_objective(problem), :ρ)
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
    return print(io, "StopWhenDensityChangeLess with threshold $(c.tolerance).\n    $(status_summary(c))")
end

# TODO/DISCUSS: Should we have Records / Debugs for
# * ρ ? A user could then easily use `record = [:ρ] to record it
# * a debug is maybe not so useful, since it seems to be a large array, but its norm maybe?

# TODO/Discuss:
# * Should a user be able to provide their own cost/grad?
#   Then we would have to change a few small things in the setup.

#
#
# The direct minimization interface
"""
    direct_minimization(basis::DFTK.PlaneWaveBasis{T}; kwargs...)

Compute a minimizer of the Hartree-Fock energy functional using the direct minimization method.

# Argument

* `basis::DFTK.PlaneWaveBasis{T}`: The plane wave basis to use for the DFT calculation.

# Keyword Arguments

* `manifold_constructor=(n,k) -> Stiefel(n,k,ℂ)`: A function that constructs a single component of the product manifold, which is the domain of the energy functional (cost)
  It maps the dimensions `(n,k)` to a manifold to be used per component. The default is the complex Stiefel manifold
* `ψ=nothing`: The initial guess for the wave functions. If not provided, random orbitals will be generated.
* `tol=1e-6`: stop when the change in the density is less than this tolerance. If you set the `stopping_criterion=` directly, this keyword has no effect.
* `maxiter=1000`: The maximum number of iterations for the optimization. If you set the `stopping_criterion=` directly, this keyword has no effect.
* `preconditioner=DFTK.PreconditionerTPA`: The preconditioner to use for the optimization.
* `solver=QuasiNewtonState`: The solver to use for the optimization. Defaults to a quasi-Newton method.
* `stopping_criterion=Manopt.StopAfterIteration(maxiter) | StopWhenDensityChangeLess(tol)`: The stopping criterion for the optimization.
* `evaluation=InplaceEvaluation()`: The evaluation strategy for the cost and gradient.

All other keyword arguments are passed to the solver state constructor as well as to
both a decorate for the objective and the state.
This allows to change defaults in the solver settings,
add for example a cache “around” the objective or add debug and/or recording functionality to the solver run.
"""
function DFTK.direct_minimization(basis::PlaneWaveBasis{T};
    ψ=nothing,
    ρ=guess_density(basis), # would be consistent with other scf solvers
    tol=1e-6,
    maxiter=1_000,
    # TODO Naming and format,
    preconditioner=DFTK.PreconditionerTPA,
    solver=QuasiNewtonState,
    _manifold=Manifolds.Stiefel,
    manifold_constructor=(n, k) -> _manifold(n, k, ℂ),
    stopping_criterion = Manopt.StopAfterIteration(maxiter) | StopWhenDensityChangeLess(tol,deepcopy(ρ)),
    evaluation=Manopt.InplaceEvaluation(),
    retraction_method=Manifolds.ProjectionRetraction(),
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
    _manifold = ProductManifold(manifold_array...)
    Pks = [preconditioner(basis, kpt) for kpt in basis.kpoints]
    Preconditioner = ManoptPreconditionersWrapper!!(Nk, Pks, basis.kweights)
    # Repackage the ψ into a more efficient structure
    recursive_ψ = ArrayPartition(ψ...)
    cost_rgrad! = HartreeFockEnergyCostGrad(basis,
        occupation,
        Nk,
        filled_occ,
        zero.(ψ),  #init different from ψ to avoid caching errors
        rand(_manifold; vector_at=recursive_ψ), # init different from X to avoid caching errors
        deepcopy(ρ),
        deepcopy(energies),
        deepcopy(ham),
   )
    local_cost = cost_rgrad! # local cost function
    if evaluation == InplaceEvaluation()
        local_grad!! = (M,X,p) -> cost_rgrad!(M, X, p)
    else
        local_grad!! = (M,p) -> cost_rgrad!(M, zero_vector(M,p), p)
    end
    # Build Objective & Problem
    objective = Manopt.ManifoldGradientObjective(local_cost, local_grad!!; evaluation=evaluation)
    deco_obj = Manopt.decorate_objective!(_manifold, objective; kwargs...)
    problem = Manopt.DefaultManoptProblem(_manifold, deco_obj)
    _stepsize = Manopt._produce_type(stepsize, _manifold)
    state = solver(
        _manifold;
        p=recursive_ψ,
        stopping_criterion=stopping_criterion,
        preconditioner=QuasiNewtonPreconditioner((M, Y, p, X) -> Preconditioner(M, Y, p, X); evaluation=InplaceEvaluation()),
        direction=Manopt.PreconditionedDirectionRule(_manifold,
            (M, Y, p, X) -> Preconditioner(M, Y, p, X);
            evaluation=InplaceEvaluation()
        ),
        # Set default, can still be overwritten by kwargs...
        stepsize=_stepsize,
        vector_transport_method=vector_transport_method,
        retraction_method=retraction_method,
        memory_size=10,
        X=cost_rgrad!(_manifold, zero_vector(_manifold, recursive_ψ), recursive_ψ), # Initial gradient
        kwargs...
    )
    deco_state = Manopt.decorate_state!(state; kwargs...)
    Manopt.solve!(problem, deco_state)
    # Parti III: Collect result in a struct and return that
    #
    #
    # Bundle the variables in a NamedTuple for debugging:
    p = get_solver_result(deco_state)
    debug_info = (
        info="This object is summarizing variables for debugging purposes",
        product_manifold=_manifold,
        ψ_reconstructed = collect(recursive_ψ.x),
        model       	 = model,
        n_bands     	 = n_bands,
        Nk          	 = Nk,
        ψ           	 = ψ,
        occupation  	 = occupation,
        ρ                = cost_rgrad!.ρ,
        cost_value=get_cost(problem, p),
        cost_grad_value=get_gradient(problem, p),
        # The “pure” solver state without debug/records.
        solver_state=Manopt.get_state(deco_state,true),
    )
    return debug_info
end
# TODO/Discuss
# * What should the `debug_info` contain?
end