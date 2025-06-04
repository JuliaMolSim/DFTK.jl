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
    ManoptPreconditioner!{T,S}

Define a wrapper for a preconditioner that applied `DFTK.precondprep!` with
a separate precinditioner `Pks[ik]` for each k-point `ik` in the

# Fields
* `Nk::Int`: Number of k-points
* `Pks::Vector{T}`: Preconditioners for each k-point, where `Pks[ik]` is the
  preconditioner for k-point `ik`.
* `kweights::Vector{S}`: Weights for each k-point, where `kweights[ik]` is
  the weight for k-point `ik`.
"""
struct ManoptPreconditioner!{T,S}
    Nk::Int
    Pks::Vector{T}
    kweights::Vector{S}
end
function (mp::ManoptPreconditioner!)(M::ProductManifold,
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

"""
    CostGradFunctor!{T,S,P,X,R,E,H}

TODO: Document and improve naming
"""
mutable struct CostGradFunctor!{T,S,P,X,R,E,H}
    basis::PlaneWaveBasis{T}
    occupation::Vector{S}
    Nk::Int
    filled_occ::Int
    ψ::P        # last iterate, While we usually have an  iterate `ArrayPartition` `p` iterate in Manopt, we store it as a vector, i.e. in the ψ format for DFTK
    local_X::X  # last gradient
    ρ::R        # the last density
    energies::E # the last vector of energies
    ham::H      # the last Hamiltonian
end
# Function shared by both cost and gradient of cost:
function _compute_density_energy_hamiltonian!(cgf::CostGradFunctor!,
    M::ProductManifold,
    p)
    # Can we improve this by copying elementwise?
    # copyto!(cgf.ψ, (copy(x) for x in p.x)) # deepcopyto!
    # Maybe like
    for i in eachindex(cgf.ψ)
        copyto!(cgf.ψ[i], p[M,i])
    end
    copyto!(cgf.ρ, compute_density(cgf.basis, cgf.ψ, cgf.occupation))
    # Below not inplace, but probably not that important.
    cgf.energies, cgf.ham = energy_hamiltonian(cgf.basis, cgf.ψ, cgf.occupation; cgf.ρ)
end
# The cost function:
function (cgf::CostGradFunctor!)(M::ProductManifold,
    p)
    # Memoization check: Are we still at the same point?
    if all(cgf.ψ[i] == p[M, i] for i in eachindex(cgf.ψ))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    return cgf.energies.total
end
# The gradient of cost function:
function (cgf::CostGradFunctor!)(M::ProductManifold, X, p)
    # Memoization check: Is this X allready been computed?
    if all(cgf.local_X[M, i] == X[M, i] for i in eachindex(cgf.ψ))
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
    copyto!(cgf.local_X, X) # Memoization
    return X
end

#
#
# Stopping Criteria
get_parameter(objective::Manopt.AbstractManifoldCostObjective, s) = get_parameter(Manopt.get_cost_function(objective), s)
get_parameter(energy_costgrad::CostGradFunctor!, s::Symbol) = get_parameter(energy_costgrad, Val(s))
get_parameter(energy_costgrad::CostGradFunctor!, ::Val{:ρ}) = energy_costgrad.ρ

"""
    StopWhenDensityChangeLess{T}

TODO: Document
"""
mutable struct StopWhenDensityChangeLess{T,F<:Real} <: Manopt.StoppingCriterion
    tolerance::F
    at_iteration::Int
    #TODO: Ask and document: Is ρ just a 4D array or does it live on some manifold?
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

# TODO: Should we have Records / Debugs for
# * ρ ? A user could also easily to a record/debug themselves.

#
#
# The direct minimization interface

"""
    direct_minimization(basis::DFTK.PlaneWaveBasis{T}; kwargs...

TODO: Documentation

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
* `alphaguess=nothing`: ???
* `stopping_criterion=Manopt.StopAfterIteration(maxiter) | StopWhenDensityChangeLess(tol)`: The stopping criterion for the optimization.
* `evaluation=InplaceEvaluation()`: The evaluation strategy for the cost and gradient.

All other keyword argments are passed to the solver state constructor as well as to
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
    alphaguess=nothing, # TODO: not implemented - wht was that? How to set?
    _manifold=Manifolds.Stiefel,
    manifold_constructor=(n, k) -> _manifold(n, k, ℂ),
    stopping_criterion = Manopt.StopAfterIteration(maxiter) | StopWhenDensityChangeLess(tol,deepcopy(ρ)),
    evaluation=Manopt.InplaceEvaluation(),
    retraction_method=Manifolds.ProjectionRetraction(),
    vector_transport_method=Manifolds.ProjectionTransport(),
    stepsize=Manopt.ArmijoLinesearch(; retraction_method=retraction_method),
    # TODO
    # find a way to maybe nicely specify cost and grad?
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
    # Initialize the preconditioner TODO: Improve interface/construction
    Pks = [preconditioner(basis, kpt) for kpt in basis.kpoints]
    Preconditioner = ManoptPreconditioner!(Nk, Pks, basis.kweights)
    # Repackage the ψ into a more efficient structure
    recursive_ψ = ArrayPartition(ψ...)
    #TODO Maybe move to a keyword argument?
    cost_rgrad! = CostGradFunctor!(basis,
        occupation,
        Nk,
        filled_occ,
        deepcopy(ψ),
        zero_vector(_manifold, recursive_ψ), # X is a zero vector of the same type as ψ
        deepcopy(ρ), # TODO: deepcopy necessary?
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
        # TODO: Add a way to specify the preconditioner depending on the solver
        # These two passed to all solvers might be misleading
        preconditioner=QuasiNewtonPreconditioner((M, Y, p, X) -> Preconditioner(M, Y, p, X); evaluation=InplaceEvaluation()),
        direction=Manopt.PreconditionedDirectionRule(_manifold,
            (M, Y, p, X) -> Preconditioner(M, Y, p, X);
            evaluation=InplaceEvaluation()
        ),
        # Set default, can still be overwritten by kwargs...
        stepsize=_stepsize,
        vector_transport_method=vector_transport_method,
        memory_size=10,
        retraction_method=retraction_method,
        kwargs...
    )
    deco_state = Manopt.decorate_state!(state; kwargs...)
    Manopt.solve!(problem, deco_state)
    # Parti III: Collect result in a struct and return that
    #
    #
    # Bundle the variables in a NamedTuple for debugging:
    # TODO: Check which we need and should return
    # TODO: Check with the PR which one we should add here
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
        # The “pure” solver state without debug/records.
        solver_state=Manopt.get_state(deco_state,true),
    )
    return debug_info
end
end