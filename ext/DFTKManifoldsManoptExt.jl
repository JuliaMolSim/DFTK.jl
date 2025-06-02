module DFTKManifoldsManoptExt
    using Manopt
    using Manifolds
    using DFTK

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
    Y::ArrayPartition,
    p::ArrayPartition,
    X::ArrayPartition)
    # Update preconditioner
    for ik = 1:mp.Nk
        DFTK.precondprep!(mp.Pks[ik], p[M, ik])
    end
    # Precondition the gradient in-place
    for ik = 1:mp.Nk
        ldiv!(Y[M, ik], mp.Pks[ik], X[M, ik])
        ldiv!(mp.kweights[ik], Y[M, ik]) # maybe remove local_Y
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
    ψ::P        # last iterate, While it is an `ArrayPartition` `p` iterate in Manopt, we store it as a vector, i.e. in the ψ format for DFTK
    local_X::X  # last gradient
    ρ::R        # the last density
    energies::E # the last vector of energies
    ham::H      # the last Hamiltonian
end
# Function shared by both cost and gradient of cost:
function _compute_density_energy_hamiltonian!(cgf::CostGradFunctor!,
    M::ProductManifold,
    p::ArrayPartition)
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
    p::ArrayPartition)
    # Memoization check: Are we still at the same point?
    if any(cgf.ψ[i] != p[M, i] for i in eachindex(cgf.ψ))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    return cgf.energies.total
end
# The gradient of cost function:
function (cgf::CostGradFunctor!)(M::ProductManifold,
    X::ArrayPartition,
    p::ArrayPartition)
    # Memoization check: Is this X allready been computed?
    if all(cgf.local_X[M, i] == X[M, i] for i in eachindex(cgf.ψ))
        # Are we still at the same point?
        if all(cgf.ψ[i] == p[M, i] for i in eachindex(cgf.ψ))
            return X
        end
    end
    # Memoization check: Are we still at the same point?
    if any(cgf.ψ[i] != p[M, i] for i in eachindex(cgf.ψ))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    # Compute the Euclidean gradient in-place
    for ik = 1:cgf.Nk
        mul!(X[M, ik], cgf.ham.blocks[ik], p[M, ik]) # mul! overload in DFTK
        get_component(M, X, ik) .*= 2 * cgf.filled_occ * cgf.basis.kweights[ik] # Using get_component(), as "X[M, ik] .*=" is not yet supported in ManifoldsBase.jl
    end
    riemannian_gradient!(M, X, p, X) # Convert to Riemannian gradient
    copyto!(cgf.local_X, X) # Memoization
    return X
end

#
#
# Stopping Criteria
get_parameter(energy_costgrad::CostGradFunctor!, ::Val{:ρ}) = energy_costgrad.ρ
get_parameter(energy_costgrad::CostGradFunctor!, ::Val{:energies}) = energy_costgrad.energies

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
function StopWhenDensityChangeLess(tol::F, ρ::T) where {T, F<:Real}
    return StopWhenDensityChangeLess{T,F}(tol, -1, ρ, 2*tol)
end
function (c::StopWhenDensityChangeLess)(problem::P, state::S, k::Int) where {P<:Manopt.AbstractManoptProblem,S<:Manopt.AbstractManoptState}
    current_ρ = get_parameter(Manopt.get_objective(problem), :ρ)
    if k == 0 # reset on init
        c.at_iteration = -1
        c.last_ρ .= current_ρ
        c.last_change = 2*c.tolerance
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
    return print(io, "StopWhenDensityChangeLess with threshold $(c.tolerance)$.\n    $(status_summary(c))")
end
# TODO: This is also the same as generically Saying StopWhenCostChangeLess?
# If so we could generically implement that in Manopt as well
"""
    StopWhenEnergyChangeLess{T}


"""
mutable struct StopWhenEnergyChangeLess{T,F<:Real} <: Manopt.StoppingCriterion
    tolerance::F
    at_iteration::Int
    last_energy_total::T
    last_change::F
end
function StopWhenEnergyChangeLess(tol::F, energy_total::T=0.0) where {T, F<:Real}
    return StopWhenEnergyChangeLess{T,F}(tol, -1, energy_total, 2*tol)
end
function Manopt.get_reason(c::StopWhenEnergyChangeLess)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the algorithm performed a step with a Energy change ($(c.last_change)) less than $(c.tolerance)."
    end
    return ""
end
function Manopt.status_summary(c::StopWhenEnergyChangeLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|ΔE| = $(c.last_change) < $(c.tolerance):\t$s"
end
function Base.show(io::IO, c::StopWhenEnergyChangeLess)
    return print(io, "StopWhenEnergyChangeLess with threshold $(c.tolerance).\n    $(status_summary(c))")
end

# TODO: Add a StopWhenForceLess?

# TODO: Should we have Records / Debugs for
# * ρ
# * energies
# * total energy
# * anything else the cost computes interimswise?
# (fields from the solver can automatically be recorded anyways)

#
#
# The direct minimization interface

"""
    direct_minimization(basis::DFTK.PlaneWaveBasis{T}; kwargs...

TODO: Documentation

# Argument
* `basis::DFTK.PlaneWaveBasis{T}`: The plane wave basis to use for the DFT calculation.

# Keyword Arguments
* `manifold_constructor=(n,k) -> Stiefel(n,k,ℂ)`: A function that constructs the manifold for the optimization.
  It maps the dimensions `(n,k)` to a manifold to be used per component. The default is the complex Stiefel manifold
"""
function DFTK.direct_minimization(basis::PlaneWaveBasis{T};
    ψ=nothing,
    # small internal helpers to make keywords nicer
    _n_bands=div(basis.model.n_electrons, basis.model.n_spin_components * filled_occupation(model), RoundUp),
    _ψ=isnothing(ψ) ? [random_orbitals(basis, kpt, _n_bands) for kpt in basis.kpoints] : ψ,
    _occupation=[DFTK.filled_occupation(basis.model) * ones(T, _n_bands) for _ = 1:length(basis.kpoints)],
    _ρ=compute_density(basis, _ψ, _occupation),
    _energies=DFTK.energy(basis, _ψ, _occupation; ρ=_ρ)[:energies],
    tol=1e-6,
    maxiter=1_000,
    prec_functor=ManoptPreconditioner!,
    prec_type=DFTK.PreconditionerTPA,
    solver=QuasiNewtonState,       #previously: optim_method=Manopt.quasi_Newton,
    alphaguess=nothing,           #TODO: not implemented - wht was that? How to set?
    # This is the initial linesearch guess for a linesearch (LineSearches.jl or Armijo or so.)
    stepsize=ArmijoLineSearch(),
    #Former default: LineSearches.BackTracking(),
    manifold_constructor=(n, k) -> Manifolds.Stiefel(n, k, ℂ),
    # TODO: Should this be unifies with the functors in scf_callbacks.jl? We do not have the info-magic available here
    stopping_criterion = Manopt.StopAfterIteration(maxiter) | StopWhenDensityChangeLess(tol, _ρ),
    retraction_method=Manifolds.ProjectionRetraction(),
    vector_transport_method=Manifolds.ProjectionTransport(),
    evaluation=InplaceEvaluation(),
    # TODO
    # find a way to specify a good preconditioner keyword argument
    # find a way to maybe nicely specify cost and grad?
    # TODO Check the old callback= keyword – can we adapt those?
    kwargs...                     # TODO: pass kwargs to solver

) where {T}
    # Part 1: Get DFTK variables
    #
    #
    model = basis.model
    @assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed
    Nk = length(basis.kpoints)
    energies, ham = energy_hamiltonian(basis, _ψ, _occupation; ρ = _ρ)
    # Part II: Setup solver
    #
    #
    # Initialize the product manifold
    dimensions = size.(ψ) # Vector of toupples of ψ dimensions
    manifold_array = map(dim -> manifold_constructor(dim[1], dim[2]), dimensions)
    manifold = ProductManifold(manifold_array...)
    # Initialize the preconditioner TODO: Improve interface/construction
    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    Preconditioner = prec_functor(Nk, Pks, basis.kweights)
    # Repackage the ψ into a more efficient structure
    recursive_ψ = ArrayPartition(ψ...)
    if isnothing(cost_grad!!)
        cost_rgrad!! = CostGradFunctor!(basis,
            _occupation,
            Nk,
            filled_occupation(model),
            deepcopy(ψ),
            zero_vector(manifold, recursive_ψ), # X is a zero vector of the same type as ψ
            deepcopy(_ρ), # TODO: deepcopy necessary?
            deepcopy(energies),
            deepcopy(ham)
        )
    end
    local_cost = (M,p) -> cost_rgrad!(M, p) # local cost function
    if evaluation == InplaceEvaluation()
        local_grad!! = (M,X,p) -> cost_rgrad!(M, X, p)
    else
        local_grad!! = (M,p) -> cost_rgrad!(M, zero_vector(M,p), p)
    end
    # Build Objective & Problem
    objective = Manopt.ManifoldGradientObjective(local_cost, local_grad!!; evaluation=evaluation)
    deco_obj = Manopt.decorate!(manifold, objective; kwargs...)
    problem = Manopt.DefaultManoptProblem(manifold, deco_obj)

    state = solver(
        manifold,
        recursive_ψ;
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
        stepsize=stepsize,
    # TODO: Add a way to specify the preconditioner depending on the solver
    # preconditioner=Preconditioner,
    kwargs...)
    #=  Gradient Descent Precon was:
            direction=PreconditionedDirection(
                (M, Y, p, X) -> Preconditioner(M, Y, p, X);
                evaluation=InplaceEvaluation()
            ),
        Quasi Newton Precon was:
            preconditioner=(M, Y, p, X) -> Preconditioner(M, Y, p, X),
        ...and it furher had
            memory_size=10,
        ...but they could also just be kwargs I think
    =#
    deco_state = Manopt.decorate!(state; kwargs...)
    Manopt.solve!(problem, deco_state;)
    # Parti III: Collect result in a struct and return that
    #
    #
    # Bundle the variables in a NamedTuple for debugging:
    # TODO: Check which we need and should return
    # TODO: Check with the PR which one we should add here
    debug_info = (
        info="This object is summarizing variables for debugging purposes",
        recursive_ψ=recursive_ψ,
        product_manifold=manifold,
        #=
        ψ_reconstructed = typeof(collect(recursive_ψ.x)),
        model       	 = model,
        filled_occ  	 = filled_occ,
        n_spin      	 = n_spin,
        n_bands     	 = n_bands,
        #Nk          	 = Nk,
        ψ           	 = ψ,
        occupation  	 = occupation,
        ρ 				 = cost_rgrad!.ρ,
        =#
        # The “pure” solver state without debug/records.
        solver_state=Manopt.get_state(deco_state,true),
    )
    debug_info
end
end