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
mutable struct CostGradFunctor!{T,S,P,X,R,E,H} # Parametric constructor
    basis::PlaneWaveBasis{T}
    occupation::Vector{S}
    Nk::Int
    filled_occ::Int
    dftk_p::P # store p as a vector of matrices, as expected by DFTK
    local_X::X  # Memoization
    ρ::R        # Memoization
    energies::E # Memoization
    ham::H      # Memoization
end
# Function shared by both cost and gradient of cost:
function _compute_density_energy_hamiltonian!(cgf::CostGradFunctor!,
    M::ProductManifold,
    p::ArrayPartition)
    copyto!(cgf.dftk_p, (copy(x) for x in p.x)) # deepcopyto!
    copyto!(cgf.ρ, compute_density(cgf.basis, cgf.dftk_p, cgf.occupation))
    # Below not inplace, but probably not that important.
    cgf.energies, cgf.ham = energy_hamiltonian(cgf.basis, cgf.dftk_p, cgf.occupation; cgf.ρ)
end
# The cost function:
function (cgf::CostGradFunctor!)(M::ProductManifold,
    p::ArrayPartition)
    # Memoization check: Are we still at the same point?
    if any(cgf.dftk_p[i] != p[M, i] for i in eachindex(cgf.dftk_p))
        _compute_density_energy_hamiltonian!(cgf, M, p)
    end
    return cgf.energies.total
end
# The gradient of cost function:
function (cgf::CostGradFunctor!)(M::ProductManifold,
    X::ArrayPartition,
    p::ArrayPartition)
    # Memoization check: Is this X allready been computed?
    if any(cgf.local_X[M, i] == X[M, i] for i in eachindex(cgf.dftk_p))
        # Are we still at the same point?
        if any(cgf.dftk_p[i] == p[M, i] for i in eachindex(cgf.dftk_p))
            return X
        end
    end
    # Memoization check: Are we still at the same point?
    if any(cgf.dftk_p[i] != p[M, i] for i in eachindex(cgf.dftk_p))
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
#
"""
    StopWhenFunctionLess

    StopWhenRhoDensityLess
    (Check, they are listed in )

TODO Check to adapt to wrap
  `ScfConvergenceDensity(tol)` (the default), `ScfConvergenceEnergy(tol)` or `ScfConvergenceForce(tol)`.


TODO Improve naming
TODO Document formula that is used

# Fields
* `tolerance::Float64`: tolerance for the stopping criterion
* `at_iteration::Int`: an internal field to indicate that at a certain iteration, the stop was indicated
* `get_current_measure::Function`: TODO: Check what this should be
* `last_measure::M`: TODO Check what this should be
* `Δ_measure::R`: TODO: Check what this should be
"""
mutable struct StopWhenFunctionLess{M,R<:Real} <: Manopt.StoppingCriterion
    tolerance::Float64
    at_iteration::Int
    #Call this one get_density,
    get_current_measure::Function # f(P, state) -> M
    last_measure::M # Maybe just density measure? becomes last_density
    Δ_measure::R
end
function StopWhenFunctionLess(tol::Float64, f::Function, m)
    #TODO: Can we initialise the last measure?
    return StopWhenFunctionLess{typeof(m)}(tol, -1, f, m, 0.0)
end
function (c::StopWhenFunctionLess{M})(
    problem::P, state::S, k::Int
) where {M,P<:Manopt.AbstractManoptProblem,S<:Manopt.AbstractManoptState}
    current_measure = c.get_current_measure(problem, state)
    # get_density(problem, get_iterate(state))
    # or use get_parameter(problem :Cost, :Density)  from Manopt
    if k == 0 # reset on init
        c.at_iteration = -1
        # TODO: Change to something more reasonable.
        c.last_measure .= current_measure
        c.Δ_measure = 0.0
        return false
    else
        # basis.dvol::Float64 = model.unit_cell_volume ./ prod(fft_size)
        # is the volume element for real-space integration:
        # sum(ρ) * dvol ~ ∫ρ.
        c.Δ_measure = norm(current_measure .- c.last_measure) * sqrt(basis.dvol)
        # gives the L²-norm of ρ normalized over a unit cell
        c.last_measure .= current_measure
        if c.tolerance > c.Δ_measure
            c.at_iteration = k
            return true
        end
    end
    return false
end
function Manopt.get_reason(c::StopWhenFunctionLess)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the algorithm performed a step with a change (log10: $(@sprintf("%.2e", log10(c.Δ_measure)))) less than ($(@sprintf("%.2e", c.tolerance))).\n"
    end
    return ""
end
function Manopt.status_summary(c::StopWhenFunctionLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|Δ_measure| = $(c.Δ_measure) < $(c.tolerance):\t$s"
end
function Base.show(io::IO, c::StopWhenFunctionLess)
    return print(io, "StopWhenChangeLess with threshold $(c.tolerance)$(s).\n    $(status_summary(c))")
end

"""
    direct_minimization(basis::DFTK.PlaneWaveBasis{T}; kwargs...

TODO: Documentation

# Argument
* `basis::DFTK.PlaneWaveBasis{T}`: The plane wave basis to use for the DFT calculation.

# Keyword Arguments
* `manifold_constructor=(n,k) -> Stiefel(n,k,ℂ)`: A function that constructs the manifold for the optimization.
  It maps the dimensions `(n,k)` to a manifold to be used per compponent. The default is the complex Stiefel manifold
"""
function DFTK.direct_minimization(basis::PlaneWaveBasis{T};
    ψ=nothing,
    tol=1e-6,
    maxiter=1_000,
    prec_functor=ManoptPreconditioner!,
    prec_type=DFTK.PreconditionerTPA,
    solver=QuasiNewtonState,       #previously: optim_method=Manopt.quasi_Newton,
    alphaguess=nothing,           #TODO: not implemented - wht was that? How to set?
    # This is the initial linesearch guess for a lineserach (LineSearches.jl or Armijo or so.)
    linesearch=ArmijoLineSearch(),
    #Former default: LineSearches.BackTracking(),
    manifold_constructor=(n, k) -> Manifolds.Stiefel(n, k, ℂ),
    # TODO:
    # Add a generic stopping criterion
    # Add a retraction
    # find a way to specify a good preconditioner keyword argument
    # find a way to maybe nicely specify cost and grad?
    # TODO Check the old callback= keyword
    kwargs...                     # TODO: pass kwargs to solver

) where {T}
    # Part 1: Get DFTK variables
    #
    #
    model = basis.model
    @assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed
    # Fermi level
    # Maximal occupation of a state (2 for non-spin-polarized electrons, 1 otherwise).
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    if isnothing(ψ)
        ψ = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
        # Filling it with wandom complex numbers. Prob. normalized
    else
        ψ = deepcopy(ψ)
    end
    occupation = [filled_occ * ones(T, n_bands) for _ = 1:Nk]
    ρ = compute_density(basis, ψ, occupation)
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ)

    # Part II: Setup solver
    #
    #
    # Initialize the product manifold
    dimensions = size.(ψ) # Vector of toupples of ψ dimensions
    manifold_array = map(dim -> manifold_constructor(dim[1], dim[2]), dimensions)
    product_manifold = ProductManifold(manifold_array...)
    # Initialize the preconditioner
    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    Preconditioner = prec_functor(Nk, Pks, basis.kweights)
    # Repackage the ψ into a more efficient structure
    recursive_ψ = ArrayPartition(ψ...)
    X = deepcopy(recursive_ψ) * NaN
    cost_rgrad! = CostGradFunctor!(basis,
        occupation,
        Nk,
        filled_occ,
        # We use the following ones as cache to here we generate copies of the corresponding
        # DFTK variables
        deepcopy(ψ),
        deepcopy(X),
        deepcopy(ρ),
        deepcopy(energies),
        deepcopy(ham)
    )
    # should be able to handle both Δψ and Δρ
    # TODO: Is that necessary? What would be best to store in the StoppingCriterion?
    function fetch_ρ(cost_rgrad!, state)
        return cost_rgrad!.objective.cost.ρ # if no LRU cache
        #return cost_rgrad!.objective.objective.objective.cost.ρ # return_objective = true
        #return cost_rgrad!.objective.objective.cost.ρ # using LRU cache
    end
    # Can be simplified if we just use Rho
    # maybe use set_parameter! for Rho on SC and that updates.
    # Check what yould be done for the other checks.deltaCost

    custom_stopping_crit = StopWhenFunctionLess(tol, fetch_ρ, ρ)

    # TODO: Generalize to all solver states instead of high-level interfaces
    solved_state = 0 # initialize
    if optim_method == Manopt.gradient_descent
        solved_state = gradient_descent!(
            product_manifold,
            #cost_rgrad_wrapper,
            cost_rgrad!, # cost functor;     (cgf)(M,p)
            cost_rgrad!, # gradient functor; (cgf)(M,X,p)
            recursive_ψ;
            return_state=true,
            evaluation=InplaceEvaluation(),
            stepsize=ls_custom,
            #
            direction=PreconditionedDirection(
                (M, Y, p, X) -> Preconditioner(M, Y, p, X);
                evaluation=InplaceEvaluation()
            ),
            stopping_criterion=custom_stopping_crit | StopAfterIteration(maxiter),
        )
    elseif optim_method == Manopt.quasi_Newton
        solved_state = quasi_Newton!(
            product_manifold,
            #cost_rgrad_wrapper,
            cost_rgrad!, # cost functor;     (cgf)(M,p)
            cost_rgrad!, # gradient functor; (cgf)(M,X,p)
            recursive_ψ;
            return_state=true,
            evaluation=InplaceEvaluation(),
            stepsize=ls_custom,
            memory_size=10,
            vector_transport_method=ProjectionTransport(),
            preconditioner=(M, Y, p, X) -> Preconditioner(M, Y, p, X),
            stopping_criterion=custom_stopping_crit | StopAfterIteration(maxiter),
        )
    end
    # TODO: Generate Problem
    # TODO: Decorate problem and state
    # TODO: call solve!

    # Parti III: Collect result in a struct and return that
    #
    #
    cost_function = cost_rgrad!(product_manifold, recursive_ψ)
    rgrad_cost_function = cost_rgrad!(product_manifold, X, recursive_ψ)
    # Bundle the variables in a NamedTuple for debugging:
    # TODO: Check which we need and should return
    # TODO: Check with the PR which one we should add here
    debug_info = (
        info="This object is summarizing variables for debugging purposes",
        #=
        recursive_ψ=recursive_ψ,
        ψ_reconstructed = typeof(collect(recursive_ψ.x)),
        product_manifold = product_manifold,
        model       	 = model,
        filled_occ  	 = filled_occ,
        n_spin      	 = n_spin,
        n_bands     	 = n_bands,
        #Nk          	 = Nk,
        ψ           	 = ψ,
        occupation  	 = occupation,
        ρ 				 = cost_rgrad!.ρ,
        =#
        cost_function=cost_function,
        rgrad_cost_function=rgrad_cost_function,
        solver_state=solver_state,
    )
    debug_info
end
end