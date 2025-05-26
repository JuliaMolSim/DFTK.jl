module DFTKManoptExt
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

TODO Improve naming
"""
mutable struct StopWhenFunctionLess{M} <: Manopt.StoppingCriterion
    tolerance::Float64
    at_iteration::Int
    get_current_measure::Function # f(P, state) -> M
    last_measure::Union{Nothing,M}
    Δ_measure::Union{Nothing,Float64}
end
function StopWhenFunctionLess(tol::Float64, f::Function, m)
    return StopWhenFunctionLess{typeof(m)}(tol, -1, f, nothing, nothing)
end
function (c::StopWhenFunctionLess{M})(
    cgf::CostGradFunctor!, state::S, k::Int
) where {M,CostGradFunctor!,S<:AbstractManoptSolverState}
    if k == 0 # reset on init
        c.at_iteration = -1
        c.last_measure = nothing
        c.Δ_measure = nothing
        return false
    end
    current_measure = c.get_current_measure(cgf, state)

    if c.last_measure === nothing
        c.last_measure = deepcopy(current_measure)
        return false
    end
    # basis.dvol::Float64 = model.unit_cell_volume ./ prod(fft_size)
    # is the volume element for real-space integration:
    # sum(ρ) * dvol ~ ∫ρ.
    c.Δ_measure = norm(current_measure .- c.last_measure) * sqrt(basis.dvol)
    # ^ gives the L²-norm of ρ normalized over a unit cell
    c.last_measure .= current_measure
    if c.tolerance > c.Δ_measure
        c.at_iteration = k
        return true
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
    alphaguess=nothing,           #TODO: not implemented
    # linesearch=LineSearches.BackTracking(), #TODO: Take one form Manopt as default?
    manifold_constructor=(n, k) -> Manifolds.Stiefel(n, k, ℂ),
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
    # ⚠️ TODO: It's hard-coded to be Stiefel, but could be made to be generalized
    # Stifel & Grassmann
    dimensions = size.(ψ) # Vector of toupples of ψ dimensions
    # Should just be able to switch to grassmann on this line only!!
    manifold_array = map(dim -> manifold_constructor(dim[1], dim[2]), dimensions)
    product_manifold = ProductManifold(manifold_array...)

    # Initialize the preconditioner
    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    Preconditioner = prec_functor(Nk, Pks, basis.kweights)

    # Repackage the ψ into a more efficient structure
    recursive_ψ = ArrayPartition(ψ...)
    X = deepcopy(recursive_ψ) * NaN
    cost_rgrad! = CostGradFunctor!(basis,      # not modifying
        occupation, # not modifying
        Nk,         # not modifying
        filled_occ, # not modifying
        deepcopy(ψ),
        deepcopy(X),
        deepcopy(ρ),
        deepcopy(energies),
        deepcopy(ham)
    )
    # TODO : move to a generic linesearch argument
    ls_custom = Manopt.LineSearchesStepsize(product_manifold, linesearch)

    # should be able to handle both Δψ and Δρ
    # TODO: Is that necessary?
    function fetch_ρ(cost_rgrad!, state)
        return cost_rgrad!.objective.cost.ρ # if no LRU cache
        #return cost_rgrad!.objective.objective.objective.cost.ρ # return_objective = true
        #return cost_rgrad!.objective.objective.cost.ρ # using LRU cache
    end
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
    # Parti III: Collect result in a struct and return that
    #
    #
    cost_function = cost_rgrad!(product_manifold, recursive_ψ)
    rgrad_cost_function = cost_rgrad!(product_manifold, X, recursive_ψ)
    # Bundle the variables in a NamedTuple for debugging:
    # TODO: Check which we need and should return
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
        solved_state=solved_state,
    )
    debug_info
end
end