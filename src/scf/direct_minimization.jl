# Direct minimization of the energy

using Optim
using LineSearches

# This is all a bit annoying because our ψ is represented as ψ[k][G,n], and Optim accepts
# only dense arrays. We do a bit of back and forth using custom `pack` (ours -> optim's) and
# `unpack` (optim's -> ours) functions

# Orbitals inside each kblock must be kept orthogonal: the
# project_tangent and retract work per kblock
struct DMManifold <: Optim.Manifold
    Nk::Int
    unpack::Function
end
function Optim.project_tangent!(m::DMManifold, g, x)
    g_unpack = m.unpack(g)
    x_unpack = m.unpack(x)
    for ik = 1:m.Nk
        Optim.project_tangent!(Optim.Stiefel(), g_unpack[ik], x_unpack[ik])
    end
    g
end
function Optim.retract!(m::DMManifold, x)
    x_unpack = m.unpack(x)
    for ik = 1:m.Nk
        Optim.retract!(Optim.Stiefel(), x_unpack[ik])
    end
    x
end

# Array of preconditioners
struct DMPreconditioner
    Nk::Int
    Pks::Vector # Pks[ik] is the preconditioner for k-point ik
    unpack::Function
end
function LinearAlgebra.ldiv!(p, P::DMPreconditioner, d)
    p_unpack = P.unpack(p)
    d_unpack = P.unpack(d)
    for ik = 1:P.Nk
        ldiv!(p_unpack[ik], P.Pks[ik], d_unpack[ik])
    end
    p
end
function LinearAlgebra.dot(x, P::DMPreconditioner, y)
    x_unpack = P.unpack(x)
    y_unpack = P.unpack(y)
    sum(dot(x_unpack[ik], P.Pks[ik], y_unpack[ik])
        for ik = 1:P.Nk)
end
function precondprep!(P::DMPreconditioner, x)
    x_unpack = P.unpack(x)
    for ik = 1:P.Nk
        precondprep!(P.Pks[ik], x_unpack[ik])
    end
    P
end


"""
Computes the ground state by direct minimization. `kwargs...` are
passed to `Optim.Options()` and `optim_method` selects the optim approach
which is employed.
"""
function direct_minimization(basis::PlaneWaveBasis{T};
                             ψ=nothing,
                             tol=1e-6,
                             is_converged=ScfConvergenceDensity(tol),
                             maxiter=1_000,
                             prec_type=PreconditionerTPA,
                             callback=ScfDefaultCallback(),
                             optim_method=Optim.LBFGS,
                             linesearch=LineSearches.BackTracking(),
                             kwargs...) where {T}
    if mpi_nprocs() > 1
        # need synchronization in Optim
        error("Direct minimization with MPI is not supported yet")
    end
    model = basis.model
    @assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed Fermi level
    filled_occ = filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)

    if isnothing(ψ)
        ψ = [random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
    end
    occupation = [filled_occ * ones(T, n_bands) for _ = 1:Nk]

    # we need to copy the reinterpret array here to not raise errors in Optim.jl
    # TODO raise this issue in Optim.jl
    pack(ψ) = copy(reinterpret_real(pack_ψ(ψ)))
    unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ))
    unsafe_unpack(x) = unsafe_unpack_ψ(reinterpret_complex(x), size.(ψ))

    # This will get updated along the iterations
    ρ    = nothing
    ham  = nothing
    info = nothing
    energies  = nothing
    converged = false
    start_ns  = time_ns()
    history_Etot = T[]
    history_Δρ   = T[]

    # Will be later overwritten by the Optim-internal state, which we need in the
    # callback to access certain quantities for convergence control.
    optim_state = nothing

    function compute_ρout(ψ, optim_state)
        # This is the current preconditioned, but unscaled gradient, which implies that
        # the next step would be ρout - ρ. We thus record convergence, but let Optim do
        # one more step.
        δψ = unsafe_unpack(optim_state.s)
        ψ_next = [ortho_qr(ψ[ik] - δψ[ik]) for ik in 1:Nk]
        compute_density(basis, ψ_next, occupation)
    end

    function optim_callback(ts)
        ts.iteration < 1 && return false
        converged        && return true
        ρout = compute_ρout(ψ, optim_state)
        Δρ = ρout - ρ
        push!(history_Δρ,   norm(Δρ) * sqrt(basis.dvol))
        push!(history_Etot, energies.total)

        info = (; ham, basis, energies, occupation, ρout, ρin=ρ, ψ,
                runtime_ns=time_ns() - start_ns, history_Δρ, history_Etot,
                stage=:iterate, algorithm="DM", n_iter=ts.iteration, optim_state)

        converged = is_converged(info)
        info = callback(info)

        false
    end

    # computes energies and gradients
    function fg!(::Any, G, x)
        ψ = unpack(x)
        ρ = compute_density(basis, ψ, occupation)
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ)

        # The energy has terms like occ * <ψ|H|ψ>, so the gradient is 2occ Hψ
        if G !== nothing
            G = unsafe_unpack(G)
            for ik = 1:Nk
                mul!(G[ik], ham.blocks[ik], ψ[ik])
                G[ik] .*= 2*filled_occ
            end
        end
        energies.total
    end

    manifold = DMManifold(Nk, unsafe_unpack)
    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    P = DMPreconditioner(Nk, Pks, unsafe_unpack)

    optim_options = Optim.Options(; allow_f_increases=true,
                                  callback=optim_callback,
                                  # Disable convergence control by Optim
                                  x_tol=-1, f_tol=-1, g_tol=-1,
                                  iterations=maxiter, kwargs...)
    optim_solver = optim_method(; P, precondprep=precondprep!, manifold, linesearch)
    ψ_packed = pack(ψ)
    objective = OnceDifferentiable(Optim.only_fg!(fg!), ψ_packed, zero(T); inplace=true)
    optim_state = Optim.initial_state(optim_solver, optim_options, objective, ψ_packed)
    res = Optim.optimize(objective, ψ_packed, optim_solver, optim_options, optim_state)
    ψ = unpack(Optim.minimizer(res))

    # Final Rayleigh-Ritz (not strictly necessary, but sometimes useful)
    eigenvalues = Vector{T}[]
    for ik = 1:Nk
        Hψk = ham[ik] * ψ[ik]
        F = eigen(Hermitian(ψ[ik]'Hψk))
        push!(eigenvalues, F.values)
        ψ[ik] .= ψ[ik] * F.vectors
    end

    εF = nothing  # does not necessarily make sense here, as the
                  # Aufbau property might not even be true

    # We rely on the fact that the last point where fg! was called is the minimizer to
    # avoid recomputing at ψ
    info = (; ham, basis, energies, converged, ρ, eigenvalues, occupation, εF,
            n_bands_converge=n_bands, n_iter=Optim.iterations(res),
            runtime_ns=time_ns() - start_ns, history_Δρ, history_Etot,
            ψ, stage=:finalize, algorithm="DM", optim_res=res)
    callback(info)
    info
end
