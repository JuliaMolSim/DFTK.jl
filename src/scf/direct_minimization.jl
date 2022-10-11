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
        Optim.project_tangent!(Optim.Stiefel(),
                               g_unpack[ik],
                               x_unpack[ik])
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
passed to `Optim.Options()`. Note that the resulting ψ are not
necessarily eigenvectors of the Hamiltonian.
"""
direct_minimization(basis::PlaneWaveBasis; kwargs...) = direct_minimization(basis, nothing; kwargs...)
function direct_minimization(basis::PlaneWaveBasis{T}, ψ0;
                             prec_type=PreconditionerTPA,
                             optim_solver=Optim.LBFGS, tol=1e-6, kwargs...) where {T}
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

    if ψ0 === nothing
        ψ0 = [random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
    end
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # we need to copy the reinterpret array here to not raise errors in Optim.jl
    # TODO raise this issue in Optim.jl
    pack(ψ) = copy(reinterpret_real(pack_ψ(ψ)))
    unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ0))
    unsafe_unpack(x) = unsafe_unpack_ψ(reinterpret_complex(x), size.(ψ0))

    # this will get updated along the iterations
    H = nothing
    energies = nothing
    ρ = nothing

    # computes energies and gradients
    function fg!(E, G, ψ)
        ψ = unpack(ψ)
        ρ = compute_density(basis, ψ, occupation)
        energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

        # The energy has terms like occ * <ψ|H|ψ>, so the gradient is 2occ Hψ
        if G !== nothing
            G = unsafe_unpack(G)
            for ik = 1:Nk
                mul!(G[ik], H.blocks[ik], ψ[ik])
                G[ik] .*= 2*filled_occ
            end
        end
        energies.total
    end

    manif = DMManifold(Nk, unsafe_unpack)

    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    P = DMPreconditioner(Nk, Pks, unsafe_unpack)

    kwdict = Dict(kwargs)
    optim_options = Optim.Options(; allow_f_increases=true, show_trace=true,
                                  x_tol=pop!(kwdict, :x_tol, tol),
                                  f_tol=pop!(kwdict, :f_tol, -1),
                                  g_tol=pop!(kwdict, :g_tol, -1), kwdict...)
    res = Optim.optimize(Optim.only_fg!(fg!), pack(ψ0),
                         optim_solver(P=P, precondprep=precondprep!, manifold=manif,
                                      linesearch=LineSearches.BackTracking()),
                         optim_options)
    ψ = unpack(res.minimizer)

    # Final Rayleigh-Ritz (not strictly necessary, but sometimes useful)
    eigenvalues = []
    for ik = 1:Nk
        Hψk = H.blocks[ik] * ψ[ik]
        F = eigen(Hermitian(ψ[ik]'Hψk))
        push!(eigenvalues, F.values)
        ψ[ik] .= ψ[ik] * F.vectors
    end

    εF = nothing  # does not necessarily make sense here, as the
                  # Aufbau property might not even be true

    # We rely on the fact that the last point where fg! was called is the minimizer to
    # avoid recomputing at ψ
    (ham=H, basis=basis, energies=energies, converged=true,
     ρ=ρ, ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF, optim_res=res)
end
