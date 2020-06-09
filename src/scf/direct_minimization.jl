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
    Pks::Vector # Pks[ik] is the preconditioner for kpoint ik
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
                             optim_solver=Optim.LBFGS, tol=1e-6, kwargs...) where T
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = filled_occupation(model)
    n_bands = div(model.n_electrons, filled_occ)
    ortho(ψk) = Matrix(qr(ψk).Q)
    Nk = length(basis.kpoints)

    if ψ0 === nothing
        ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), n_bands))
              for kpt in basis.kpoints]
    end
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    ## vec and unpack
    # length of ψ[ik]
    lengths = [length(ψ0[ik]) for ik = 1:Nk]
    # psi[ik] is in psi_flat[starts[ik]:starts[ik]+lengths[ik]-1]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    pack(ψ) = vcat(Base.vec.(ψ)...) # TODO as an optimization, do that lazily? See LazyArrays
    unpack(ψ) = [@views reshape(ψ[starts[ik]:starts[ik]+lengths[ik]-1], size(ψ0[ik]))
                 for ik = 1:Nk]

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
            G = unpack(G)
            for ik = 1:Nk
                mul!(G[ik], H.blocks[ik], ψ[ik])
                G[ik] .*= 2*filled_occ
            end
        end
        sum(values(energies))
    end

    manif = DMManifold(Nk, unpack)

    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    P = DMPreconditioner(Nk, Pks, unpack)

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
    # These concepts do not make sense in direct minimization,
    # although we could maybe do a final Rayleigh-Ritz
    eigenvalues = nothing
    εF = nothing

    # We rely on the fact that the last point where fg! was called is the minimizer to
    # avoid recomputing at ψ
    (H=H, energies=energies, converged=true,
     ρ=ρ, ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF, optim_res=res, ham=H)
end
