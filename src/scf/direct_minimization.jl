# Direct minimization of the energy

using Optim

# This is all a bit annoying because our Psi is represented as Psi[k][G,n], and Optim accepts only dense arrays
# We do a bit of back and forth using custom `pack` (ours -> optim's) and `unpack` (optim's -> ours) functions

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
    Pks::Array # Pks[ik] is the preconditioner for kpoint ik
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
Computes the ground state by direct minimization. `kwargs...` are passed to `Optim.Options()`.
"""
direct_minimization(basis::PlaneWaveBasis; kwargs...) = direct_minimization(basis, nothing; kwargs...)
function direct_minimization(basis::PlaneWaveBasis{T}, Psi0;
                             prec_type=PreconditionerTPA,
                             optim_solver=Optim.LBFGS, kwargs...) where T
    model = basis.model
    @assert model.spin_polarisation in (:none, :spinless)
    @assert model.assume_band_gap # temperature is not yet supported
    filled_occ = filled_occupation(model)
    n_bands = div(model.n_electrons, filled_occ)
    ortho(Psik) = Matrix(qr(Psik).Q)
    Nk = length(basis.kpoints)

    if Psi0 === nothing
        Psi0 = [ortho(randn(Complex{T}, length(k.basis), n_bands)) for k in basis.kpoints]
    end
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    ham = Hamiltonian(basis)

    ## vec and unpack
    # length of Psi[ik]
    lengths = [length(Psi0[ik]) for ik = 1:Nk]
    # psi[ik] is in psi_flat[starts[ik]:starts[ik]+lengths[ik]-1]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    pack(Psi) = vcat(Base.vec.(Psi)...) # TODO as an optimization, do that lazily? See LazyArrays
    unpack(Psi) = [@views reshape(Psi[starts[ik]:starts[ik]+lengths[ik]-1], size(Psi0[ik])) for ik in 1:Nk]

    # this will get updated
    energies = nothing

    # computes energies and gradients
    function fg!(E, G, Psi)
        Psi = unpack(Psi)
        ρ = compute_density(basis, Psi, occupation)
        ham = update_hamiltonian(ham, ρ)
        if E != nothing
            energies = update_energies(ham, Psi, occupation, ρ)
            E = sum(values(energies))
        end

        # The energy has terms like occ * <ψ|H|ψ>, so the gradient is 2occ Hψ
        if G != nothing
            G = unpack(G)
            for ik = 1:Nk
                mul!(G[ik], kblock(ham, basis.kpoints[ik]), Psi[ik])
                G[ik] .*= 2*filled_occ
            end
        end
        E
    end

    manif = DMManifold(Nk, unpack)

    Pks = [prec_type(ham, kpt) for kpt in basis.kpoints]
    P = DMPreconditioner(Nk, Pks, unpack)

    res = Optim.optimize(Optim.only_fg!(fg!), pack(Psi0),
                         optim_solver(P=P, precondprep=precondprep!, manifold=manif),
                         Optim.Options(; allow_f_increases=true, show_trace=true, kwargs...))
    Psi = unpack(res.minimizer)
    ρ = compute_density(basis, Psi, occupation)
    ham = update_hamiltonian(ham, ρ)
    # These concepts do not make sense in direct minimization,
    # although we could maybe do a final Rayleigh-Ritz
    orben = nothing
    εF = nothing
    (ham=ham, energies=energies, converged=true,
     ρ=ρ, Psi=Psi, orben=orben, occupation=occupation, εF=εF, optim_res=res)
end
