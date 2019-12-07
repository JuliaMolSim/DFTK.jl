# Direct minimization of the energy

using Optim
import Optim.project_tangent!
import Optim.retract!
import Optim.Manifold
import Optim.precondprep!
import LinearAlgebra.ldiv!
import LinearAlgebra.dot

# This is all a bit annoying because our Psi is represented as Psi[k][G,n], and Optim accepts only dense arrays
# We do a bit of back and forth using custom `vec` (ours -> optim's) and `devec` (optim's -> ours) functions

# Impose the orthogonality constraint on orbitals inside each k-block
struct DMManifold <: Optim.Manifold
    Nk::Int
    devec_fun::Function
end
function Optim.project_tangent!(m::DMManifold, g, x)
    g_devec = m.devec_fun(g)
    x_devec = m.devec_fun(x)
    for ik = 1:m.Nk
        Optim.project_tangent!(Optim.Stiefel(),
                               g_devec[ik],
                               x_devec[ik])
    end
    g
end
function Optim.retract!(m::DMManifold, x)
    x_devec = m.devec_fun(x)
    for ik = 1:m.Nk
        Optim.retract!(Optim.Stiefel(), x_devec[ik])
    end
    x
end

# Array of preconditioners
struct DMPreconditioner
    Nk::Int
    Pks::Array # Pks[ik] is the preconditioner for kpoint ik
    devec_fun::Function
end
function LinearAlgebra.ldiv!(p, P::DMPreconditioner, d)
    p_devec = P.devec_fun(p)
    d_devec = P.devec_fun(d)
    for ik = 1:P.Nk
        ldiv!(p_devec[ik], P.Pks[ik], d_devec[ik])
    end
    p
end
function LinearAlgebra.dot(x, P::DMPreconditioner, y)
    x_devec = P.devec(x)
    y_devec = P.devec(y)
    sum(dot(x_devec[ik], P.Pks[ik], y_devec[ik])
        for ik = 1:P.Nk)
end
function Optim.precondprep!(P::DMPreconditioner, x)
    x_devec = P.devec
    for ik = 1:P.Nk
        precondprep!(Pks[ik], x_devec[ik])
    end
    P
end


direct_minimization(basis::PlaneWaveBasis) = direct_minimization(basis, nothing)
function direct_minimization(basis::PlaneWaveBasis{T}, Psi0;
                             prec_type=PreconditionerTPA) where T
    model = basis.model
    @assert model.spin_polarisation in (:none, :spinless)
    @assert model.assume_band_gap
    filled_occ = filled_occupation(model)
    n_bands = div(model.n_electrons, filled_occ)
    ortho(Psik) = Matrix{Complex{T}}(Matrix(qr(Psik).Q))
    Nk = length(basis.kpoints)

    if Psi0 === nothing
        Psi0 = [ortho(randn(Complex{T}, length(k.basis), n_bands)) for k in basis.kpoints]
    end
    occupations = [filled_occ*ones(T, n_bands) for ik = 1:Nk]

    ham = Hamiltonian(basis)

    ## vec and devec
    # length of Psi[ik]
    lengths = [length(Psi0[ik]) for ik = 1:Nk]
    # psi[ik] is in psi_flat[starts[ik]:starts[ik]+lengths[ik]-1]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    vec(Psi) = vcat(Base.vec.(Psi)...) # TODO as an optimization, do that lazily? See LazyArrays
    devec(Psi) = [@views reshape(Psi[starts[ik]:starts[ik]+lengths[ik]-1], size(Psi0[ik])) for ik in 1:Nk]

    # computes energies and gradients
    function fg!(E, G, Psi)
        Psi = devec(Psi)
        ρ = compute_density(basis, Psi, occupations)
        ham = update_hamiltonian(ham, ρ)
        if E != nothing
            energies = update_energies(ham, Psi, occupations, ρ)
            E = sum(values(energies))
        end

        # TODO check for possible 1/2 factors
        if G != nothing
            G = devec(G)
            for ik = 1:Nk
                mul!(G[ik], kblock(ham, basis.kpoints[ik]), Psi[ik])
            end
        end
        E
    end

    manif = DMManifold(Nk, devec)

    Pks = [prec_type(ham, kpt) for kpt in basis.kpoints]
    P = DMPreconditioner(Nk, Pks, devec)

    Optim.optimize(Optim.only_fg!(fg!), vec(Psi0), Optim.LBFGS(P=P, manifold=manif), Optim.Options(show_trace=true))
end
