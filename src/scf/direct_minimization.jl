using Optim
import Optim.project_tangent!
import Optim.retract!
import Optim.Manifold
import Optim.precondprep!
import LinearAlgebra.ldiv!
import LinearAlgebra.dot

struct MyManifold <: Optim.Manifold
    n_bands::Int
    Nk::Int
end
@views function Optim.project_tangent!(m::MyManifold, g, x)
    for ik = 1:m.Nk
        Optim.project_tangent!(Optim.Stiefel(),
                               g[:, (ik-1)*m.n_bands+1:ik*m.n_bands],
                               x[:, (ik-1)*m.n_bands+1:ik*m.n_bands])
    end
    g
end
@views function Optim.retract!(m::MyManifold, x)
    for ik = 1:m.Nk
        Optim.retract!(Optim.Stiefel(),
                       x[:, (ik-1)*m.n_bands+1:ik*m.n_bands])
    end
    x
end

struct MyPreconditioner
    n_bands::Int
    Nk::Int
    Pks # Pks[ik] is the preconditioner for kpoint k
end
@views function LinearAlgebra.ldiv!(p, m::MyPreconditioner, d)
    for ik = 1:m.Nk
        ldiv!(p[:, (ik-1)*m.n_bands+1:ik*m.n_bands],
              m.Pks[ik],
              d[:, (ik-1)*m.n_bands+1:ik*m.n_bands])
    end
    p
end
@views function LinearAlgebra.dot(x, m::MyPreconditioner, y)
    sum(dot(x[:, (ik-1)*m.n_bands+1:ik*m.n_bands],
            m.Pks[ik],
            y[:, (ik-1)*m.n_bands+1:ik*m.n_bands])
        for ik = 1:m.Nk)
end
@views function Optim.precondprep!(P::MyPreconditioner, x)
    for ik = 1:m.Nk
        precondprep!(Pk, x[:, (ik-1)*m.n_bands+1:ik*m.n_bands])
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

    vec(Psi) = hcat(Psi...) # TODO as an optimization, do that lazily? See LazyArrays
    devec(Psi) = [@views Psi[:, (ik-1)*n_bands+1:ik*n_bands] for ik in 1:Nk]

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
            HPsi = [kblock(ham, k) * Psi[ik] for (ik, k) in enumerate(basis.kpoints)]
            copy!(G, vec(HPsi))
        end
        E
    end

    Pks = [prec_type(ham, kpt) for kpt in basis.kpoints]
    P = MyPreconditioner(n_bands, Nk, Pks)
    manif = MyManifold(n_bands, Nk)
    Optim.optimize(Optim.only_fg!(fg!), vec(Psi0), Optim.LBFGS(P=P, manifold=manif), Optim.Options(show_trace=true))
end
