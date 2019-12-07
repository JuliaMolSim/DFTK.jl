using Optim
import Optim.project_tangent!
import Optim.retract!
import Optim.Manifold

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

direct_minimization(basis::PlaneWaveBasis) = direct_minimization(basis, nothing)
function direct_minimization(basis::PlaneWaveBasis{T}, Psi0) where T
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

    Optim.optimize(Optim.only_fg!(fg!), vec(Psi0), Optim.LBFGS(manifold=MyManifold(n_bands, Nk)), Optim.Options(show_trace=true))
end
