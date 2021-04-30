using DFTK
using Test

@testset "Externel potential from Fourier coefficients" begin
    pot(G) = G == 0 ? zero(G) : 1 / abs(G)

    a = 10
    lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

    C = 1
    α = 2

    n_electrons = 1
    terms = [Kinetic(),
             ExternalFromFourier(G -> pot(G[1])),
             PowerNonlinearity(C, α),
    ]
    model = Model(lattice; n_electrons=n_electrons, terms=terms,
                  spin_polarization=:spinless)  # use "spinless electrons"

    Ecut = 15
    basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))
    scfres_dm = direct_minimization(basis, tol=1e-10)
    scfres_scf = self_consistent_field(basis, tol=1e-10)
    @test norm(scfres_scf.energies.total - scfres_dm.energies.total) < 1e-6
end
