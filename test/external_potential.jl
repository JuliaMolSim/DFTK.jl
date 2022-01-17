using DFTK
using Test

if mpi_nprocs() == 1  # Direct minimisation does not yet support MPI
@testset "Externel potential from Fourier coefficients" begin
    lattice = [[10 0 0.]; [0 0 0]; [0 0 0]]

    pot(G) = G == 0 ? zero(G) : 1 / abs(G)
    C = 1
    α = 2
    terms = [Kinetic(),
             ExternalFromFourier(G -> pot(G[1])),
             LocalNonlinearity(ρ -> C * ρ^α),
    ]
    model = Model(lattice; n_electrons=1, terms=terms, spin_polarization=:spinless)

    Ecut = 15
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    scfres_dm = direct_minimization(basis, tol=1e-10)
    scfres_scf = self_consistent_field(basis, tol=1e-10)
    @test abs(scfres_scf.energies.total - scfres_dm.energies.total) < 1e-6
end
end
