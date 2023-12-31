@testitem "External potential from Fourier coefficients" #=
    =#    tags=[:core, :dont_test_mpi] begin
    using DFTK

    lattice = [[10 0 0.]; [0 0 0]; [0 0 0]]

    pot(G) = G == 0 ? zero(G) : 1 / abs(G)
    C = 1
    α = 2
    terms = [Kinetic(),
             ExternalFromFourier(G -> pot(G[1])),
             LocalNonlinearity(ρ -> C * ρ^α)]
    model = Model(lattice; n_electrons=1, terms, spin_polarization=:spinless)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=(1, 1, 1))
    scfres_dm  = direct_minimization(basis; tol=1e-10)
    scfres_scf = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceEnergy(1e-10))
    @test abs(scfres_scf.energies.total - scfres_dm.energies.total) < 1e-6
end
