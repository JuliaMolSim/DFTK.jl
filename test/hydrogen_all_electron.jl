using Test
using DFTK
using LinearAlgebra

@testset "Hydrogen anion all electron" begin
    function energy_forces(;Ecut, tol)
        lattice = 10 * Matrix{Float64}(I, 3, 3)
        atoms = [ElementCoulomb(:H) => [zeros(3)]]
        model = model_DFT(lattice, atoms, [], n_electrons=2)
        basis = PlaneWaveBasis(model, Ecut, kgrid=(1, 1, 1))

        is_converged = DFTK.ScfConvergenceDensity(tol)
        scfres = self_consistent_field(basis, is_converged=is_converged) #, mixing=SimpleMixing())
        scfres.energies.total, DFTK.forces(scfres)
    end

    E, forces = energy_forces(Ecut=5, tol=1e-10)
    @test E ≈ -0.28310157350203013
    @test norm(forces) < 1e-9
end
