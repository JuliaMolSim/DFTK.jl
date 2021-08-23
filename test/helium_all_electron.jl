using Test
using DFTK
using LinearAlgebra

@testset "Helium all electron" begin
    function energy_forces(;Ecut, tol)
        lattice = 10 * Matrix{Float64}(I, 3, 3)
        atoms = [ElementCoulomb(:He) => [zeros(3)]]
        model = model_DFT(lattice, atoms, [], n_electrons=2)
        basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))

        is_converged = DFTK.ScfConvergenceDensity(tol)
        scfres = self_consistent_field(basis, is_converged=is_converged)
        scfres.energies.total, DFTK.compute_forces(scfres)
    end

    E, forces = energy_forces(Ecut=5, tol=1e-10)
    @test E ≈ -1.5869009433016852
    @test norm(forces) < 1e-9
end
