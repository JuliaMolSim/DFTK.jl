# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual

@testitem "ForwardDiff stresses on silicon" setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    import FiniteDiff
    using MPI
    using LinearAlgebra
    silicon = TestCases.silicon

    function make_basis(lattice, symmetries, element)
        model = model_PBE(lattice, [element, element], silicon.positions; symmetries)
        PlaneWaveBasis(model; Ecut=7, kgrid=(3, 3, 3))
    end

    function recompute_energy(lattice, symmetries, element)
        basis = make_basis(lattice, symmetries, element)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-13))
        (; energies) = energy_hamiltonian(scfres.ψ, scfres.occupation; ρ=scfres.ρ)
        energies.total
    end

    function hellmann_feynman_energy(scfres, lattice, symmetries, element)
        basis = make_basis(lattice, symmetries, element)
        ψ = BlochWaves(basis, denest(scfres.ψ))
        ρ = DFTK.compute_density(ψ, scfres.occupation)
        (; energies) = energy_hamiltonian(ψ, scfres.occupation; ρ)
        energies.total
    end

    function test_stresses(lattice, element)
        is_converged = DFTK.ScfConvergenceDensity(1e-11)
        scfres = self_consistent_field(make_basis(lattice, true, element); is_converged)
        scfres_nosym = self_consistent_field(make_basis(lattice, false, element); is_converged)
        stresses = compute_stresses_cart(scfres)
        @test isapprox(stresses, compute_stresses_cart(scfres_nosym), atol=1e-10)

        dir = MPI.bcast(randn(3, 3), 0, MPI.COMM_WORLD)

        dE_stresses = dot(dir, stresses) * scfres.basis.model.unit_cell_volume
        ref_recompute = FiniteDiff.finite_difference_derivative(0.0) do ε
            recompute_energy(lattice + ε*dir*lattice, false, element)
        end
        ref_scfres = FiniteDiff.finite_difference_derivative(0.0) do ε
            basis = make_basis(lattice + ε*dir*lattice, false, element)
            scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-13))
            scfres.energies.total
        end
        ref_HF = FiniteDiff.finite_difference_derivative(0.0) do ε
            hellmann_feynman_energy(scfres_nosym, lattice+ε*dir*lattice, false, element)
        end
        FD_HF = ForwardDiff.derivative(0.0) do ε
            hellmann_feynman_energy(scfres_nosym, lattice+ε*(dir*lattice), false, element)
        end

        @test isapprox(ref_recompute, ref_scfres, atol=1e-8)
        @test isapprox(ref_HF, ref_recompute, atol=1e-5)
        @test isapprox(ref_HF, FD_HF, atol=1e-5)
        @test isapprox(dE_stresses, ref_recompute, atol=1e-5)
    end

    a = 10.0  # slightly compressed and twisted
    lattice = a / 2 * [[0 1 1.1];
                       [1 0 1.];
                       [1 1 0.]]
    test_stresses(lattice, ElementPsp(silicon.atnum, :Si, load_psp(silicon.psp_hgh)))
    test_stresses(lattice, ElementPsp(silicon.atnum, :Si, load_psp(silicon.psp_upf)))
end
