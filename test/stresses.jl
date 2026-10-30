# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual

@testitem "ForwardDiff stresses on silicon" tags=[:minimal] setup=[TestCases] begin
    using DFTK
    using DFTK: mpi_bcast
    using ForwardDiff
    import FiniteDiff
    using LinearAlgebra
    using Unitful
    using UnitfulAtomic
    silicon = TestCases.silicon

    function make_basis(model, lattice, symmetries)
        PlaneWaveBasis(Model(model; lattice, symmetries); Ecut=7, kgrid=(3, 3, 3))
    end

    function recompute_energy(model, lattice, symmetries)
        basis = make_basis(model, lattice, symmetries)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-13))
        (; energies) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation;
                                          scfres.ρ, scfres.hubbard_n)
        energies.total
    end

    function hellmann_feynman_energy(scfres, model, lattice, symmetries)
        basis = make_basis(model, lattice, symmetries)
        ρ = DFTK.compute_density(basis, scfres.ψ, scfres.occupation)
        hubbard_n = DFTK.compute_hubbard_n(basis, scfres.ψ, scfres.occupation)
        (; energies) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ, hubbard_n)
        energies.total
    end

    function test_stresses(model)
        lattice = model.lattice
        scfres = self_consistent_field(make_basis(model, lattice, true); tol=1e-11)
        scfres_nosym = self_consistent_field(make_basis(model, lattice, false); tol=1e-11)
        stresses = compute_stresses_cart(scfres)
        @test isapprox(stresses, compute_stresses_cart(scfres_nosym); atol=1e-10)

        dir = mpi_bcast(randn(3, 3), 0, scfres.basis.comm_kpts)

        dE_stresses = dot(dir, stresses) * scfres.basis.model.unit_cell_volume
        ref_recompute = FiniteDiff.finite_difference_derivative(0.0) do ε
            recompute_energy(model, lattice + ε*dir*lattice, false)
        end
        ref_scfres = FiniteDiff.finite_difference_derivative(0.0) do ε
            basis = make_basis(model, lattice + ε*dir*lattice, false)
            scfres = self_consistent_field(basis; tol=1e-13)
            scfres.energies.total
        end
        ref_HF = FiniteDiff.finite_difference_derivative(0.0) do ε
            hellmann_feynman_energy(scfres_nosym, model, lattice+ε*dir*lattice, false)
        end
        FD_HF = ForwardDiff.derivative(0.0) do ε
            hellmann_feynman_energy(scfres_nosym, model, lattice+ε*(dir*lattice), false)
        end

        @test isapprox(ref_recompute, ref_scfres; atol=1e-8)
        @test isapprox(ref_HF, ref_recompute; atol=1e-5)
        @test isapprox(ref_HF, FD_HF; atol=1e-5)
        @test isapprox(dE_stresses, ref_recompute; atol=1e-5)
        scfres
    end

    a = 10.0  # slightly compressed and twisted
    lattice = a / 2 * [[0 1 1.1];
                       [1 0 1.];
                       [1 1 0.]]

    # GTH pseudopotential (cannot apply Hubbard U because it has no pseudo-atomic orbitals)
    Si_gth = ElementPsp(silicon.atnum, load_psp(silicon.psp_gth))
    test_stresses(model_DFT(lattice, [Si_gth, Si_gth], silicon.positions; functionals=PBE()))

    # UPF pseudopotential (with Hubbard U)
    Si_upf = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))
    hubbard = Hubbard(OrbitalManifold(:Si, "3P") => 5u"eV")
    scfres = test_stresses(model_DFT(lattice, [Si_upf, Si_upf], silicon.positions;
                                     functionals=PBE(), extra_terms=[hubbard]))
    @test scfres.energies.Hubbard > 1e-3  # Check Hubbard term is nonzero
end
