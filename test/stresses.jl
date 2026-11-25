# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual

@testmodule TestStresses begin
    using Test
    using DFTK
    using DFTK: mpi_bcast
    using ForwardDiff
    import FiniteDiff
    using LinearAlgebra
    using Unitful
    using UnitfulAtomic

    function hellmann_feynman_energy(scfres, model, lattice; symmetries, Ecut, kgrid)
        basis = PlaneWaveBasis(Model(model; lattice, symmetries); Ecut, kgrid)
        ρ = DFTK.compute_density(basis, scfres.ψ, scfres.occupation)
        hubbard_n = DFTK.compute_hubbard_n(basis, scfres.ψ, scfres.occupation)
        (; energies) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ, hubbard_n)
        energies.total
    end

    function test_stresses(model; Ecut=7, kgrid=(3, 3, 3), atol=1e-8, strain=nothing)
        lattice = model.lattice
        scfres = let
            basis = PlaneWaveBasis(Model(model; lattice, symmetries=true); Ecut, kgrid)
            self_consistent_field(basis; tol=1e-11)
        end
        scfres_nosym = let
            basis = PlaneWaveBasis(Model(model; lattice, symmetries=false); Ecut, kgrid)
            self_consistent_field(basis; tol=1e-11)
        end
        stresses = compute_stresses_cart(scfres)
        @test stresses ≈ compute_stresses_cart(scfres_nosym) atol=1e-10

        # A hard-coded strain direction may be passed, e.g. to respect orthogonality
        # constraints of extra terms such as the dipole correction.
        strain = @something strain mpi_bcast(randn(3, 3), 0, scfres.basis.comm_kpts)

        dE_stresses = dot(strain, stresses) * scfres.basis.model.unit_cell_volume
        ref_recompute = FiniteDiff.finite_difference_derivative(0.0) do ε
            new_lattice = lattice + ε*strain*lattice
            basis = PlaneWaveBasis(Model(model; lattice=new_lattice, symmetries=false); Ecut, kgrid)
            scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-13))
            (; energies) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation;
                                              scfres.ρ, scfres.hubbard_n)
            energies.total
        end
        ref_scfres = FiniteDiff.finite_difference_derivative(0.0) do ε
            new_lattice = lattice + ε*strain*lattice
            basis = PlaneWaveBasis(Model(model; lattice=new_lattice, symmetries=false); Ecut, kgrid)
            scfres = self_consistent_field(basis; tol=1e-13)
            scfres.energies.total
        end
        ref_HF = FiniteDiff.finite_difference_derivative(0.0) do ε
            new_lattice = lattice + ε*strain*lattice
            hellmann_feynman_energy(scfres_nosym, model, new_lattice; symmetries=false, Ecut, kgrid)
        end
        FD_HF = ForwardDiff.derivative(0.0) do ε
            new_lattice = lattice + ε*strain*lattice
            hellmann_feynman_energy(scfres_nosym, model, new_lattice; symmetries=false, Ecut, kgrid)
        end

        @test ref_recompute ≈ ref_scfres     atol=atol
        @test ref_HF        ≈ ref_recompute  atol=1e-5
        @test ref_HF        ≈ FD_HF          atol=1e-5
        @test dE_stresses   ≈ ref_recompute  atol=1e-5
        scfres
    end
end

@testitem "Stresses on silicon" tags=[:minimal, :stresses] setup=[TestCases,TestStresses] begin
    using DFTK
    using Unitful
    silicon = TestCases.silicon
    test_stresses = TestStresses.test_stresses

    a = 10.0  # slightly compressed and twisted
    lattice = a / 2 * [[0 1 1.1];
                       [1 0 1.];
                       [1 1 0.]]

    @testset "PBE" begin
        # GTH pseudopotential (cannot apply Hubbard U because it has no pseudo-atomic orbitals)
        Si_gth = ElementPsp(silicon.atnum, load_psp(silicon.psp_gth))
        test_stresses(model_DFT(lattice, [Si_gth, Si_gth], silicon.positions; functionals=PBE()))
    end

    @testset "LDA+U" begin
        # UPF pseudopotential (with Hubbard U)
        Si_upf = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))
        hubbard = Hubbard(OrbitalManifold(:Si, "3P") => 5u"eV")
        scfres = test_stresses(model_DFT(lattice, [Si_upf, Si_upf], silicon.positions;
                                         functionals=PBE(), extra_terms=[hubbard]))
        @test scfres.energies.Hubbard > 1e-3  # Check Hubbard term is nonzero
    end
end

@testitem "Stresses with Bengtsson dipole correction" setup=[TestStresses] tags=[:stresses, :dont_test_mpi] begin
    using DFTK
    using Test
    using PseudoPotentialData
    using LinearAlgebra
    test_stresses = TestStresses.test_stresses

    Li = ElementCoulomb(:Li)
    lattice   = diagm([6.0, 7.0, 24.0])
    positions = [[0.5, 0.5, 0.30], [0.5, 0.5, 0.50]]
    model     = model_DFT(lattice, [Li, Li], positions; functionals=LDA(),
                          extra_terms=[DipoleCorrection()], symmetries=false)
    # Strain, which scales the normal lengths and strains within the lateral plane
    # of the dipole correction, but does not tilt between them.
    strain = [0.15 -0.2 0.0; 0.1 0.3 0.0; 0.0 0.0 -0.25]
    test_stresses(model; Ecut=8, kgrid=[1, 1, 1], atol=5e-8, strain)
end
