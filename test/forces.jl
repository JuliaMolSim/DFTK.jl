@testmodule TestForces begin
    using Test
    using DFTK
    using DFTK: mpi_mean!
    using MPI
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic

    function compute_energy(system, dx;
            functionals=PBE(), terms=nothing, Ecut, kgrid, temperature=0,
            smearing=Smearing.Gaussian(),
            pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"),
            symmetries=true, basis_kwargs...)
        particles = map(system, position(system, :) + dx) do atom, pos
            Atom(atom; position=pos)
        end
        sysmod = AbstractSystem(system; particles)

        if isnothing(terms)
            model = model_DFT(sysmod; functionals, pseudopotentials, symmetries,
                              temperature, smearing)
        else
            model = Model(sysmod; terms, pseudopotentials, symmetries,
                          temperature, smearing)
        end
        basis = PlaneWaveBasis(model; kgrid, Ecut, basis_kwargs...)

        self_consistent_field(basis; tol=1e-12)
    end

    function test_forces(system; testatoms=1:length(system), ε=1e-5, atol=1e-8, kwargs...)
        particles = [Atom(; pairs(atom)...) for atom in system]
        system = AbstractSystem(system; particles)

        scfres = compute_energy(system, zeros(length(system)))
        forces = compute_forces_cart(scfres)

        for i in 1:testatoms
            dx = [zeros(3) * u"Å" for _ in 1:length(system)]
            dx[i]  = rand((3, )) * u"Å"
            dx[i]  = [0.1, 0.02, 0.1] * u"Å"  # avoid random for testing
            mpi_mean!(dx, MPI.COMM_WORLD)  # must be identical on all processes

            Fε_ref = sum(map(forces, dx) do Fi, dxi
                -dot(Fi, austrip.(dxi))
            end)

            Fε = let
                (  compute_energy(dx,  ε).energies.total
                 - compute_energy(dx, -ε).energies.total) / 2ε
            end

            @show Fε abs(Fε_ref - Fε)
            @test abs(Fε_ref - Fε) < atol
        end

        (; forces_cart=forces)
    end
end

@testitem "Forces silicon with non-linear core correction" setup=[TestCases,TestForces] begin
    using DFTK
    using PseudoPotentialData

    silicon = TestCases.silicon
    test_forces = TestForces.test_forces

    positions = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    system = atomic_system(silicon.lattice, silicon.atoms, positions)

    pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    (; forces_cart) = test_forces(system; functionals=LDA(), tol=1e-7, pseudopotentials,
                                  testatoms=1:1, Ecut=7, kgrid=[2, 2, 2], kshift=[0, 0, 0],
                                  symmetries_respect_rgrid=true,
                                  fft_size=(18, 18, 18))  # FFT chosen to match QE

    # Test against Abinit v9.6.2 using LibXC v4.3.2 lda_x+lda_c_pw
    # (see testcases_ABINIT/silicon_NLCC_forces)
    reference = [[-0.00574838157984, -0.00455216015517, -0.00333786048065],
                 [ 0.00574838157984,  0.00455216015517,  0.00333786048065]]
    @test maximum(v -> maximum(abs, v), reference - Fc1) < 1e-5
end

@testitem "Forces on silicon with spin and temperature" setup=[TestCases,TestForces] begin
    using DFTK
    using PseudoPotentialData
    silicon = TestCases.silicon
    test_forces = TestForces.test_forces

    positions = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    system = atomic_system(silicon.lattice, silicon.atoms, positions)

    pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    for (tol, smearing) in [(0.003, Smearing.FermiDirac()), (5e-5, Smearing.Gaussian())]
        test_forces(system; pseudopotentials, functionals=Xc(:lda_xc_teter93),
                    temperature=0.03, smearing, ε=1e-6, tol,
                    testatoms=1:1, Ecut=7, kgrid=[4, 1, 2], kshift=[1/2, 0, 0])
    end
end

@testitem "Iron with spin and temperature"  setup=[TestForces] begin
    using DFTK
    using AtomsBuilder
    using PseudoPotentialData
    test_forces = TestForces.test_forces

    system = bulk(:Fe, cubic=true)
    rattle!(system, 1e-3u"Å")
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    test_forces(system; pseudopotentials, functionals=PBE(),
                temperature=1e-3, ε=1e-6, tol=1e-6,
                testatoms=1:1, Ecut=10, kgrid=[8, 8, 8], kshift=[0, 0, 0])
end

@testset "Rutile without non-local" begin
    system = load_system("SnO2(1).cif")
    rattle!(system, 1e-1u"Å")
    terms = [Kinetic(), AtomicLocal(), PspCorrection(), Entropy(), Ewald() ]
    test_forces(system; kgrid=[1, 1, 1], Ecut=20, ε=1e-5, atol=1e-8, terms)
end


@testset "Rutile PBE" begin
    system = load_system("structures/SnO2.cif")
    rattle!(system, 1e-1u"Å")
    test_forces(system; kgrid=[1, 1, 1], Ecut=20, ε=1e-5, atol=1e-8)
end

@testset "Rutile PBE (GTH)" begin
    using PseudoPotentialData

    system = load_system("structures/SnO2.cif")
    rattle!(system, 1e-1u"Å")
    pseudopotentials = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
    test_forces(system; kgrid=[1, 1, 1], Ecut=20, ε=1e-5, atol=1e-8, pseudopotentials)
end

@testset "Rutile PBE full" begin
    system = load_system("structures/GeO2_distorted.extxyz")
    test_forces(system; kgrid=[6, 6, 9], Ecut=40,
                        symmetries=false)
end


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


@testitem "Forces silicon with non-linear core correction" setup=[TestCases] begin
    using DFTK
    using DFTK: mpi_mean!
    using MPI
    using LinearAlgebra
    silicon = TestCases.silicon

    function energy_forces(positions)
        Si = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))
        atoms = fill(Si, length(silicon.atoms))
        model = model_DFT(silicon.lattice, atoms, positions; functionals=LDA())
        basis = PlaneWaveBasis(model; Ecut=7, kgrid=[2, 2, 2], kshift=[0, 0, 0],
                               symmetries_respect_rgrid=true,
                               fft_size=(18, 18, 18))  # FFT chosen to match QE
        is_converged = DFTK.ScfConvergenceDensity(1e-11)
        scfres = self_consistent_field(basis; is_converged)
        scfres.energies.total, compute_forces(scfres), compute_forces_cart(scfres)
    end

    # symmetrical positions, forces should be 0
    _, F0, _ = energy_forces([(ones(3)) / 8, -ones(3) / 8])
    @test norm(F0) < 1e-4

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    disp = rand(3)
    mpi_mean!(disp, MPI.COMM_WORLD)  # must be identical on all processes
    ε = 1e-5
    pos2 = [pos1[1] + ε * disp, pos1[2]]
    pos3 = [pos1[1] - ε * disp, pos1[2]]

    # second-order finite differences for accurate comparison
    # TODO switch the other tests to this too
    E1, F1, Fc1 = energy_forces(pos1)
    E2,  _,  _  = energy_forces(pos2)
    E3,  _,  _  = energy_forces(pos3)

    diff_findiff = -(E2 - E3) / (2ε)
    diff_forces = dot(F1[1], disp)
    @test abs(diff_findiff - diff_forces) < 1e-7

    # Test against Abinit v9.6.2 using LibXC v4.3.2 lda_x+lda_c_pw
    # (see testcases_ABINIT/silicon_NLCC_forces)
    reference = [[-0.00574838157984, -0.00455216015517, -0.00333786048065],
                 [ 0.00574838157984,  0.00455216015517,  0.00333786048065]]
    @test maximum(v -> maximum(abs, v), reference - Fc1) < 1e-5
end

@testitem "Forces on silicon with spin and temperature" setup=[TestCases] begin
    using DFTK
    using DFTK: mpi_mean!
    using MPI
    using LinearAlgebra
    silicon = TestCases.silicon

    function silicon_energy_forces(positions; smearing=Smearing.FermiDirac())
        model  = model_DFT(silicon.lattice, silicon.atoms, positions;
                           functionals=[:lda_xc_teter93], temperature=0.03,
                           smearing, spin_polarization=:collinear)
        basis  = PlaneWaveBasis(model; Ecut=4, kgrid=[4, 1, 2], kshift=[1/2, 0, 0])
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(5e-10))
        scfres.energies.total, compute_forces(scfres)
    end

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    disp = rand(3)
    mpi_mean!(disp, MPI.COMM_WORLD)  # must be identical on all processes
    ε = 1e-6
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    for (tol, smearing) in [(0.003, Smearing.FermiDirac()), (5e-5, Smearing.Gaussian())]
        E1, F1 = silicon_energy_forces(pos1; smearing)
        E2, _  = silicon_energy_forces(pos2; smearing)

        diff_findiff = -(E2 - E1) / ε
        diff_forces  = dot(F1[1], disp)
        @test abs(diff_findiff - diff_forces) < tol
    end
end

@testitem "Forces on oxygen with spin and temperature" setup=[TestCases] tags=[:dont_test_mpi] begin
    using DFTK
    using DFTK: mpi_mean!
    using MPI
    using LinearAlgebra
    o2molecule = TestCases.o2molecule

    function oxygen_energy_forces(positions)
        magnetic_moments = [1.0, 1.0]
        model = model_DFT(diagm([7.0, 7.0, 7.0]), o2molecule.atoms, positions;
                          functionals=PBE(), temperature=0.02,
                          smearing=Smearing.Gaussian(), magnetic_moments)
        basis = PlaneWaveBasis(model; Ecut=4, kgrid=[1, 1, 1])

        scfres = self_consistent_field(basis;
                                       is_converged=DFTK.ScfConvergenceDensity(1e-7),
                                       ρ=guess_density(basis, magnetic_moments),
                                       damping=0.7, mixing=SimpleMixing())
        scfres.energies.total, compute_forces(scfres)
    end

    pos1 = [[0, 0, 0.1155], [0.01, -2e-3, -0.2]]
    disp = rand(3)
    mpi_mean!(disp, MPI.COMM_WORLD)  # must be identical on all processes
    ε = 1e-6
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    E1, F1 = oxygen_energy_forces(pos1)
    E2, _  = oxygen_energy_forces(pos2)

    diff_findiff = -(E2 - E1) / ε
    diff_forces  = dot(F1[1], disp)

    @test abs(diff_findiff - diff_forces) < 5e-4
end

@testitem "Forces match partial derivative of each term" setup=[TestCases] begin
    using AtomsIO
    using DFTK: mpi_mean!
    using LinearAlgebra
    using MPI

    function get_term_forces(system; kgrid=[1,1,1], Ecut=4, symmetries=false, ε=1e-8)
        model = model_DFT(system; pseudopotentials=TestCases.gth_lda_semi,
                                  functionals=LDA(), symmetries, temperature=1e-3)
        basis = PlaneWaveBasis(model; kgrid, Ecut)

        scfres = self_consistent_field(basis; tol=1e-7)

        map(1:length(basis.terms)) do iterm
            # must be identical on all processes
            test_atom = MPI.Bcast(rand(1:length(model.atoms)), 0, MPI.COMM_WORLD)
            test_dir = rand(3)
            mpi_mean!(test_dir, MPI.COMM_WORLD)

            forces_HF = DFTK.compute_forces(basis.terms[iterm], basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
            force_HF = isnothing(forces_HF) ? 0 : dot(test_dir, forces_HF[test_atom])

            function term_energy(ε)
                displacement = [[0.0, 0.0, 0.0] for _ in 1:length(model.atoms)]
                displacement[test_atom] = test_dir
                modmodel = Model(model; positions=model.positions .+ ε.*displacement)
                basis = PlaneWaveBasis(modmodel; kgrid, Ecut)
                DFTK.ene_ops(basis.terms[iterm], basis, scfres.ψ, scfres.occupation;
                             ρ=scfres.ρ, εF=scfres.εF, eigenvalues=scfres.eigenvalues).E
            end

            e1 = term_energy(ε)
            e2 = term_energy(-ε)
            force_ε = -(e1 - e2) / 2ε

            (; force_HF, force_ε)
        end
    end

    system = load_system("structures/tio2_stretched.extxyz")
    terms_forces = get_term_forces(system)
    for term_forces in terms_forces
        @test abs(term_forces.force_HF - term_forces.force_ε) < 2e-5
    end
end
