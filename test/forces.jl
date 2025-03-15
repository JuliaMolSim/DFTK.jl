@testmodule TestForces begin
    using Test
    using AtomsBase
    using DFTK
    using MPI
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    using LinearAlgebra

    function test_term_forces(system; ε=1e-5, atol=1e-8,
        functionals=PBE(), Ecut, kgrid, temperature=1e-3, smearing=Smearing.Gaussian(),
        pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"),
        magnetic_moments=[], symmetries=true, basis_kwargs...)

        model = model_DFT(system; functionals, pseudopotentials, symmetries,
                          temperature, smearing, magnetic_moments)
        basis = PlaneWaveBasis(model; kgrid, Ecut, basis_kwargs...)
        scfres = self_consistent_field(basis; tol=1e-10)

        for iterm in 1:length(basis.terms)
            @testset "$(typeof(model.term_types[iterm]))" begin
                # must be identical on all processes
                test_atom = MPI.Bcast(rand(1:length(model.atoms)), 0, MPI.COMM_WORLD)
                test_dir  = MPI.Bcast(rand(3), 0, MPI.COMM_WORLD)

                forces_HF = DFTK.compute_forces(basis.terms[iterm], basis,
                                                scfres.ψ, scfres.occupation; scfres.ρ, scfres.τ)
                force_HF  = isnothing(forces_HF) ? 0.0 : dot(test_dir, forces_HF[test_atom])

                function term_energy(ε)
                    displacement = [zeros(3) for _ in 1:length(model.atoms)]
                    displacement[test_atom] = test_dir
                    modmodel = Model(model; positions=model.positions .+ ε.*displacement)
                    basis = PlaneWaveBasis(modmodel; kgrid, Ecut)
                    DFTK.ene_ops(basis.terms[iterm], basis, scfres.ψ, scfres.occupation;
                                 scfres.ρ, scfres.εF, scfres.τ, scfres.eigenvalues).E
                end

                force_ε = -( (term_energy(ε) - term_energy(-ε)) / 2ε )
                @show force_ε force_HF abs(term_forces.force_HF - term_forces.force_ε)
                @test abs(term_forces.force_HF - term_forces.force_ε) < atol
            end
        end
    end

    function compute_energy(system, dx;
            functionals=PBE(), Ecut, kgrid, temperature=0,
            smearing=Smearing.Gaussian(),
            pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"),
            magnetic_moments=[], symmetries=true, basis_kwargs...)
        particles = map(system, position(system, :) + dx) do atom, pos
            Atom(atom; position=pos)
        end
        sysmod = AbstractSystem(system; particles)

        model = model_DFT(sysmod; functionals, pseudopotentials, symmetries,
                          temperature, smearing, magnetic_moments)
        basis = PlaneWaveBasis(model; kgrid, Ecut, basis_kwargs...)

        ρ = guess_density(basis, magnetic_moments)
        self_consistent_field(basis; ρ, tol=1e-12)
    end

    function test_forces(system; ε=1e-5, atol=1e-8, δx=nothing,
                         iatom=rand(1:length(system)),
                         kwargs...)
        particles = [Atom(; pairs(atom)...) for atom in system]
        system = AbstractSystem(system; particles)

        scfres = compute_energy(system, [zeros(3)u"Å" for _ in 1:length(system)]; kwargs...)
        forces = compute_forces_cart(scfres)

        dx = [zeros(3) * u"Å" for _ in 1:length(system)]
        δx = @something δx rand(3)
        δx    = MPI.Bcast(δx, 0, MPI.COMM_WORLD)
        iatom = MPI.Bcast(iatom, 0, MPI.COMM_WORLD)
        dx[iatom]  = δx * u"Å"

        Fε_ref = sum(map(forces, dx) do Fi, dxi
            -dot(Fi, austrip.(dxi))
        end)

        Fε = let
            (  compute_energy(system,  ε * dx; kwargs...).energies.total
             - compute_energy(system, -ε * dx; kwargs...).energies.total) / 2ε
        end

        @test abs(Fε_ref - Fε) < atol
        (; forces_cart=forces)
    end
end

@testitem "Forces match partial derivative of each term" setup=[TestCases] begin
    using AtomsIO
    using LinearAlgebra
    using MPI

    system = load_system("structures/tio2_stretched.extxyz")
    test_term_forces(system, Ecut=10, kgrid=(2,2,2), atol=1e-6)
end

@testitem "Forces silicon" setup=[TestCases,TestForces] begin
    using DFTK
    using PseudoPotentialData

    silicon = TestCases.silicon
    test_forces = TestForces.test_forces

    positions = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    system = atomic_system(silicon.lattice, silicon.atoms, positions)

    pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    (; forces_cart) = test_forces(system; functionals=LDA(), atol=1e-8, pseudopotentials,
                                  Ecut=7, kgrid=[2, 2, 2], kshift=[0, 0, 0],
                                  symmetries_respect_rgrid=true,
                                  fft_size=(18, 18, 18))  # FFT chosen to match QE

    # Test against Abinit v9.6.2 using LibXC v4.3.2 lda_x+lda_c_pw
    # (see testcases_ABINIT/silicon_NLCC_forces)
    reference = [[-0.00574838157984, -0.00455216015517, -0.00333786048065],
                 [ 0.00574838157984,  0.00455216015517,  0.00333786048065]]
    @test maximum(v -> maximum(abs, v), reference - forces_cart) < 1e-5
end

@testitem "Forces on silicon with spin and temperature" setup=[TestCases,TestForces] begin
    using DFTK
    using PseudoPotentialData
    silicon = TestCases.silicon
    test_forces = TestForces.test_forces

    positions = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    system = atomic_system(silicon.lattice, silicon.atoms, positions)

    pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    for smearing in [Smearing.FermiDirac(), Smearing.Gaussian()]
        test_forces(system; pseudopotentials, functionals=Xc(:lda_xc_teter93),
                    temperature=0.03, smearing, atol=5e-6, magnetic_moments=[2.0, 1.0],
                    Ecut=7, kgrid=[4, 1, 2], kshift=[1/2, 0, 0])
    end
end


# TODO TiO2 test may be cheaper than the Rutile PBE ?
@testitem "Rutile PBE"  setup=[TestForces]  begin
    using DFTK
    using AtomsBuilder
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    using AtomsIO
    test_forces = TestForces.test_forces

    system = load_system("structures/GeO2.cif")
    rattle!(system, 0.02u"Å")
    iatom = 2
    δx = [0.482, 0.105, 0.452]
    test_forces(system; kgrid=[2, 2, 3], Ecut=20, atol=1e-7, iatom, δx)
end

@testitem "Rutile PBE full"  setup=[TestForces] tags=[:slow] begin
    using DFTK
    using AtomsBuilder
    using PseudoPotentialData
    using AtomsIO
    test_forces = TestForces.test_forces

    system = load_system("structures/GeO2.cif")
    rattle!(system, 0.02u"Å")
    iatom = 2
    δx = [0.482, 0.105, 0.452]
    test_forces(system; kgrid=[6, 6, 9], Ecut=30, atol=1e-6, iatom, δx)
end

@testitem "Iron with spin and temperature"  setup=[TestForces] tags=[:slow] begin
    using DFTK
    using AtomsBuilder
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    test_forces = TestForces.test_forces

    system = bulk(:Fe, cubic=true)
    rattle!(system, 0.01u"Å")
    pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
    test_forces(system; pseudopotentials, functionals=LDA(), temperature=1e-3,
                Ecut=13, kgrid=[6, 6, 6], atol=1e-6, magnetic_moments=[5.0, 5.0])
end
