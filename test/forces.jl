@testmodule TestForces begin
    using Logging
    using Test
    using AtomsBase
    using DFTK
    using MPI
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    using LinearAlgebra

    function test_term_forces(system; ε=1e-6, atol=1e-8,
        functionals=PBE(), Ecut, kgrid, temperature=0, smearing=Smearing.Gaussian(),
        pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"),
        magnetic_moments=[], symmetries=true, mixing=HybridMixing(), basis_kwargs...)

        model = model_DFT(system; functionals, pseudopotentials, symmetries,
                          temperature, smearing, magnetic_moments)
        basis = PlaneWaveBasis(model; kgrid, Ecut, basis_kwargs...)
        ρ = guess_density(basis, magnetic_moments)
        scfres = self_consistent_field(basis; ρ, tol=1e-12, mixing)

        # must be identical on all processes
        test_atom = MPI.bcast(rand(1:length(model.atoms)), 0, MPI.COMM_WORLD)
        test_dir  = MPI.bcast(rand(3), 0, MPI.COMM_WORLD)
        normalize!(test_dir)

        for iterm in 1:length(basis.terms)
            term_type = model.term_types[iterm]
            @testset "$(typeof(term_type))" begin
                forces_HF = DFTK.compute_forces(basis.terms[iterm], basis,
                                                scfres.ψ, scfres.occupation; scfres.ρ, scfres.τ)
                force_HF  = isnothing(forces_HF) ? 0.0 : dot(test_dir, forces_HF[test_atom])

                function term_energy(ε)
                    displacement = [zeros(3) for _ in 1:length(model.atoms)]
                    displacement[test_atom] = test_dir
                    modbasis = with_logger(NullLogger()) do
                        modmodel = Model(model; positions=model.positions .+ ε.*displacement)
                        PlaneWaveBasis(modmodel; kgrid, Ecut, basis_kwargs...)
                    end
                    DFTK.ene_ops(modbasis.terms[iterm], modbasis, scfres.ψ, scfres.occupation;
                                 scfres.ρ, scfres.εF, scfres.τ, scfres.eigenvalues).E
                end

                force_ε = -( (term_energy(ε) - term_energy(-ε)) / 2ε )
                @test abs(force_HF - force_ε) < atol
            end
        end
    end

    function compute_energy(system, dx;
            functionals=PBE(), Ecut, kgrid, temperature=0,
            smearing=Smearing.Gaussian(),
            pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"),
            magnetic_moments=[], symmetries=true, ρ=nothing,
            mixing=HybridMixing(), basis_kwargs...)
        particles = map(system, position(system, :) + dx) do atom, pos
            Atom(atom; position=pos)
        end
        sysmod = AbstractSystem(system; particles)

        model = model_DFT(sysmod; functionals, pseudopotentials, symmetries,
                          temperature, smearing, magnetic_moments)
        basis = PlaneWaveBasis(model; kgrid, Ecut, basis_kwargs...)

        ρ = @something ρ guess_density(basis, magnetic_moments)
        self_consistent_field(basis; ρ, tol=1e-12, mixing)
    end

    function test_forces(system; ε=1e-5, atol=1e-8, δx=nothing,
                         forward_ρ=true, iatom=rand(1:length(system)),
                         kwargs...)
        particles = [Atom(; pairs(atom)...) for atom in system]
        system = AbstractSystem(system; particles)

        scfres = compute_energy(system, [zeros(3)u"Å" for _ in 1:length(system)]; kwargs...)
        forces = compute_forces_cart(scfres)

        dx = [zeros(3) * u"Å" for _ in 1:length(system)]
        δx = @something δx rand(3)
        δx    = MPI.bcast(δx, 0, MPI.COMM_WORLD)
        normalize!(δx)
        iatom = MPI.bcast(iatom, 0, MPI.COMM_WORLD)
        dx[iatom]  = δx * u"Å"

        Fε_ref = sum(map(forces, dx) do Fi, dxi
            -dot(Fi, austrip.(dxi))
        end)

        Fε = let
            ρ = forward_ρ ? scfres.ρ : nothing
            (  compute_energy(system,  ε * dx; ρ, kwargs...).energies.total
             - compute_energy(system, -ε * dx; ρ, kwargs...).energies.total) / 2ε
        end

        @test abs(Fε_ref - Fε) < atol
        (; forces_cart=forces)
    end
end

@testitem "Forces term-wise TiO2 (GTH)" setup=[TestForces] tags=[:forces] begin
    # Test HF forces on non-symmetric multi-species structure using analytical pseudos
    using AtomsIO
    using PseudoPotentialData
    system = load_system("structures/tio2_stretched.extxyz")
    pseudopotentials = PseudoFamily("cp2k.nc.sr.pbe.v0_1.largecore.gth")
    TestForces.test_term_forces(system; Ecut=15, kgrid=(2,2,3), temperature=1e-4, pseudopotentials,
                                mixing=DielectricMixing(εr=10))
end

@testitem "Forces term-wise TiO2 (UPF)" setup=[TestForces] tags=[:forces] begin
    # Test HF forces on non-symmetric multi-species structure with NLCC
    using AtomsIO
    using PseudoPotentialData
    system = load_system("structures/tio2_stretched.extxyz")
    TestForces.test_term_forces(system; Ecut=25, kgrid=(2,2,3), temperature=1e-4,
                                mixing=DielectricMixing(εr=10))
end

@testitem "Forces term-wise Fe (GTH)"  setup=[TestForces] tags=[:forces] begin
    # TODO: If this test is too slow for github CI, then we should add the :slow tag above
    # Test HF forces on system with spin and magnetism
    using AtomsBuilder
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    using AtomsIO

    system = bulk(:Fe, cubic=true)
    rattle!(system, 0.001u"Å")
    system = load_system("structures/Fe_rattled.extxyz")
    pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
    TestForces.test_term_forces(system; pseudopotentials, functionals=LDA(),
                                temperature=1e-3, Ecut=20, kgrid=[6, 6, 6],
                                magnetic_moments=[5.0, 5.0], mixing=KerkerMixing())
end

@testitem "Forces term-wise Rutile (full)"  setup=[TestForces] tags=[:slow,:forces] begin
    # An example that failed previously with realistic Ecut and k-grid
    using AtomsIO
    system = load_system("structures/GeO2_stretched.extxyz")
    TestForces.test_term_forces(system; kgrid=[6, 6, 9], Ecut=30, atol=1e-6)
end

@testitem "Forces silicon" setup=[TestCases,TestForces] tags=[:forces] begin
    # End-to end test on silicon and comparison against quantum espresso
    using DFTK
    using PseudoPotentialData

    silicon = TestCases.silicon
    test_forces = TestForces.test_forces

    positions = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    system = atomic_system(silicon.lattice, silicon.atoms, positions)

    pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    (; forces_cart) = test_forces(system; functionals=LDA(), atol=1e-8, pseudopotentials,
                                  Ecut=7, kgrid=[2, 2, 2], kshift=[0, 0, 0], forward_ρ=false,
                                  symmetries_respect_rgrid=true,
                                  fft_size=(18, 18, 18))  # FFT chosen to match QE

    # Test against Abinit v9.6.2 using LibXC v4.3.2 lda_x+lda_c_pw
    # (see testcases_ABINIT/silicon_NLCC_forces)
    reference = [[-0.00574838157984, -0.00455216015517, -0.00333786048065],
                 [ 0.00574838157984,  0.00455216015517,  0.00333786048065]]
    @test maximum(v -> maximum(abs, v), reference - forces_cart) < 1e-5
end

@testitem "Forces silicon (spin, temperature)" setup=[TestCases,TestForces] tags=[:forces] begin
    # End-to end test on silicon using setup with very strange k-grid
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
                    Ecut=7, kgrid=[4, 1, 2], kshift=[1/2, 0, 0], forward_ρ=false)
    end
end

@testitem "Forces TiO2 PBE" setup=[TestForces] tags=[:forces,:slow] begin
    using AtomsIO
    system = load_system("structures/tio2_stretched.extxyz")
    TestForces.test_forces(system; kgrid=[2, 2, 3], Ecut=25,
                           mixing=DielectricMixing(εr=10),
                           atol=1e-7, temperature=1e-3)
end
