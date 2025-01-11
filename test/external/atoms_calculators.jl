@testitem "Test AtomsCalculators interfaces" setup=[TestCases] tags=[:atomsbase, :dont_test_mpi] begin
    using AtomsBase
    using AtomsCalculators
    using AtomsCalculators.Testing: test_energy_forces_virial
    using DFTK
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    AC = AtomsCalculators

    silicon = TestCases.silicon
    calculator = DFTKCalculator(;
        model_kwargs=(; temperature=1e-3, functionals=LDA(),
                        pseudopotentials=PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")),
        basis_kwargs=(; kgrid=[4, 4, 4], Ecut=5.0),
        scf_kwargs=(; tol=1e-7),
    )
    perturbed_system = periodic_system(silicon.lattice, silicon.atoms,
                                       [silicon.positions[1] + [0.05, 0, 0],
                                        silicon.positions[2]])

    @testset "Energy" begin
        energy = AC.potential_energy(perturbed_system, calculator)
        ref_energy = -7.86059
        @test energy isa Unitful.Energy
        @test austrip(energy) ≈ ref_energy rtol=1e-3
    end

    @testset "Forces" begin
        forces = AC.forces(perturbed_system, calculator)
        ref_forces = [[ 0.01283, -0.0329, -0.0329],
                      [-0.01283,  0.0329,  0.0329]]
        @test all(f -> eltype(f) <: Unitful.Force, forces)
        @test [austrip.(x) for x in forces] ≈ ref_forces rtol=1e-3
    end

    @testset "Virial" begin
        virial = AC.virial(perturbed_system, calculator)
        ref_virial = [[-0.0660718, -0.0489815, -0.0489815],
                      [-0.0489815, -0.078915,   0.0142345],
                      [-0.0489815,  0.0142345, -0.078915]]
        @test austrip.(virial) ≈ hcat(ref_virial...) rtol=1e-3
    end

    # TODO Since the most recent AtomsCalculator update (Aug 2024) this is broken
    # let
    #     calculator_cheap = DFTKCalculator(;
    #         model_kwargs=(; temperature=1e-3, functionals=LDA(), pseudopotentials),
    #         basis_kwargs=(; kgrid=[1, 1, 1], Ecut=5.0),
    #         scf_kwargs=(; tol=1e-6),
    #     )
    #     test_energy_forces_virial(perturbed_system, calculator_cheap; rtol=1e-6)
    # end

    @testset "State transfer reduces number of scf steps" begin
        using AtomsCalculators: calculate, Energy
        using LinearAlgebra

        ps    = AC.get_parameters(calculator)
        state = AC.get_state(calculator)
        res   = AC.calculate(Energy(), perturbed_system, calculator, ps, state)
        n_iter_ref = res.state.n_iter

        res = AC.calculate(Energy(), perturbed_system, calculator, ps, res.state)
        @test res.state.n_iter < n_iter_ref

        F = I + 1e-3randn(3, 3)
        cv = [F * v for v in cell_vectors(perturbed_system)]
        rattled = AbstractSystem(perturbed_system; cell_vectors=cv)
        res = AC.calculate(Energy(), rattled, calculator, ps, res.state)
        @test res.state.n_iter ≤ n_iter_ref
    end
end
