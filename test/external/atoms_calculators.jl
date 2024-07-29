@testitem "Test AtomsCalculators interfaces" setup=[TestCases] tags=[:atomsbase] begin
    using AtomsCalculators
    using AtomsCalculators.AtomsCalculatorsTesting: test_energy_forces_virial
    using Unitful
    using UnitfulAtomic

    silicon = TestCases.silicon
    calculator = DFTKCalculator(;
        model_kwargs=(; temperature=1e-3, functionals=[:lda_x, :lda_c_pw]),
        basis_kwargs=(; kgrid=[4, 4, 4], Ecut=5.0),
        scf_kwargs=(; tol=1e-7),
        verbose=true
    )
    perturbed_system = periodic_system(silicon.lattice, silicon.atoms,
                                       [silicon.positions[1] + [0.05, 0, 0],
                                        silicon.positions[2]])

    @testset "Energy" begin
        energy = AtomsCalculators.potential_energy(perturbed_system, calculator)
        ref_energy = -7.86059
        @test energy isa Unitful.Energy
        @test austrip(energy) ≈ ref_energy rtol=1e-3
    end

    @testset "Forces" begin
        forces = AtomsCalculators.forces(perturbed_system, calculator)
        ref_forces = [[ 0.01283, -0.0329, -0.0329],
                      [-0.01283,  0.0329,  0.0329]]
        @test all(f -> eltype(f) <: Unitful.Force, forces)
        @test [austrip.(x) for x in forces] ≈ ref_forces rtol=1e-3
    end

    @testset "Virial" begin
        virial = AtomsCalculators.virial(perturbed_system, calculator)
        ref_virial = [[-0.0660718, -0.0489815, -0.0489815],
                      [-0.0489815, -0.078915, 0.0142345],
                      [-0.0489815, 0.0142345, -0.078915]]
        @test austrip.(virial) ≈ hcat(ref_virial...) rtol=1e-3
    end

    let
        calculator_cheap = DFTKCalculator(;
            model_kwargs=(; temperature=1e-3, functionals=[:lda_x, :lda_c_pw]),
            basis_kwargs=(; kgrid=[1, 1, 1], Ecut=5.0),
            scf_kwargs=(; tol=1e-4),
        )
        test_energy_forces_virial(perturbed_system, calculator_cheap)
    end
end
