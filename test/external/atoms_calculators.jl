@testitem "Test AtomsCalculators energy interface" setup=[TestCases] begin
    using AtomsCalculators
    using Unitful
    using UnitfulAtomic
    
    energy = AtomsCalculators.potential_energy(TestCases.silicon.perturbed_system, TestCases.silicon.calculator)
    ref_energy = -7.86059
    @test austrip(energy) ≈ ref_energy rtol=1e-3
end

@testitem "Test AtomsCalculators energy forces interface" setup=[TestCases] begin
    using AtomsCalculators
    using Unitful
    using UnitfulAtomic
    
    forces = AtomsCalculators.forces(TestCases.silicon.perturbed_system, TestCases.silicon.calculator)
    ref_forces = [[ 0.01283, -0.0329, -0.0329],
                  [-0.01283,  0.0329,  0.0329]]
    @test [austrip.(x) for x in forces] ≈ ref_forces rtol=1e-3
end

@testitem "Test AtomsCalculators virial interface" setup=[TestCases] begin
    using AtomsCalculators
    using Unitful
    using UnitfulAtomic
    
    virial = AtomsCalculators.virial(TestCases.silicon.perturbed_system, TestCases.silicon.calculator)
    ref_virial = [[-0.0660718, -0.0489815, -0.0489815],
                  [-0.0489815, -0.078915, 0.0142345], 
                  [-0.0489815, 0.0142345, -0.078915]]
    @test austrip.(virial) ≈ hcat(ref_virial...) rtol=1e-3
end
