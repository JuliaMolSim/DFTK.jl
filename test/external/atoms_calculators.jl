@testitem "Test AtomsCalculators energy interface" setup=[TestCases] begin
    using DFTK
    using AtomsBase
    using AtomsCalculators
    using Unitful
    using UnitfulAtomic
    silicon = TestCases.silicon

    # Perturb from equilibrium so forces are not 0.
    positions = [silicon.positions[1] + [0.05, 0, 0], silicon.positions[2]]
    silicon = periodic_system(silicon.lattice, silicon.atoms, positions)

    model_kwargs = (; temperature=1e-3, functionals=[:lda_x, :lda_c_pw])
    basis_kwargs = (; kgrid=[4, 4, 4], Ecut=5.0)
    scf_kwargs = (; tol=1e-7)
    calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)

    energy = AtomsCalculators.potential_energy(silicon, calculator)
    ref_energy = -7.86059
    @test austrip.(energy) ≈ ref_energy rtol=1e-3
end

@testitem "Test AtomsCalculators energy forces interface" setup=[TestCases] begin
    using DFTK
    using AtomsBase
    using AtomsCalculators
    using Unitful
    using UnitfulAtomic
    silicon = TestCases.silicon

    # Perturb from equilibrium so forces are not 0.
    positions = [silicon.positions[1] + [0.05, 0, 0], silicon.positions[2]]
    silicon = periodic_system(silicon.lattice, silicon.atoms, positions)

    model_kwargs = (; temperature=1e-3, functionals=[:lda_x, :lda_c_pw])
    basis_kwargs = (; kgrid=[4, 4, 4], Ecut=5.0)
    scf_kwargs = (; tol=1e-7)
    calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)

    forces = AtomsCalculators.forces(silicon, calculator)
    ref_forces = [[ 0.01283, -0.0329, -0.0329],
                  [-0.01283,  0.0329,  0.0329]]
    @test [austrip.(x) for x in forces] ≈ ref_forces rtol=1e-3
end

@testitem "Test AtomsCalculators virial interface" setup=[TestCases] begin
    using DFTK
    using AtomsBase
    using AtomsCalculators
    using Unitful
    using UnitfulAtomic
    silicon = TestCases.silicon

    # Perturb from equilibrium so forces are not 0.
    positions = [silicon.positions[1] + [0.05, 0, 0], silicon.positions[2]]
    silicon = periodic_system(silicon.lattice, silicon.atoms, positions)

    model_kwargs = (; temperature=1e-3, functionals=[:lda_x, :lda_c_pw])
    basis_kwargs = (; kgrid=[4, 4, 4], Ecut=5.0)
    scf_kwargs = (; tol=1e-7)
    calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)

    virial = AtomsCalculators.virial(silicon, calculator)
    ref_virial = [[-0.0660718, -0.0489815, -0.0489815],
                  [-0.0489815, -0.078915, 0.0142345], 
                  [-0.0489815, 0.0142345, -0.078915]]
    @test austrip.(virial) ≈ hcat(ref_virial...) rtol=1e-3
end
