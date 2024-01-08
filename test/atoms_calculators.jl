@testitem "Test AtomsCalculators energy and forces interface" setup=[TestCases] begin
    using DFTK
    using AtomsBase
    using AtomsCalculators
    silicon = TestCases.silicon

    # Perturb from equilibrium so forces are not 0.
    silicon.positions[1] += [0.05, 0, 0]
    # Converto to AtomsBase system
    silicon = periodic_system(silicon.lattice, silicon.atoms, silicon.positions)
    
    model_kwargs = (; temperature=1e-6, functionals=[:lda_x, :lda_c_pw])
    basis_kwargs = (; kgrid=[4, 4, 4], Ecut=5.0)
    scf_kwargs = (; tol=1e-7)
    calculator = DFTKCalculator(silicon; model_kwargs, basis_kwargs, scf_kwargs)

    energy = AtomsCalculators.potential_energy(silicon, calculator)
    ref_energy = -7.86054
    @test energy ≈ ref_energy rtol=1e-3

    forces = AtomsCalculators.forces(silicon, calculator)
    ref_forces = [[0.01283, -0.0329, -0.0329],
                  [-0.01283, 0.0329, 0.0329]]
    @test forces ≈ ref_forces rtol=1e-3
end
