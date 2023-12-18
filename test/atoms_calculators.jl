@testitem "Test AtomsCalculators energy and forces interface" tags=[:dont_test_mpi, :dont_test_windows] setup=[TestCases] begin
    using DFTK
    using AtomsBase
    using AtomsCalculators
    silicon = TestCases.silicon

    # Converto to AtomsBase system
    silicon = periodic_system(silicon.lattice, silicon.atoms, silicon.positions)
    
    model_kwargs = (; functionals = [:lda_x, :lda_c_pw])
    basis_kwargs = (; kgrid = [4, 4, 4], Ecut = 5.0)
    scf_kwargs = (; tol = 1e-7)
    calculator = DFTKCalculator(silicon; model_kwargs, basis_kwargs, scf_kwargs)

    energy = AtomsCalculators.potential_energy(silicon, calculator)
    ref_energy = -7.8691334531
    @test isapprox(energy, ref_energy; rtol=1e-3)

    forces = AtomsCalculators.forces(silicon, calculator)
    ref_forces = [[6.071013955320855e-10, -4.690795889974796e-10, -6.299121885664237e-10],
                  [4.750013095864486e-10, -3.6292771194166813e-10, 3.5873004768426427e-10]]
    @test isapprox(forces, ref_forces; atol=1e-5)
end
