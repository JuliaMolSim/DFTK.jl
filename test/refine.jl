@testitem "Refinement (practical error bounds)" setup=[TestCases] begin
using DFTK
using LinearAlgebra
silicon = TestCases.silicon

# Displace for nonzero forces
positions = deepcopy(silicon.positions)
positions[1] .+= [-0.022, 0.028, 0.035]

model = model_DFT(silicon.lattice, silicon.atoms, positions; functionals=LDA())
kgrid = [1, 1, 1]
tol = 1e-10

# Low Ecut computation
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol)

# Refinement in higher Ecut basis
Ecut_ref = 35
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid)

refinement = refine_scfres(scfres, basis_ref; tol)
energies_refined = refine_energies(refinement)
(; F, dF) = refine_forces(refinement)

# High Ecut computation
scfres_ref = self_consistent_field(basis_ref; tol)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;

# Low Ecut quantities of interest are less precise than refined quantities:

@test norm(scfres.energies.total - scfres_ref.energies.total) / norm(scfres_ref.energies.total) > 0.0002
@test norm(energies_refined.total - scfres_ref.energies.total) / norm(scfres_ref.energies.total) < 0.0001

@test norm(refinement.ρ - scfres_ref.ρ) / norm(scfres_ref.ρ) > 0.003
@test norm(refinement.ρ + refinement.δρ - scfres_ref.ρ) / norm(scfres_ref.ρ) < 0.0015

# Forces from the reference solution:
forces_ref = compute_forces(scfres_ref)

@test norm(F - forces_ref) / norm(forces_ref) > 0.005
@test norm(F + dF - forces_ref) / norm(forces_ref) < 0.002

end