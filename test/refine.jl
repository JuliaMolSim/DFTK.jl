@testitem "Force refinement practical error bounds" setup=[TestCases] begin
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
(; F, dF) = refine_forces(refinement)

# High Ecut computation
scfres_ref = self_consistent_field(basis_ref; tol)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;

# Forces from the reference solution:
forces_ref = compute_forces(scfres_ref)

# Low Ecut forces are imprecise:
@test norm(F - forces_ref) / norm(forces_ref) > 0.005
# Refined forces are more precise:
@test norm(F + dF - forces_ref) / norm(forces_ref) < 0.002

end