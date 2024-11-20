@testitem "Force refinement practical error bounds (TiO2)" begin
using DFTK
using LinearAlgebra

Ti = ElementPsp(:Ti; psp=load_psp("hgh/lda/ti-q4.hgh"))
O  = ElementPsp(:O; psp=load_psp("hgh/lda/o-q6.hgh"))
atoms     = [Ti, Ti, O, O, O, O]
positions = [[0.5,     0.5,     0.5],  # Ti
             [0.0,     0.0,     0.0],  # Ti
             [0.19542, 0.80458, 0.5],  # O
             [0.80458, 0.19542, 0.5],  # O
             [0.30458, 0.30458, 0.0],  # O
             [0.69542, 0.69542, 0.0]]  # O
lattice   = [[8.79341  0.0      0.0];
             [0.0      8.79341  0.0];
             [0.0      0.0      5.61098]]

# Displace for nonzero forces
positions[1] .+= [-0.022, 0.028, 0.035]

model = model_DFT(lattice, atoms, positions; functionals=LDA())
kgrid = [1, 1, 1]
tol = 1e-5

# Low Ecut computation
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol)

# Refinement in higher Ecut basis
Ecut_ref = 35
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid)

refinement = refine_scfres(scfres, basis_ref; ΩpK_tol=tol)
(; F, dF) = refine_forces(refinement)

# High Ecut computation
scfres_ref = self_consistent_field(basis_ref; tol)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;

# Forces from the reference solution:
forces_ref = compute_forces(scfres_ref)

# Low Ecut forces are imprecise:
@test norm(F - forces_ref) / norm(forces_ref) > 0.15
# Refined forces are more precise:
@test norm(F - dF - forces_ref) / norm(forces_ref) < 0.1

end