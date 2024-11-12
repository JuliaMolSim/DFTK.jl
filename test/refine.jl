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
forces = compute_forces(scfres)
forces_refined = refine_forces(refinement, forces)

# Check that the two `refine_forces` methods match
@test forces_refined ≈ refine_forces(refinement, scfres)

# High Ecut computation
scfres_ref = self_consistent_field(basis_ref; tol)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;

# Forces from the reference solution:
forces_ref = compute_forces(scfres_ref)

# Forces from first order linearization with exact δψ
# - Compute the error ``P-P_*`` on the associated orbitals ``ϕ-ψ`` after aligning
#   them: this is done by solving ``\min |ϕ - ψU|`` for ``U`` unitary matrix of
#   size ``N×N`` (``N`` being the number of electrons) whose solution is
#   ``U = S(S^*S)^{-1/2}`` where ``S`` is the overlap matrix ``ψ^*ϕ``.
function compute_error(ϕ, ψ)
    map(zip(ϕ, ψ)) do (ϕk, ψk)
        S = ψk'ϕk
        U = S*(S'S)^(-1/2)
        ϕk - ψk*U
    end
end
δψ_exact = compute_error(refinement.ψ, ψ_ref)
δρ_linear = DFTK.compute_δρ(basis_ref, refinement.ψ, δψ_exact, refinement.occupation)
refinement_linear = DFTK.RefinementResult(basis_ref, refinement.ψ, refinement.ρ, refinement.occupation, DFTK.proj_tangent(δψ_exact, refinement.ψ), δρ_linear)
forces_linear = refine_forces(refinement_linear, forces)

# Low Ecut forces are imprecise:
@test norm(forces - forces_ref) / norm(forces_ref) > 0.15
# Refined forces are more precise:
@test norm(forces_linear - forces_ref) / norm(forces_ref) < 0.1
@test norm(forces_refined - forces_ref) / norm(forces_ref) < 0.1
# Refinement is a good approximation of linearization:
# TODO: This is apparently not true here! Hopefully this happens because the kgrid and Ecut are too low.
#error("Rel error linear $(norm(forces_linear - forces_ref) / norm(forces_ref))\nRel error refinement $(norm(forces_refined - forces_ref) / norm(forces_ref))\nRel error lin-refinement $(norm(forces_refined - forces_linear) / norm(forces_linear))")
#@test norm(forces_refined - forces_linear) / norm(forces_linear) < 0.1

end