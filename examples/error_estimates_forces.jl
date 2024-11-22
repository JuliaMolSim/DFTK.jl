# # Practical error bounds for the forces
#
# DFTK includes an implementation of the strategy from [^CDKL2022] to compute
# practical error bounds for forces and other quantities of interest.
#
# This is an example showing how to compute error estimates for the forces
# on a ``{\rm TiO}_2`` molecule, from which we can either compute asymptotically
# valid error bounds or increase the precision on the computation of the forces.
#
# [^CDKL2022]:
#     E. Cancès, G. Dusson, G. Kemlin, and A. Levitt
#     *Practical error bounds for properties in plane-wave electronic structure calculations*
#     [SIAM Journal on Scientific Computing 44 (5), B1312-B1340](https://doi.org/10.1137/21M1456224)
using DFTK
using Printf
using LinearAlgebra
using ForwardDiff

# ## Setup
# We setup manually the ``{\rm TiO}_2`` configuration from
# [Materials Project](https://materialsproject.org/materials/mp-2657/).
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
             [0.0      0.0      5.61098]];

# We apply a small displacement to one of the ``\rm Ti`` atoms to get nonzero
# forces.
positions[1] .+= [-0.022, 0.028, 0.035]

# We build a model with one ``k``-point only, not too high `Ecut_ref` and small
# tolerance to limit computational time. These parameters can be increased for
# more precise results.
model = model_DFT(lattice, atoms, positions; functionals=LDA())
kgrid = [1, 1, 1]
Ecut_ref = 35
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid)
tol = 1e-5;

# We also build a basis with smaller `Ecut`, to compute a variational approximation of the reference solution.
#
# !!! note "Choice of convergence parameters"
#     Be careful to choose `Ecut` not too close to `Ecut_ref`.
#     Note also that the current choice `Ecut_ref = 35` is such that the
#     reference solution is not converged and `Ecut = 15` is such that the
#     asymptotic regime (crucial to validate the approach) is barely established.
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid);

# ## Computations
# Compute the solution on the smaller basis:
scfres = self_consistent_field(basis; tol, callback=identity);

# Compute first order corrections `refinement.δψ` and `refinement.δρ`.
# Note that `refinement.ψ` and `refinement.ρ` are the quantities computed with `Ecut`
# and then extended to the reference grid.
# This step is roughly as expensive as the `self_consistent_field` call above.
refinement = refine_scfres(scfres, basis_ref; tol, callback=identity);

# ## Error estimates
# - Computation of the force from the variational solution without any post-processing:
f = compute_forces(scfres)

# - Computation of the forces by a linearization argument when replacing the
#   error ``P-P_*`` by the modified residual ``R_{\rm Schur}(P)``. The latter
#   quantity is computable in practice.
force_refinement = refine_forces(refinement)
forces_refined = f + force_refinement.dF

# A practical estimate of the error on the forces is then the following:
dF_estimate = forces_refined - f

# # Comparisons against non-practical estimates.
# For practical computations one can stop at `forces_refined` and `dF_estimate`.
# We continue here with a comparison of different ways to obtain the refined forces,
# noting that the computational cost is much higher.

# ## Computations
# We compute the reference solution ``P_*`` from which we will compute the
# references forces.
scfres_ref = self_consistent_field(basis_ref; tol, callback=identity)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;

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
error = compute_error(refinement.ψ, ψ_ref);

# ## Error estimates

# We start with different estimations of the forces:
# - Force from the reference solution
f_ref = compute_forces(scfres_ref)
forces   = Dict("F(P_*)" => f_ref)
relerror = Dict("F(P_*)" => 0.0)
compute_relerror(f) = norm(f - f_ref) / norm(f_ref);

# - Force from the variational solution and relative error without
#   any post-processing:
forces["F(P)"]   = f

relerror["F(P)"] = compute_relerror(f);

# We then try to improve ``F(P)`` using the first order linearization:
#
# ```math
# F(P) = F(P_*) + {\rm d}F(P)·(P-P_*).
# ```

# To this end, we use the `ForwardDiff.jl` package to compute ``{\rm d}F(P)``
# using automatic differentiation.
function df(basis, occupation, ψ, δψ, ρ)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    ForwardDiff.derivative(ε -> compute_forces(basis, ψ.+ε.*δψ, occupation; ρ=ρ+ε.*δρ), 0)
end;

# - Computation of the forces by a linearization argument if we have access to
#   the actual error ``P-P_*``. Usually this is of course not the case, but this
#   is the "best" improvement we can hope for with a linearisation, so we are
#   aiming for this precision.
df_err = df(basis_ref, refinement.occupation, refinement.ψ,
            DFTK.proj_tangent(error, refinement.ψ), refinement.ρ)
forces["F(P) - df(P)⋅(P-P_*)"]   = f - df_err
relerror["F(P) - df(P)⋅(P-P_*)"] = compute_relerror(f - df_err);

# - Computation of the forces by a linearization argument when replacing the
#   error ``P-P_*`` by the modified residual ``R_{\rm Schur}(P)``. The latter
#   quantity is computable in practice.
forces["F(P) - df(P)⋅Rschur(P)"]   = forces_refined
relerror["F(P) - df(P)⋅Rschur(P)"] = compute_relerror(forces_refined);

# Summary of all forces on the first atom (Ti)
for (key, value) in pairs(forces)
    @printf("%30s = [%7.5f, %7.5f, %7.5f]   (rel. error: %7.5f)\n",
            key, (value[1])..., relerror[key])
end

# Notice how close the computable expression ``F(P) - {\rm d}F(P)⋅R_{\rm Schur}(P)``
# is to the best linearization ansatz ``F(P) - {\rm d}F(P)⋅(P-P_*)``.
