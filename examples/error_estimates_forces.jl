# # Practical error bounds for the forces
#
# This is a simple example showing how to compute error estimates for the forces
# on a ``{\rm TiO}_2`` molecule, from which we can either compute asymptotically
# valid error bounds or increase the precision on the computation of the forces.
#
# The strategy we follow is described with more details in [^CDKL2021] and we
# will use in comments the density matrices framework. We will also needs
# operators and functions from
# [`src/scf/newton.jl`](https://dftk.org/blob/master/src/scf/newton.jl).
#
# [^CDKL2021]:
#     E. Cancès, G. Dusson, G. Kemlin, and A. Levitt
#     *Practical error bounds for properties in plane-wave electronic structure
#     calculations* Preprint, 2021. [arXiv](https://arxiv.org/abs/2111.01470)
using DFTK
using Printf
using LinearAlgebra
using ForwardDiff
using LinearMaps
using IterativeSolvers

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
model = model_LDA(lattice, atoms, positions)
kgrid = [1, 1, 1]
Ecut_ref = 35
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid)
tol = 1e-5;

# ## Computations

# We compute the reference solution ``P_*`` from which we will compute the
# references forces.
scfres_ref = self_consistent_field(basis_ref; tol, callback=identity)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ, scfres_ref.occupation).ψ;

# We compute a variational approximation of the reference solution with
# smaller `Ecut`. `ψr`, `ρr` and `Er` are the quantities computed with `Ecut`
# and then extended to the reference grid.
#
# !!! note "Choice of convergence parameters"
#     Be careful to choose `Ecut` not too close to `Ecut_ref`.
#     Note also that the current choice `Ecut_ref = 35` is such that the
#     reference solution is not converged and `Ecut = 15` is such that the
#     asymptotic regime (crucial to validate the approach) is barely established.
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol, callback=identity)
ψr = DFTK.transfer_blochwave(scfres.ψ, basis, basis_ref)
ρr = compute_density(basis_ref, ψr, scfres.occupation)
Er, hamr = energy_hamiltonian(basis_ref, ψr, scfres.occupation; ρ=ρr);

# We then compute several quantities that we need to evaluate the error bounds.

# - Compute the residual ``R(P)``, and remove the virtual orbitals, as required
#   in [`src/scf/newton.jl`](https://github.com/JuliaMolSim/DFTK.jl/blob/fedc720dab2d194b30d468501acd0f04bd4dd3d6/src/scf/newton.jl#L121).
res = DFTK.compute_projected_gradient(basis_ref, ψr, scfres.occupation)
res, occ = DFTK.select_occupied_orbitals(basis_ref, res, scfres.occupation)
ψr = DFTK.select_occupied_orbitals(basis_ref, ψr, scfres.occupation).ψ;

# - Compute the error ``P-P_*`` on the associated orbitals ``ϕ-ψ`` after aligning
#   them: this is done by solving ``\min |ϕ - ψU|`` for ``U`` unitary matrix of
#   size ``N×N`` (``N`` being the number of electrons) whose solution is
#   ``U = S(S^*S)^{-1/2}`` where ``S`` is the overlap matrix ``ψ^*ϕ``.
function compute_error(basis, ϕ, ψ)
    map(zip(ϕ, ψ)) do (ϕk, ψk)
        S = ψk'ϕk
        U = S*(S'S)^(-1/2)
        ϕk - ψk*U
    end
end
err = compute_error(basis_ref, ψr, ψ_ref);

# - Compute ``{\boldsymbol M}^{-1}R(P)`` with ``{\boldsymbol M}^{-1}`` defined in [^CDKL2021]:
P = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
map(zip(P, ψr)) do (Pk, ψk)
    DFTK.precondprep!(Pk, ψk)
end
function apply_M(φk, Pk, δφnk, n)
    DFTK.proj_tangent_kpt!(δφnk, φk)
    δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
    DFTK.proj_tangent_kpt!(δφnk, φk)
    δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
    DFTK.proj_tangent_kpt!(δφnk, φk)
end
function apply_inv_M(φk, Pk, δφnk, n)
    DFTK.proj_tangent_kpt!(δφnk, φk)
    op(x) = apply_M(φk, Pk, x, n)
    function f_ldiv!(x, y)
        x .= DFTK.proj_tangent_kpt(y, φk)
        x ./= (Pk.mean_kin[n] .+ Pk.kin)
        DFTK.proj_tangent_kpt!(x, φk)
    end
    J = LinearMap{eltype(φk)}(op, size(δφnk, 1))
    δφnk = cg(J, δφnk; Pl=DFTK.FunctionPreconditioner(f_ldiv!),
              verbose=false, reltol=0, abstol=1e-15)
    DFTK.proj_tangent_kpt!(δφnk, φk)
end
function apply_metric(φ, P, δφ, A::Function)
    map(enumerate(δφ)) do (ik, δφk)
        Aδφk = similar(δφk)
        φk = φ[ik]
        for n = 1:size(δφk,2)
            Aδφk[:,n] = A(φk, P[ik], δφk[:,n], n)
        end
        Aδφk
    end
end
Mres = apply_metric(ψr, P, res, apply_inv_M);

# We can now compute the modified residual ``R_{\rm Schur}(P)`` using a Schur
# complement to approximate the error on low-frequencies[^CDKL2021]:
#
# ```math
# \begin{bmatrix}
# (\boldsymbol Ω + \boldsymbol K)_{11} & (\boldsymbol Ω + \boldsymbol K)_{12} \\
# 0 & {\boldsymbol M}_{22}
# \end{bmatrix}
# \begin{bmatrix}
# P_{1} - P_{*1} \\ P_{2}-P_{*2}
# \end{bmatrix} =
# \begin{bmatrix}
# R_{1} \\ R_{2}
# \end{bmatrix}.
# ```

# - Compute the projection of the residual onto the high and low frequencies:
resLF = DFTK.transfer_blochwave(res, basis_ref, basis)
resHF = res - DFTK.transfer_blochwave(resLF, basis, basis_ref);

# - Compute ``{\boldsymbol M}^{-1}_{22}R_2(P)``:
e2 = apply_metric(ψr, P, resHF, apply_inv_M);

# - Compute the right hand side of the Schur system:
## Rayleigh coefficients needed for `apply_Ω`
Λ = map(enumerate(ψr)) do (ik, ψk)
    Hk = hamr.blocks[ik]
    Hψk = Hk * ψk
    ψk'Hψk
end
ΩpKe2 = DFTK.apply_Ω(e2, ψr, hamr, Λ) .+ DFTK.apply_K(basis_ref, e2, ψr, ρr, occ)
ΩpKe2 = DFTK.transfer_blochwave(ΩpKe2, basis_ref, basis)
rhs = resLF - ΩpKe2;

# - Solve the Schur system to compute ``R_{\rm Schur}(P)``: this is the most
#   costly step, but inverting ``\boldsymbol{Ω} + \boldsymbol{K}`` on the small space has
#   the same cost than the full SCF cycle on the small grid.
(; ψ) = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)
e1 = DFTK.solve_ΩplusK(basis, ψ, rhs, occ; tol).δψ
e1 = DFTK.transfer_blochwave(e1, basis, basis_ref)
res_schur = e1 + Mres;

# ## Error estimates

# We start with different estimations of the forces:
# - Force from the reference solution
f_ref = compute_forces(scfres_ref)
forces   = Dict("F(P_*)" => f_ref)
relerror = Dict("F(P_*)" => 0.0)
compute_relerror(f) = norm(f - f_ref) / norm(f_ref);

# - Force from the variational solution and relative error without
#   any post-processing:
f = compute_forces(scfres)
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
df_err = df(basis_ref, occ, ψr, DFTK.proj_tangent(err, ψr), ρr)
forces["F(P) - df(P)⋅(P-P_*)"]   = f - df_err
relerror["F(P) - df(P)⋅(P-P_*)"] = compute_relerror(f - df_err);

# - Computation of the forces by a linearization argument when replacing the
#   error ``P-P_*`` by the modified residual ``R_{\rm Schur}(P)``. The latter
#   quantity is computable in practice.
df_schur = df(basis_ref, occ, ψr, res_schur, ρr)
forces["F(P) - df(P)⋅Rschur(P)"]   = f - df_schur
relerror["F(P) - df(P)⋅Rschur(P)"] = compute_relerror(f - df_schur);

# Summary of all forces on the first atom (Ti)
for (key, value) in pairs(forces)
    @printf("%30s = [%7.5f, %7.5f, %7.5f]   (rel. error: %7.5f)\n",
            key, (value[1])..., relerror[key])
end

# Notice how close the computable expression ``F(P) - {\rm d}F(P)⋅R_{\rm Schur}(P)``
# is to the best linearization ansatz ``F(P) - {\rm d}F(P)⋅(P-P_*)``.
