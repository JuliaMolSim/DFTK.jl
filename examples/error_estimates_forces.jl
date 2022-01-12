# # Practical error bounds for the forces
#
# This is a simple example showing how to compute error estimates for the forces on a
# ``{\rm TiO}_2`` molecule, from which we can either compute asymptotically valid error
# bounds or increase the precision on the computation of the forces.
#
# The strategy we follow is described with more details in[^CDKL2021] and we
# will use in comments the density matrices framework. We will also needs
# operators and functions from
# [`src/scf/newton.jl`](https://dftk.org/blob/master/src/scf/newton.jl).
#
# [^CDKL2021]:
#     E. Cancès, G. Dusson, G. Kemlin, and A. Levitt
#     *Practical error bounds for properties in plane-wave electronic structure
#     calculations* Preprint, 2021. [arXiv](https://arxiv.org/abs/2111.01470)
using DFTK
using LinearAlgebra
using ForwardDiff
using LinearMaps
using IterativeSolvers

# ## Setup
# We setup manually the ``{\rm TiO}_2`` configuration from
# [Materials Project](https://materialsproject.org/materials/mp-2657/).
Ti = ElementPsp(:Ti, psp=load_psp("hgh/lda/ti-q4.hgh"))
O = ElementPsp(:O, psp=load_psp("hgh/lda/o-q6.hgh"))
atoms = [Ti => [[0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0]],
         O  => [[0.19542, 0.80458, 0.5],
                [0.80458, 0.19542, 0.5],
                [0.30458, 0.30458, 0.0],
                [0.69542, 0.69542, 0.0]]]
lattice = [[8.79341  0.0      0.0];
           [0.0      8.79341  0.0];
           [0.0      0.0      5.61098]];

# We apply a small displacement to one of the ``\rm Ti`` atoms to get nonzero forces.
el, pos = atoms[1]
atoms[1] = Pair(el, pos .+ [[0.22, -0.28, 0.35] / 5, [0, 0, 0]]);
# We build a model with one k-point only, not too high `Ecut_ref` and small tolerance to
# limit computational time. These parameters can be increased for more precise
# results.
model = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]
Ecut_ref = 50
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid=kgrid)
tol = 1e-6;

# ## Computations
# We compute the reference solution ``P_*`` from which we will compute the
# references forces.
scfres_ref = self_consistent_field(basis_ref, tol=tol, callback=info->nothing)
ψ_ref = DFTK.select_occupied_orbitals(basis_ref, scfres_ref.ψ)
filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk];
# We compute a variational approximation of the reference solution with
# smaller `Ecut`. Be careful not to choose `Ecut` to close to `Ecut_ref`.
Ecut = 20
basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
scfres = self_consistent_field(basis, tol=tol, callback=info->nothing)
ψ = DFTK.select_occupied_orbitals(basis, scfres.ψ)
ψr = DFTK.transfer_blochwave(scfres.ψ, basis, basis_ref)
ρr = compute_density(basis_ref, ψr, scfres.occupation)
_, ham = energy_hamiltonian(basis_ref, ψr, scfres.occupation; ρ=ρr);
# We then compute several quantities that we need to evaluate the error bounds.
# - Compute the residual ``R(P)``, and remove the virtual orbitals, as required in [`src/scf/newton.jl`](https://github.com/JuliaMolSim/DFTK.jl/blob/fedc720dab2d194b30d468501acd0f04bd4dd3d6/src/scf/newton.jl#L121).
res = DFTK.compute_projected_gradient(basis_ref, ψr, scfres.occupation)
res = DFTK.select_occupied_orbitals(basis_ref, res)
ψr = DFTK.select_occupied_orbitals(basis_ref, ψr);
# - Compute the error ``P-P_*`` on the orbitals ``ψ`` and ``ψ_*`` by aligning the eigenvectors: this is done by solving ``\min |ψ - ψ_*U|`` for ``U`` unitary matrix of size ``N\times N`` (``N`` being the number of electrons) whose solution is ``U = S(S^*S)^{-1/2}`` where ``S`` is the overlap matrix ``ψ_*^*ψ``.
function compute_error(basis, ϕ, ψ)
    Nk = length(basis.kpoints)
    err = similar(ϕ)
    for ik = 1:Nk
        ϕk = ϕ[ik]
        ψk = ψ[ik]
        M = ψk'ϕk
        U = M*(M'M)^(-1/2)
        err[ik] = ϕk - ψk*U
    end
    err
end
err = compute_error(basis_ref, ψr, ψ_ref);
# - Compute ``{\bm M}^{-1}R(P)`` with ``{\bm M}^{-1}`` defined in[^CDKL2021]:
P = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
for (ik, ψk) in enumerate(ψr)
    DFTK.precondprep!(P[ik], ψk)
end
function apply_M(φk, Pk, δφnk, n)
    δφnk = DFTK.proj_tangent_kpt(δφnk, φk)
    δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
    δφnk = DFTK.proj_tangent_kpt(δφnk, φk)
    δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
    δφnk = DFTK.proj_tangent_kpt(δφnk, φk)
end
function apply_inv_M(φk, Pk, δφnk, n)
    δφnk = DFTK.proj_tangent_kpt(δφnk, φk)
    function op(x)
        apply_M(φk, Pk, x, n)
    end
    function f_ldiv!(x, y)
        x .= DFTK.proj_tangent_kpt(y, φk)
        x ./= (Pk.mean_kin[n] .+ Pk.kin)
        DFTK.proj_tangent_kpt!(x, φk)
    end
    J = LinearMap{eltype(φk)}(op, size(δφnk, 1))
    δφnk = cg(J, δφnk, Pl=DFTK.FunctionPreconditioner(f_ldiv!),
              verbose=false, reltol=0, abstol=1e-15)
    DFTK.proj_tangent_kpt(δφnk, φk)
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
# ```math
# \begin{bmatrix} (\bm \Omega + \bm K)_{11} & (\bm \Omega + \bm K)_{12} \\ 0 & {\bm M}_{22} \end{bmatrix} \begin{bmatrix} P_{1} - P_{*1}\\ P_{2}-P_{*2} \end{bmatrix}= \begin{bmatrix} R_{1}\\R_{2} \end{bmatrix}.
# ```
# - Compute the projection of the residual onto the high and low frequencies:
resLF = DFTK.transfer_blochwave(res, basis_ref, basis)
resHF = res - DFTK.transfer_blochwave(resLF, basis, basis_ref);
# - Compute ``{\bm M}^{-1}_{22}R_2(P)``:
e2 = apply_metric(ψr, P, resHF, apply_inv_M);
# - Compute the right hand side of the Schur system:
## Rayleigh coefficients needed for `apply_Ω`
Λ = map(enumerate(ψr)) do (ik, ψk)
    Hk = ham.blocks[ik]
    Hψk = Hk * ψk
    ψk'Hψk
end
ΩpKe2 = DFTK.apply_Ω(e2, ψr, ham, Λ) .+ DFTK.apply_K(basis_ref, e2, ψr, ρr, occupation)
ΩpKe2 = DFTK.transfer_blochwave(ΩpKe2, basis_ref, basis)
rhs = resLF - ΩpKe2;
# - Solve the Schur system to compute ``R_{\rm Schur}(P)``: this is the most costly step, but inverting ``\bm{\Omega} + \bm{K}`` on the small space has the same cost than the full SCF cycle on the small grid.
e1 = DFTK.solve_ΩplusK(basis, ψ, rhs, occupation; tol_cg=tol)
e1 = DFTK.transfer_blochwave(e1, basis, basis_ref)
res_schur = e1 + Mres;

# ## Error estimates
# We start with different estimations of the forces on the first ``\rm Ti`` atom, in direction 1.
# - Forces computed with the reference solution:
f_ref = compute_forces(scfres_ref)
println("F(P_*) = $(f_ref[1][1][1])")
# - Forces computed with the variational approximation:
f = compute_forces(scfres)
println("F(P) = $(f[1][1][1])")
# We then try to improve ``F(P)`` using the first order linearization:
# ```math
# F(P) = F(P_*) + {\rm d}F(P)\cdot(P-P_*).
# ```
# To this end, we use the `ForwardDiff.jl` package to compute ``{\rm d}F(P)``
# using automatic differentiation.
function df(basis, occupation, ψ, δψ)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    function f(ε)
        compute_forces(basis, ψ .+ ε.*δψ, occupation; ρ=ε.*δρ)
    end
    ForwardDiff.derivative(f, 0)
end;
# - Computation of the forces by a linearization argument if we have access to the actual error ``P-P_*``:
df_err = df(basis_ref, occupation, ψr, DFTK.proj_tangent(err, ψr))
println("F(P) - df(P).(P-P_*) = $(f[1][1][1]-df_err[1][1][1])")
# - Computation of the forces by a linearization argument when replacing the error ``P-P_*`` by the modified residual:
df_schur = df(basis_ref, occupation, ψr, res_schur)
println("F(P) - df(P).Rschur(P) = $(f[1][1][1]-df_schur[1][1][1])")
# Then, we estimate the error on the forces made by the different computations
# above:
# - Relative error on the forces with no post-processing:
println("|F(P) - F(P_*)| / |F(P_*)| = $(norm(f-f_ref)/norm(f_ref))")
# - Relative error made by the linearization ``F(P) - dF(P).(P-P_*)`` if we had access to the actual error ``P-P_*``, which is not the case in practice (we are aiming at reaching this precision):
println("|F(P) - dF(P).(P-P_*) - F(P_*)| / |F(P_*)| = $(norm(f-df_err-f_ref)/norm(f_ref))")
# - Relative error made by replacing ``P-P_*`` by the modified residual ``R_{\rm Schur}(P)`` (computable in practice) in the linearization (note how closer we are to the previous one):
println("|F(P) - dF(P).Rschur(P) - F(P_*)| / |F(P_*)| = $(norm(f-df_schur-f_ref)/norm(f_ref))")
