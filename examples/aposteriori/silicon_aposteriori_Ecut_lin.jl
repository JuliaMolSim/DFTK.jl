# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# and
#
# M^1/2(P-P*) = M^1/2(Ω+K)^{-1}M^1/2 * M^-1/2[P,[P,H(P)]]
#
# translated to orbitals, in the linear case.
# We look at is the basis error : φ* is computed for a reference
# Ecut_ref and then we measure the error φ-φ* and the residual obtained for
# smaller Ecut
#

using DFTK
using LinearAlgebra
using PyPlot

# import aux files
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

# Very basic setup, useful for testing
# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

## local potential only
model = model_atomic(lattice, atoms)

kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-12
tol_krylov = 1e-15
Ecut_ref = 100           # kinetic energy cutoff in Hartree
Ecut_list = 10:5:(Ecut_ref-10)

## changing norm for error estimation
change_norm = true

println("--------------------------------")
println("reference computation")
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

## We work on the solution to keep only occupied orbitals
T = typeof(scfres_ref.ρ.real[1])
# number of kpoints
Nk = length(basis_ref.kpoints)
# number of eigenvalue/eigenvectors we are looking for
filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]
φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end
gap = scfres_ref.eigenvalues[1][N+1] - scfres_ref.eigenvalues[1][N]

## error lists
norm_err_list = []
norm_res_list = []

if change_norm
    # mean kin
    Pk_kin = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    for ik = 1:length(Pk_kin)
        DFTK.precondprep!(Pk_kin[ik], φ_ref[ik])
    end
    norm_Pk_kin_res_list  = []
    norm_Pk_kin_err_list  = []
    norm_ΠPk_kin_res_list = []
    norm_ΠPk_kin_err_list = []
end

for Ecut in Ecut_list

    println("--------------------------------")
    println("Ecut = $(Ecut)")

    # compute solution
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol),
                                   callback=info->nothing)

    ## keep only the occupied orbitals
    φ = similar(scfres.ψ)
    for ik = 1:Nk
        φ[ik] = scfres.ψ[ik][:,1:N]
    end
    ρ = scfres.ρ
    H = scfres.ham
    egval = scfres.eigenvalues
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    # compute residual and error after interpolating to the reference basis
    φr = DFTK.interpolate_blochwave(φ, basis, basis_ref)
    res = compute_residual(basis_ref, φr, occupation)
    err = compute_error(basis_ref, φr, φ_ref)

    # update lists
    append!(norm_err_list, norm(err))
    append!(norm_res_list, norm(res))
    if change_norm
        append!(norm_Pk_kin_err_list,  norm(apply_sqrt(Pk_kin, err)))
        append!(norm_Pk_kin_res_list,  norm(apply_inv_sqrt(Pk_kin, res)))
        append!(norm_ΠPk_kin_err_list, norm(proj(apply_sqrt(Pk_kin, err), φ_ref)))
        append!(norm_ΠPk_kin_res_list, norm(proj(apply_inv_sqrt(Pk_kin, res), φ_ref)))
    end
end

## error estimates
println("--------------------------------")
println("Computing operator norms...")
normop_invΩpK, svd_min_ΩpK, svd_max_ΩpK = compute_normop_invΩpK(basis_ref, φ_ref, occupation;
                                                                tol_krylov=tol_krylov, Pks=nothing)
err_estimator = normop_invΩpK .* norm_res_list
if change_norm
    normop_invΩ_kin, svd_min_Ω_kin, svd_max_Ω_kin = compute_normop_invΩ(basis_ref, φ_ref, occupation;
                                                                        tol_krylov=tol_krylov, Pks=Pk_kin,
                                                                        change_norm=change_norm)
    err_Pk_estimator = normop_invΩ_kin .* norm_Pk_kin_res_list
end

figure()
title("Silicon
      error estimators vs Ecut, Atomic, kgrid = $(kgrid), N = $(N), M = (T-Δ), gap = $(@sprintf("%.3f", gap))
      Ω^-1 : norm = $(@sprintf("%.3f", normop_invΩpK)), min_svd = $(@sprintf("%.3f", svd_min_ΩpK)), max_svd = $(@sprintf("%.3f", svd_max_ΩpK))
      M^1/2Ω^-1M^1/2 : norm = $(@sprintf("%.3f", normop_invΩ_kin)), min_svd = $(@sprintf("%.3f", svd_min_Ω_kin)), max_svd = $(@sprintf("%.3f", svd_max_Ω_kin))")
semilogy(Ecut_list, norm_err_list, "x-", label="|φ-φref|")
semilogy(Ecut_list, norm_res_list, "x-", label="|res|")
semilogy(Ecut_list, err_estimator, "x-", label="|(Ω+K)^-1| * |res|")
if change_norm
    semilogy(Ecut_list, norm_Pk_kin_err_list, "+--", label="|M^1/2(φ-φref)| kin")
    semilogy(Ecut_list, norm_Pk_kin_res_list, "+--", label="|M^-1/2res| kin")
    semilogy(Ecut_list, norm_ΠPk_kin_err_list, "+:", label="|ΠM^1/2(φ-φref)| kin")
    semilogy(Ecut_list, norm_ΠPk_kin_res_list, "+:", label="|ΠM^-1/2res| kin")
    semilogy(Ecut_list, err_Pk_estimator, "+--", label="|M^1/2ε^-TM^-1/2| * |M^1/2Ω^-1M^1/2| * |M^-1/2res| kin")
end
legend()
xlabel("Ecut")
