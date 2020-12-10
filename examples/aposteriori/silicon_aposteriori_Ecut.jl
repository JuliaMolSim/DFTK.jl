# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# when the error we look at is the basis error : P* is computed for a reference
# Ecut_ref and then we measure the error P-P* and the residual obtained for
# smaller Ecut (currently, only Nk = 1 kpt only is supported)
#
#            !!! NOT OPTIMIZED YET, WE USE PLAIN DENSITY MATRICES !!!
#

# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")
include("aposteriori_callback.jl")
include("newton.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# define different models
modelLDA = model_LDA(lattice, atoms)
#  modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()])
#  modelAtomic = model_atomic(lattice, atoms)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-15
tol_krylov = 1e-15
Ecut_ref = 30           # kinetic energy cutoff in Hartree

Ecut_list = 5:5:25
model_list = ["Atomic", "Hartree", "LDA"]
k = 0

change_norm = true

for model in [modelLDA]#, modelHartree, modelLDA]
    println("--------------------------------")
    println("--------------------------------")
    global k
    k += 1
    println(model_list[k])
    ## reference density matrix
    println("--------------------------------")
    println("reference computation")
    basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol),
                                       callback=info->nothing)
    T = typeof(scfres_ref.ρ.real[1])
    ## number of kpoints
    Nk = length(basis_ref.kpoints)
    ## number of eigenvalue/eigenvectors we are looking for
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]
    φ_ref = similar(scfres_ref.ψ)
    for ik = 1:Nk
        φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
    end

    ## error lists
    norm_err_list = []
    norm_res_list = []

    if change_norm
        Pks = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
        for ik = 1:length(Pks)
            DFTK.precondprep!(Pks[ik], φ_ref[ik])
        end
        norm_Pkres_list = []
        norm_Pkerr_list = []
    else
        Pks = nothing
    end

    for Ecut in Ecut_list

        println("--------------------------------")
        println("Ecut = $(Ecut)")

        # compute solution
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        scfres = self_consistent_field(basis, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol),
                                       callback=info->nothing)
        ## converged values
        φ = similar(scfres.ψ)
        for ik = 1:Nk
            φ[ik] = scfres.ψ[ik][:,1:N]
        end
        ρ = scfres.ρ
        H = scfres.ham
        egval = scfres.eigenvalues
        occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

        # compute residual after interpolating to the reference basis (_ref is here
        # to remember which basis quantities live on, except for Pref which is
        # the reference density matrix)
        φr = DFTK.interpolate_blochwave(φ, basis, basis_ref)
        res = compute_residual(basis_ref, φr, occupation)

        # update list
        append!(norm_err_list, dm_distance(φr[1], φ_ref[1]))
        append!(norm_res_list, norm(res))
        if change_norm
            append!(norm_Pkerr_list, dm_distance(φr[1], φ_ref[1], Pks))
            append!(norm_Pkres_list, norm(apply_inv_sqrt(Pks, res)))
        end
    end

    ## error estimates
    println("--------------------------------")
    println("Computing operator norms...")
    normop_ΩpK, svd_min_ΩpK, svd_max_ΩpK = compute_normop_invΩpK(basis_ref, φ_ref, occupation;
                                                                 tol_krylov=tol_krylov, Pks=nothing)
    err_estimator = normop_ΩpK .* norm_res_list
    if change_norm
        normop_invΩ, svd_min_Ω, svd_max_Ω = compute_normop_invΩ(basis_ref, φ_ref, occupation;
                                                                tol_krylov=tol_krylov, Pks=Pks)
        if model == modelAtomic
            normop_invε = 1.0
            svd_min_ε = 1.0
            svd_max_ε = 1.0
        else
            normop_invε, svd_min_ε, svd_max_ε = compute_normop_invε(basis_ref, φ_ref, occupation;
                                                                    tol_krylov=tol_krylov, Pks=Pks)
        end
        err_Pk_estimator = normop_invε .* normop_invΩ .* norm_Pkres_list
    end

    # plot
    figure()
    title("Silicon
          error estimators vs Ecut, LDA, N = $(N), M = (1-Δ),
          (Ω+K)^-1 : norm = $(@sprintf("%.3f", normop_ΩpK)), min_svd = $(@sprintf("%.3f", svd_min_ΩpK)), max_svd = $(@sprintf("%.3f", svd_max_ΩpK))
          ε^-1 : norm = $(@sprintf("%.3f", normop_invε)), min_svd = $(@sprintf("%.3f", svd_min_ε)), max_svd = $(@sprintf("%.3f", svd_max_ε))
          Ω^-1 : norm = $(@sprintf("%.3f", normop_invΩ)), min_svd = $(@sprintf("%.3f", svd_min_Ω)), max_svd = $(@sprintf("%.3f", svd_max_Ω))")
    semilogy(Ecut_list, norm_err_list, "x-", label="|P-Pref|")
    semilogy(Ecut_list, norm_res_list, "x-", label="|res|")
    semilogy(Ecut_list, err_estimator, "x-", label="|(Ω+K)^{-1}|*|res|")
    if change_norm
        semilogy(Ecut_list, norm_Pkerr_list, "+--", label="|M^1/2(P-Pref)|")
        semilogy(Ecut_list, norm_Pkres_list, "+--", label="|M^-1/2res_φ|")
        semilogy(Ecut_list, err_Pk_estimator, "+--", label="|M^1/2(Ω+K)^-1M^1/2| * |M^-1/2res_φ|")
    end
    legend()
    xlabel("Ecut")
end
