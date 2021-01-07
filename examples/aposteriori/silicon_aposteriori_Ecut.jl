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
#  modelLDA = model_LDA(lattice, atoms)
#  modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()])
#  modelAtomic = model_atomic(lattice, atoms, n_electrons=2)
modelAtomic = model_atomic(lattice, atoms)

## local potential only
#  modelAtomic = Model(lattice; atoms=atoms,
#                      terms=[Kinetic(), AtomicLocal()],
#                      n_electrons=4)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-10
tol_krylov = 1e-15
Ecut_ref = 50           # kinetic energy cutoff in Hartree

Ecut_list = 10:5:(Ecut_ref-10)
model_list = ["Atomic", "Hartree", "LDA"]
k = 0

change_norm = true

for model in [modelAtomic]#, modelHartree, modelLDA]
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
                                       determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                       is_converged=DFTK.ScfConvergenceDensity(tol))
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
    gap = scfres_ref.eigenvalues[1][N+1] - scfres_ref.eigenvalues[1][N]
    H_ref = scfres_ref.ham
    println(typeof.(H_ref.blocks[1].operators))

    ## error lists
    norm_err_list = []
    norm_res_list = []

    ## fourier coef of Vlocal
    Vloc = Complex{Float64}.(DFTK.total_local_potential(scfres_ref.ham))
    Vloc_fourier = r_to_G(basis_ref, basis_ref.kpoints[1], Vloc)

    if change_norm
        # mean kin
        Pk_kin = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
        for ik = 1:length(Pk_kin)
            DFTK.precondprep!(Pk_kin[ik], φ_ref[ik])
        end
        norm_Pk_kin_res_list = []
        norm_Pk_kin_err_list = []

        # mean pot
        Pk_pot = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
        function LinearAlgebra.Diagonal(opnl::DFTK.NonlocalOperator)
            [dot(p, opnl.D * p) for p in eachrow(opnl.P)]
        end
        for ik = 1:length(Pk_pot)
            non_local_op = [op for op in scfres_ref.ham.blocks[ik].operators if (op isa DFTK.NonlocalOperator)][1]
            Pk_pot[ik].mean_kin = [abs.(Diagonal(non_local_op)) .+ real(dot(x, Pk_pot[ik].kin .* x)) for x in eachcol(φ_ref[ik])]
        end
        norm_Pk_pot_res_list = []
        norm_Pk_pot_err_list = []

    else
        Pk_kin = nothing
        Pk_pot = nothing
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
        ## converged values
        φ = similar(scfres.ψ)
        for ik = 1:Nk
            φ[ik] = scfres.ψ[ik][:,1:N]
        end
        ρ = scfres.ρ
        H = scfres.ham
        egval = scfres.eigenvalues
        occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

        # compute residual after interpolating to the reference basis
        φr = DFTK.interpolate_blochwave(φ, basis, basis_ref)
        res = compute_residual(basis_ref, φr, occupation)

        # update list
        append!(norm_err_list, dm_distance(φr[1], φ_ref[1]))
        append!(norm_res_list, norm(res))
        if change_norm
            append!(norm_Pk_kin_err_list, dm_distance(φr[1], φ_ref[1], Pk_kin))
            append!(norm_Pk_kin_res_list, norm(apply_inv_sqrt(Pk_kin, res)))
            append!(norm_Pk_pot_err_list, dm_distance(φr[1], φ_ref[1], Pk_pot))
            append!(norm_Pk_pot_res_list, norm(apply_inv_sqrt(Pk_pot, res)))
        end

        # to compare conditioning when Ecut varies
        #  if change_norm
        #      Pk = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        #      for ik = 1:length(Pk)
        #          DFTK.precondprep!(Pk[ik], φ[ik])
        #      end
        #      normop, svd_min, svd_max = compute_normop_invΩ(basis, φ, occupation;
        #                                                     tol_krylov=tol_krylov, Pks=Pk)
        #      println("min_svd = ", svd_min)
        #      println("max_svd = ", svd_max)
        #      println("condition = ", svd_max/svd_min)
        #  end
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
        normop_invΩ_pot, svd_min_Ω_pot, svd_max_Ω_pot = compute_normop_invΩ(basis_ref, φ_ref, occupation;
                                                                            tol_krylov=tol_krylov, Pks=Pk_pot,
                                                                            change_norm=change_norm)
        if model == modelAtomic
            normop_invε = 1.0
            svd_min_ε = 1.0
            svd_max_ε = 1.0
        else
            normop_invε, svd_min_ε, svd_max_ε = compute_normop_invε(basis_ref, φ_ref, occupation;
                                                                    tol_krylov=tol_krylov, Pks=Pk_kin,
                                                                    change_norm=change_norm)
        end
        err_Pk_estimator = normop_invε .* normop_invΩ_kin .* norm_Pk_kin_res_list
        err_Pk_pot_estimator = normop_invε .* normop_invΩ_pot .* norm_Pk_pot_res_list
    end

    # plot
    #  M^1/2ε^-TM^-1/2 : norm = $(@sprintf("%.3f", normop_invε)), min_svd = $(@sprintf("%.3f", svd_min_ε)), max_svd = $(@sprintf("%.3f", svd_max_ε))
    figure()
    title("Silicon
          error estimators vs Ecut, Atomic, N = $(N), M = (T-Δ), gap = $(@sprintf("%.3f", gap)),
          (Ω+K)^-1 : norm = $(@sprintf("%.3f", normop_invΩpK)), min_svd = $(@sprintf("%.3f", svd_min_ΩpK)), max_svd = $(@sprintf("%.3f", svd_max_ΩpK))
          kin : M^1/2Ω^-1M^1/2 : norm = $(@sprintf("%.3f", normop_invΩ_kin)), min_svd = $(@sprintf("%.3f", svd_min_Ω_kin)), max_svd = $(@sprintf("%.3f", svd_max_Ω_kin))
          pot : M^1/2Ω^-1M^1/2 : norm = $(@sprintf("%.3f", normop_invΩ_pot)), min_svd = $(@sprintf("%.3f", svd_min_Ω_pot)), max_svd = $(@sprintf("%.3f", svd_max_Ω_pot))")
    semilogy(Ecut_list, norm_err_list, "x-", label="|P-Pref|")
    semilogy(Ecut_list, norm_res_list, "x-", label="|res|")
    semilogy(Ecut_list, err_estimator, "x-", label="|(Ω+K)^{-1}|*|res|")
    if change_norm
        semilogy(Ecut_list, norm_Pk_kin_err_list, "+--", label="|M^1/2(P-Pref)| kin")
        semilogy(Ecut_list, norm_Pk_kin_res_list, "+--", label="|M^-1/2res| kin")
        if model == modelAtomic
            semilogy(Ecut_list, err_Pk_estimator, "+--", label="|M^1/2Ω^-1M^1/2| * |M^-1/2res| kin")
        else
            semilogy(Ecut_list, err_Pk_estimator, "+--", label="|M^1/2(Ω+K)^-1M^-1/2| * |M^1/2Ω^-1M^1/2| * |M^-1/2res| kin")
        end
        semilogy(Ecut_list, norm_Pk_pot_err_list, "+:", label="|M^1/2(P-Pref)| pot")
        semilogy(Ecut_list, norm_Pk_pot_res_list, "+:", label="|M^-1/2res| pot")
        if model == modelAtomic
            semilogy(Ecut_list, err_Pk_pot_estimator, "+:", label="|M^1/2Ω^-1M^1/2| * |M^-1/2res| pot")
        else
            semilogy(Ecut_list, err_Pk_pot_estimator, "+:", label="|M^1/2(Ω+K)^-1M^-1/2| * |M^1/2Ω^-1M^1/2| * |M^-1/2res| pot")
        end
    end
    legend()
    xlabel("Ecut")

end
