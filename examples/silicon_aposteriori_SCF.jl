# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# when the error we look at is the SCF error : P* is computed as the converged
# density matrix and then we measure the error P-P* and the residual along the
# iterations. To this end, we defined a custom callback function
# (currently, only Nk = 1 kpt only is supported)
#
#            !!! NOT OPTIMIZED YET !!!
#

# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")
include("newton.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
#  atoms = [Si => [ones(3)/8 .- 1e-3, -ones(3)/8]]
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# define different models
n_el = 8
modelLDA = model_LDA(lattice, atoms, n_electrons=n_el)
modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()], n_electrons=n_el)
modelAtomic = model_atomic(lattice, atoms, n_electrons=n_el)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-15
tol_krylov = 1e-15
Ecut = 15           # kinetic energy cutoff in Hartree

## custom callback to follow estimators
function callback_estimators()

    global ite, φ_list, basis_list
    φ_list = []                 # list of φ^k
    basis_list = []
    ite = 0

    function callback(info)

        if info.stage == :finalize

            basis_ref = info.basis
            model = info.basis.model

            ## number of kpoints
            Nk = length(basis_ref.kpoints)
            ## number of eigenvalue/eigenvectors we are looking for
            filled_occ = DFTK.filled_occupation(model)
            N = div(model.n_electrons, filled_occ)
            occupation = [filled_occ * ones(N) for ik = 1:Nk]

            φ_ref = similar(info.ψ)
            for ik = 1:Nk
                φ_ref[ik] = info.ψ[ik][:,1:N]
            end

            ## converged values
            ρ_ref = info.ρ
            H_ref = info.ham
            egval_ref = info.eigenvalues
            T = typeof(ρ_ref.real[1])


            ## packing routines to pack vectors for KrylovKit solver
            packing = packing_routines(basis_ref, φ_ref)

            ## filling residuals and errors
            err_ref_list = []
            err_newton_list = []
            norm_δφ_list = []
            norm_res_list = []
            for i in 1:ite
                φ = φ_list[i]
                basis = basis_list[i]
                for ik = 1:Nk
                    φ[ik] = φ[ik][:,1:N]
                end

                res, ρ, H, egval = compute_residual(basis, φ, occupation;
                                                    φproj=φ)

                φ_newton, δφ = newton_step(basis, φ, res, ρ, H, egval,
                                           occupation, packing;
                                           φproj=φ, tol_krylov=tol_krylov)
                append!(err_ref_list, dm_distance(φ[1], φ_ref[1]))
                append!(err_newton_list, dm_distance(φ_newton[1], φ_ref[1]))
                append!(norm_δφ_list, norm(δφ))
                append!(norm_res_list, norm(res))
            end


            ## error estimates
            normop = compute_normop(basis_ref, φ_ref, ρ_ref, H_ref,
                                    egval_ref, occupation,
                                    packing; tol_krylov=tol_krylov)
            err_estimator = normop .* norm_res_list

            ## plotting convergence info
            figure()
            title("error estimators vs iteration for $(model_list[k]), N = $(N)")
            semilogy(1:ite, err_ref_list, "x-", label="|P-Pref|")
            semilogy(1:ite, norm_res_list, "x-", label="|res_φ|")
            semilogy(1:ite, err_estimator, "x-", label="|(Ω+K)|^-1 * |res_φ|")
            semilogy(1:ite, norm_δφ_list, "+-", label="|δφ|")
            semilogy(1:ite, err_newton_list, "x-", label="|P_newton-Pref|")
            legend()
            xlabel("iterations")

        else
            ite += 1
            append!(φ_list, [info.ψ])
            append!(basis_list, [info.basis])
        end
    end
    callback
end

#  model_list = ["Atomic", "Hartree", "LDA"]
model_list = ["LDA"]
k = 0
ite = nothing
φ_list = nothing
basis_list = nothing

#  for model in [modelAtomic, modelHartree, modelLDA]
for model in [modelLDA]
    println("--------------------------------")
    println("--------------------------------")
    global k
    k += 1
    println(model_list[k])

    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    scfres = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol),
                                   #  determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-12),
                                   callback=callback_estimators())
end
