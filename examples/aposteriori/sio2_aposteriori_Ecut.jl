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
# smaller Ecut (currently, only Nk = 1 kpt only is supported)
#

using DFTK
using LinearAlgebra
using PyPlot

# import aux files
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

# Lattice parameters for α-quartz from
# https://www.materialsproject.org/materials/mp-549166/
# All lengths in Ångström (used by ASE)
a = b = 4.98530291
c = 5.47852971
α = β = 90
γ = 120

silicon = [  # Fractional coordinates
           [0.000000  0.471138  0.833333],
           [0.471138  0.000000  0.166666],
           [0.528862  0.528862  0.500000],
          ]
oxygen = [  # Fractional coordinates
          [0.148541  0.734505  0.046133],
          [0.265495  0.414036  0.712800],
          [0.585964  0.851459  0.379466],
          [0.734505  0.148541  0.953867],
          [0.414036  0.265495  0.287200],
          [0.851459  0.585964  0.620534],
         ]
symbols = append!(fill("Si", 3), fill("O", 6))
atoms_ase = pyimport("ase").Atoms(symbols=symbols, cell=[a, b, c, α, β, γ], pbc=true,
                                  scaled_positions=vcat(vcat(silicon, oxygen)...))

atoms = load_atoms(atoms_ase)
atoms = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="pbe")) => position
         for (el, position) in atoms]

lattice = load_lattice(atoms_ase)

## local potential only
#  model = model_atomic(lattice, atoms)
model = model_LDA(lattice, atoms)

kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-10
tol_krylov = 1e-12
Ecut_ref = 55           # kinetic energy cutoff in Hartree
Ecut_list = 10:5:(Ecut_ref-5)

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
    normop_invε_kin, svd_min_ε_kin, svd_max_ε_kin = compute_normop_invε(basis_ref, φ_ref, occupation;
                                                                        tol_krylov=tol_krylov, Pks=Pk_kin,
                                                                        change_norm=change_norm)
    err_Pk_estimator = normop_invε_kin .* normop_invΩ_kin .* norm_Pk_kin_res_list
end

h5open("sio2_Ecut_nl.h5", "w") do file
    file["Ecut_list"] = collect(Ecut_list)
    file["kgrid"] = kgrid
    file["N"] = N
    file["gap"] = gap
    file["normop_invΩpK"] = normop_invΩpK
    file["svd_min_ΩpK"] = svd_min_ΩpK
    file["svd_max_ΩpK"] = svd_max_ΩpK
    file["normop_invΩ_kin"] = normop_invΩ_kin
    file["svd_min_Ω_kin"] = svd_min_Ω_kin
    file["svd_max_Ω_kin"] = svd_max_Ω_kin
    file["normop_invε_kin"] = normop_invε_kin
    file["svd_min_ε_kin"] = svd_min_ε_kin
    file["svd_max_ε_kin"] = svd_max_ε_kin
    file["norm_err_list"] = Float64.(norm_err_list)
    file["norm_res_list"] = Float64.(norm_res_list)
    file["err_estimator"] = Float64.(err_estimator)
    file["norm_Pk_kin_err_list"] = Float64.(norm_Pk_kin_err_list)
    file["norm_Pk_kin_res_list"] = Float64.(norm_Pk_kin_res_list)
    file["err_Pk_estimator"] = Float64.(err_Pk_estimator)
end

