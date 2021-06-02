# Computation of error estimate and corrections for the forces for the linear
# silicon system, in the form Ax=b
#
# Very basic setup, useful for testing
using DFTK
using HDF5
using PyPlot
using KrylovKit

include("aposteriori_forces.jl")
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8 + [0.42, 0.35, 0.24] ./ 50, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 60   # kinetic energy cutoff in Hartree
tol = 1e-10
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)

filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

## reference values
φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end
λ_ref = scfres_ref.eigenvalues[1][1]
f_ref = compute_forces(scfres_ref)

## min and max Ecuts for the two grid solution
Ecut_min = 5
Ecut_max = 40

Ecut_list = Ecut_min:5:Ecut_max
K = length(Ecut_list)
diff_list = zeros((K,K))
diff_list_res = zeros((K,K))
diff_list_err = zeros((K,K))

v = nothing
Mv = nothing
i = 0
j = 0

for Ecut_g in Ecut_list

    println("---------------------------")
    println("Ecut grossier = $(Ecut_g)")
    global i,j
    i += 1
    j = i
    basis_g = PlaneWaveBasis(model, Ecut_g; kgrid=kgrid)

    ## solve eigenvalue system
    scfres_g = self_consistent_field(basis_g, tol=tol,
                                     determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                     is_converged=DFTK.ScfConvergenceDensity(tol))
    ham = scfres_g.ham

    ## quantities
    φ = similar(scfres_g.ψ)
    for ik = 1:Nk
        φ[ik] = scfres_g.ψ[ik][:,1:N]
    end
    f_g = compute_forces(scfres_g)

    #  for Ecut_f in Ecut_g:5:Ecut_max
    for Ecut_f in [Ecut_ref]

        println("Ecut fin = $(Ecut_f)")
        # fine grid
        basis_f = PlaneWaveBasis(model, Ecut_f; kgrid=kgrid)

        # compute residual
        φr = DFTK.interpolate_blochwave(φ, basis_g, basis_f)
        res = compute_residual(basis_f, φr, occupation)
        err = compute_error(basis_f, φr, φ_ref)

        ## prepare Pks
        kpt = basis_f.kpoints[1]
        Pks = [PreconditionerTPA(basis_f, kpt) for kpt in basis_f.kpoints]
        t = nothing
        for ik = 1:length(Pks)
            DFTK.precondprep!(Pks[ik], φr[ik])
            t = Pks[ik].mean_kin[1]
        end

        #  Apply M^+-1/2
        Mres = apply_inv_sqrt_M(basis_f, φr, Pks, res)
        Merr = apply_sqrt_M(φr, Pks, err)

        f_res_nonloc = compute_forces_estimate(basis_f, Mres, φr, Pks, occupation; term="nonlocal")
        f_res_loc = compute_forces_estimate(basis_f, Mres, φr, Pks, occupation; term="local")
        f_err_nonloc = compute_forces_estimate(basis_f, Merr, φr, Pks, occupation; term="nonlocal")
        f_err_loc = compute_forces_estimate(basis_f, Merr, φr, Pks, occupation; term="local")

        diff_list[i,j] = abs(f_g[1][2][1]-f_ref[1][2][1])
        diff_list_err[i,j] = abs(f_err_nonloc[1][2][1]+f_err_loc[1][2][1])
        diff_list_res[i,j] = abs(f_res_nonloc[1][2][1]+f_res_loc[1][2][1])
        j += 1
    end
end

semilogy([diff_list[i,i] for i = 1:length(diff_list[1,:])], label="ref")
semilogy([diff_list_res[i,i] for i = 1:length(diff_list_res[1,:])], label="res")
semilogy([diff_list_err[i,i] for i = 1:length(diff_list_err[1,:])], label="err")
legend()

