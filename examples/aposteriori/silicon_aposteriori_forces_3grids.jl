# Computation of error estimate and corrections for the forces for the linear
# silicon system, in the form Ax=b
#
# Very basic setup, useful for testing
import DFTK: apply_K, apply_Ω, newton_step, proj_tangent, pack_arrays, unpack_arrays, compute_scf_residual, OrthogonalizeAndProject
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
atoms = [Si => [ones(3)/8 + [0.42, 0.35, 0.24] ./ 20, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 200   # kinetic energy cutoff in Hartree
tol = 1e-10
tol_krylov = 1e-12
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
f_ref = compute_forces(scfres_ref)

## min and max Ecuts for the two grid solution
Ecut_min = 5
Ecut_max = 80

Ecut_list = Ecut_min:5:Ecut_max
K = length(Ecut_list)
diff_list = zeros((K,K))
diff_list_res = zeros((K,K))
diff_list_schur = zeros((K,K))
Mres_list = zeros(K)
Mschur_list = zeros(K)
Merr_list = zeros(K)

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

    for Ecut_f in Ecut_g:5:Ecut_max

        println("Ecut fin = $(Ecut_f)")
        # fine grid
        basis_f = PlaneWaveBasis(model, Ecut_f; kgrid=kgrid)
        # packing routine on fine grid
        pack(φ) = pack_arrays(basis_f, φ)
        unpack(x) = unpack_arrays(basis_f, x)
        packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

        # compute residual
        φr = DFTK.interpolate_blochwave(φ, basis_g, basis_f)
        res = compute_scf_residual(basis_f, φr, occupation)

        ## prepare Pks
        kpt = basis_f.kpoints[1]
        Pks = [PreconditionerTPA(basis_f, kpt) for kpt in basis_f.kpoints]
        for ik = 1:length(Pks)
            DFTK.precondprep!(Pks[ik], φr[ik])
        end

        ## compute error on LF with Schur
        ρr = compute_density(basis_f, φr, occupation)
        E, ham_f = energy_hamiltonian(basis_f, φr, occupation; ρ=ρr)
        function f(x)
            x = unpack(x)
            x = keep_LF(x, basis_f, Ecut_g)
            x = proj_tangent(x, φr)
            Kx = apply_K(basis_f, x, φr, ρr, occupation)
            Ωx = apply_Ω(basis_f, x, φr, ham_f)
            x = proj_tangent(Kx .+ Ωx, φr)
            x = keep_LF(x, basis_f, Ecut_g)
            pack(x)
        end


        err = compute_error(basis_f, φr, φ_ref)
        Merr = apply_sqrt_M(φr, Pks, err)

        resLF = keep_LF(res, basis_f, Ecut_g)
        resHF = keep_HF(res, basis_f, Ecut_g)
        resHF = apply_inv_T(Pks, resHF)
        ΩpKres = apply_Ω(basis_f, resHF, φr, ham_f) .+ apply_K(basis_f, resHF, φr, ρr, occupation)
        ΩpKresLF = keep_LF(ΩpKres, basis_f, Ecut_g)
        eLF, info = linsolve(f, pack(proj_tangent(resLF-ΩpKresLF, φr)), tol=1e-14;
                             orth=OrthogonalizeAndProject(packed_proj, pack(φr)))

        # Apply M^+-1/2
        MeLF = apply_sqrt_M(φr, Pks, unpack(eLF))
        Mres = apply_inv_sqrt_M(basis_f, φr, Pks, res)
        # only 1 kpt for the moment
        Mschur = [Mres[1] + MeLF[1]]

        # plot carots
        #  G_energies = DFTK.G_vectors_cart(basis_f.kpoints[1])
        #  normG = norm.(G_energies)
        #  println(size(Merr[1]))
        #  println(size(Merr[1][sortperm(normG)]))
        #  STOP
        #  figure(i)
        #  title("Ecut_g = $(Ecut_g)")
        #  plot(Merr[1][sortperm(normG)], label="Merr")
        #  plot(Mschur[1][sortperm(normG)], label="Mres_schur")
        #  plot(Mres[1][sortperm(normG)], label="Mres")
        #  xlabel("index of G by increasing norm")
        #  legend()

        #  figure(10+i)
        #  plot(res[1][sortperm(normG)], label="Mres")
        #  plot(err[1][sortperm(normG)], label="Merr")
        #  xlabel("index of G by increasing norm")
        #  legend()

        # approximate forces f-f*
        f_res = compute_forces_estimate(basis_f, Mres, φr, Pks, occupation)
        f_schur = compute_forces_estimate(basis_f, Mschur, φr, Pks, occupation)

        diff_list[i,j] = abs(f_g[1][2][1]-f_ref[1][2][1])
        diff_list_res[i,j] = abs(f_res[1][2][1])
        diff_list_schur[i,j] = abs(f_schur[1][2][1])
        j += 1
    end
end

h5open("3grids_forces.h5", "w") do file
    println("writing h5 file")
    file["Ecut_ref"] = Ecut_ref
    file["Ecut_list"] = collect(Ecut_list)
    file["diff_list"] = diff_list
    file["diff_list_res"] = diff_list_res
    file["diff_list_schur"] = diff_list_schur
end

#  figure()
#  semilogy(Ecut_list, [diff_list[i,i] for i in 1:length(Ecut_list)], label="F-F*")
#  semilogy(Ecut_list, [diff_list_res[i,i] for i in 1:length(Ecut_list)], label="Fres")
#  semilogy(Ecut_list, [diff_list_schur[i,i] for i in 1:length(Ecut_list)], label="Fschur")
#  legend()

#  figure()
#  semilogy(Ecut_list, Mres_list, label="Mres")
#  semilogy(Ecut_list, Merr_list, label="Merr")
#  legend()
