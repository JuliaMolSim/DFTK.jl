# Very basic setup, useful for testing
using DFTK
using PyPlot

include("aposteriori_forces.jl")
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8 + [0.42, 0.35, 0.24] ./ 10, -ones(3)/8]]

model = Model(lattice; atoms=atoms, n_electrons=2,
              terms=[Kinetic(), AtomicLocal()])
nl = false
#  model = model_LDA(lattice, atoms; n_electrons=2)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 80           # kinetic energy cutoff in Hartree
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

φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end

f_ref = forces(scfres_ref)

Ecut_list = 50:5:Ecut_ref-30
forces_list = []
estimator_forces_list_err = []
estimator_forces_list_res = []
estimator_forces_list_res_N1 = []
estimator_forces_list_res_N2 = []

estimator_forces_list_err_HF = []
estimator_forces_list_res_HF = []

estimator_forces_list_err_φp = []
estimator_forces_list_err_φ  = []
estimator_forces_list_res_φp = []
estimator_forces_list_res_φ = []

ratio_list = []
ratio_list_N1 = []
ratio_list_N2 = []

v = nothing
CS_list_err = []
CS_list_res = []
CS_list_HF_err = []
CS_list_HF_res = []
CS_list_φp_err = []
CS_list_φp_res = []

for Ecut in Ecut_list

    println("--------------------------")
    println("Ecut = $(Ecut)")

    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

    ## explicit computation of f-f*
    f = forces(scfres)
    append!(forces_list, norm(f[1][1]-f_ref[1][1]))

    ## f_est ~ |f-f*|
    # compute residual
    φ = similar(scfres.ψ)
    for ik = 1:Nk
        φ[ik] = scfres.ψ[ik][:,1:N]
    end
    φr = DFTK.interpolate_blochwave(φ, basis, basis_ref)
    res = compute_residual(basis_ref, φr, occupation)
    err = compute_error(basis_ref, φr, φ_ref)

    ## prepare Pks
    kpt = basis_ref.kpoints[1]
    Pks = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], φr[ik])
    end

    # compute residual with one Neumann step
    V = DFTK.total_local_potential(scfres_ref.ham)
    Mres = apply_inv_M(basis_ref, φr, Pks, res)
    Mres_real = G_to_r(basis_ref, kpt, Mres[1][:,1])
    VMres_real = (V .- scfres.eigenvalues[1][1] .- Pks[1].mean_kin[1]) .* Mres_real
    VMres = r_to_G(basis_ref, kpt, VMres_real)
    res_N1 = [res[1] .- VMres]

    # compute residual with two Neumann steps
    MVMres = apply_inv_M(basis_ref, φr, Pks, [VMres])
    MVMres_real = G_to_r(basis_ref, kpt, MVMres[1][:,1])
    VMVMres_real = (V .- scfres.eigenvalues[1][1] .- Pks[1].mean_kin[1]) .* MVMres_real
    VMVMres = r_to_G(basis_ref, kpt, VMVMres_real)
    res_N2 = [res_N1[1] .+ VMVMres]

    # Apply M^+-1/2
    Mres = apply_inv_sqrt_M(basis_ref, φr, Pks, res)
    Mres_N1 = apply_inv_sqrt_M(basis_ref, φr, Pks, res_N1)
    Mres_N2 = apply_inv_sqrt_M(basis_ref, φr, Pks, res_N2)
    Merr = apply_sqrt_M(φr, Pks, err)

    f_err, cs_err = compute_forces_estimate(basis_ref, Merr, φr, Pks)
    f_res, cs_res = compute_forces_estimate(basis_ref, Mres, φr, Pks)
    f_res_N1, cs_res_N1 = compute_forces_estimate(basis_ref, Mres_N1, φr, Pks)
    f_res_N2, cs_res_N2 = compute_forces_estimate(basis_ref, Mres_N2, φr, Pks)
    figure()
    plot([f_ref[1][2][1] - f[1][2][1] for i in cs_res], "k-", label="f_ref")
    plot(cumsum(cs_err), "r-", label="cs err")
    plot([f_err[1][2][1] for i in cs_err], "r--", label="f_err")
    plot(cumsum(cs_res), "g-", label="cs res")
    plot([f_res[1][2][1] for i in cs_res], "g--", label="f_res")
    plot(cumsum(cs_res_N1), "b-", label="cs res_N1")
    plot([f_res_N1[1][2][1] for i in cs_res], "b--", label="f_res_N1")
    plot(cumsum(cs_res_N2), "y-", label="cs res_N2")
    plot([f_res_N2[1][2][1] for i in cs_res], "y--", label="f_res_N2")
    legend()
    STOP

    f_err = norm(compute_forces_estimate(basis_ref, Merr, φr, Pks)[1][1])
    f_res = norm(compute_forces_estimate(basis_ref, Mres, φr, Pks)[1][1])
    f_res_N1 = norm(compute_forces_estimate(basis_ref, Mres_N1, φr, Pks)[1][1])
    f_res_N2 = norm(compute_forces_estimate(basis_ref, Mres_N2, φr, Pks)[1][1])

    append!(estimator_forces_list_err, f_err)
    append!(estimator_forces_list_res, f_res)
    append!(estimator_forces_list_res_N1, f_res_N1)
    append!(estimator_forces_list_res_N2, f_res_N2)

    # HF test
    Merr_HF = keep_HF(Merr, basis_ref, Ecut)
    Mres_HF = keep_HF(Mres, basis_ref, Ecut)
    f_err_HF = norm(compute_forces_estimate(basis_ref, Merr_HF, φr, Pks)[1][1])
    f_res_HF = norm(compute_forces_estimate(basis_ref, Mres_HF, φr, Pks)[1][1])

    append!(estimator_forces_list_err_HF, f_err_HF)
    append!(estimator_forces_list_res_HF, f_res_HF)

    # φ^⟂ test
    Merr_φp = proj(Merr, φr)
    Mres_φp = proj(Mres, φr)
    Merr_φ = Merr .- Merr_φp
    Mres_φ = Mres .- Mres_φp
    f_err_φp = norm(compute_forces_estimate(basis_ref, Merr_φp, φr, Pks)[1][1])
    f_err_φ = norm(compute_forces_estimate(basis_ref, Merr_φ, φr, Pks)[1][1])
    f_res_φp = norm(compute_forces_estimate(basis_ref, Mres_φp, φr, Pks)[1][1])
    f_res_φ = norm(compute_forces_estimate(basis_ref, Mres_φ, φr, Pks)[1][1])

    append!(estimator_forces_list_err_φp, f_err_φp)
    append!(estimator_forces_list_err_φ, f_err_φ)
    append!(estimator_forces_list_res_φp, f_res_φp)
    append!(estimator_forces_list_res_φ, f_res_φ)

    # ratios
    append!(ratio_list, norm(Merr)/norm(Mres))
    append!(ratio_list_N1, norm(Merr)/norm(Mres_N1))
    append!(ratio_list_N2, norm(Merr)/norm(Mres_N2))

    # CS
    global v
    v_HF = keep_HF([v], basis_ref, Ecut)[1]
    v_φp = proj([v], φr)[1]
    append!(CS_list_err, norm(v)*norm(Merr[1]) .* 4 / sqrt(basis.model.unit_cell_volume))
    append!(CS_list_res, norm(v)*norm(Mres[1]) .* 4 / sqrt(basis.model.unit_cell_volume))
    append!(CS_list_HF_err, norm(v_HF)*norm(Merr_HF[1]) .* 4 / sqrt(basis.model.unit_cell_volume))
    append!(CS_list_HF_res, norm(v_HF)*norm(Mres_HF[1]) .* 4 / sqrt(basis.model.unit_cell_volume))
    append!(CS_list_φp_err, norm(v_φp)*norm(Merr_φp[1]) .* 4 / sqrt(basis.model.unit_cell_volume))
    append!(CS_list_φp_res, norm(v_φp)*norm(Mres_φp[1]) .* 4 / sqrt(basis.model.unit_cell_volume))
end

figure(1)
rc("font", size=16)
title("Ecutref=$(Ecut_ref)")
semilogy(Ecut_list, forces_list, "r", label = "|F - F*|")
semilogy(Ecut_list, estimator_forces_list_err, "b", label = "estimator Merr")
semilogy(Ecut_list, estimator_forces_list_res, "b--", label = "estimator Mres")
semilogy(Ecut_list, estimator_forces_list_res_N1, "g--", label = "estimator Mres N1")
semilogy(Ecut_list, estimator_forces_list_res_N2, "g:", label = "estimator Mres N2")
xlabel("Ecut")
legend()

figure(2)
rc("font", size=16)
title("Ecutref=$(Ecut_ref), HF")
semilogy(Ecut_list, forces_list, "r", label = "|F - F*|")
semilogy(Ecut_list, estimator_forces_list_err, "g", label = "estimator Merr")
semilogy(Ecut_list, estimator_forces_list_res, "b", label = "estimator Mres")
semilogy(Ecut_list, estimator_forces_list_err_HF, "gx--", label = "estimator Merr HF")
semilogy(Ecut_list, estimator_forces_list_res_HF, "bx--", label = "estimator Mres HF")
xlabel("Ecut")
legend()

figure(3)
rc("font", size=16)
title("Ecutref=$(Ecut_ref), HF")
semilogy(Ecut_list, forces_list, "r", label = "|F - F*|")
semilogy(Ecut_list, estimator_forces_list_err, "g", label = "estimator Merr")
semilogy(Ecut_list, estimator_forces_list_res, "b", label = "estimator Mres")
semilogy(Ecut_list, estimator_forces_list_err_φp, "gx--", label = "estimator Merr φp")
semilogy(Ecut_list, estimator_forces_list_err_φ, "gx:", label = "estimator Merr φ")
semilogy(Ecut_list, estimator_forces_list_res_φp, "bx--", label = "estimator Mres φp")
semilogy(Ecut_list, estimator_forces_list_res_φ, "bx:", label = "estimator Mres φ")
xlabel("Ecut")
legend()

figure(4)
rc("font", size=16)
title("Ecutref=$(Ecut_ref), HF")
semilogy(Ecut_list, ratio_list, "g", label = "ratio")
semilogy(Ecut_list, ratio_list_N1, "g--", label = "ratio_N1")
semilogy(Ecut_list, ratio_list_N2, "g:", label = "ratio_N2")
xlabel("Ecut")
legend()

figure(5)
rc("font", size=16)
title("Ecutref=$(Ecut_ref), Cauchy-Schwarz")
semilogy(Ecut_list, forces_list, "r", label = "|F - F*|")
semilogy(Ecut_list, CS_list_err, "g", label = "CS err")
semilogy(Ecut_list, CS_list_res, "b", label = "CS res")
semilogy(Ecut_list, CS_list_HF_err, "g--", label = "CS err HF")
semilogy(Ecut_list, CS_list_HF_res, "b--", label = "CS res HF")
semilogy(Ecut_list, CS_list_φp_err, "g:x", label = "CS err φp")
semilogy(Ecut_list, CS_list_φp_res, "b:x", label = "CS res φp")
xlabel("Ecut")
legend()

#  h5open("silicon_Ecut_lin_forces.h5", "w") do file
#      file["Ecut_list"] = collect(Ecut_list)
#      file["Ecut_ref"] = Ecut_ref
#      file["kgrid"] = kgrid
#      file["forces_list"] = Float64.(forces_list)
#      file["estimator_forces_list"] = Float64.(estimator_forces_list)
#      file["estimator_res_Neuman_list"] = Float64.(res_list)
#      file["ratio_list"] = Float64.(ratio_list)
#      file["cos_ratio_list"] = Float64.(cos_ratio_list)
#      file["N"] = N
#      file["nl"] = nl
#  end



