# Very basic setup, useful for testing
using DFTK

include("perturbations.jl")
include("perturbations_tests.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
tol = 1e-12
nel = 8


# apply perturbations
s_list = 0:0.1:1
compute_forces = false
avg = true

err_list = []
err2_list = []
err3_list = []
err_rr_list = []

for s in s_list
    model = model_LDA(lattice, atoms, n_electrons=nel, scaling_factor=s)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut_ref = 60           # kinetic energy cutoff in Hartree
    kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)

    basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol),
                                      callback=info->nothing)

    Ecut = 15           # kinetic energy cutoff in Hartree
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
    scfres = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol),
                                  callback=info->nothing)

    Ep_fine, ψp_fine, ρp_fine, egvalp2, egvalp3, egvalp_rr, forcesp = perturbation(basis, kcoords, ksymops, scfres, 2.5*Ecut, compute_forces)


    # coarse basis error
    err = abs.(hcat((scfres_ref.eigenvalues .- scfres.eigenvalues)...))

    # second order egval
    err2 = abs.(hcat((scfres_ref.eigenvalues .- egvalp2)...))
    #  display(err .- err2)

    # third order egval
    err3 = abs.(hcat((scfres_ref.eigenvalues .- egvalp3)...))
    #  display(err .- err3)

    # third order egval
    err_rr = abs.(hcat((scfres_ref.eigenvalues .- egvalp_rr)...))
    #  display(err .- err_rr)

    global err2_list, err3_list, err_rr_list, err_list
    push!(err_list, err[1,1])
    push!(err2_list, err2[1,1])
    push!(err3_list, err3[1,1])
    push!(err_rr_list, err_rr[1,1])
end

figure(figsize=(10,10))
semilogy(s_list, err_list, label="coarse")
semilogy(s_list, err2_list, label="2")
semilogy(s_list, err3_list, label="3")
semilogy(s_list, err_rr_list, label="rr")
legend()
