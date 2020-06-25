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

model = model_LDA(lattice, atoms, n_electrons=nel)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 60           # kinetic energy cutoff in Hartree
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)

basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

Ecut = 15           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
scfres = self_consistent_field(basis, tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol))

# apply perturbations
α_list = 1:0.2:3
compute_forces = false
avg = true

err2_list = []
err3_list = []
err_rr_list = []
err_fine_list = []
err_norm_list = []
err_norm_fine_list = []
nrj_list = []
ρ_list = []

for α in α_list
    global Ecut
    Ep_fine, ψp_fine, ρp_fine, egvalp2, egvalp3, egvalp_rr, forcesp = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, compute_forces)


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

    # err norm
    global model, basis_ref, kcoords, ksymops, scfres_ref
    basis_fine = PlaneWaveBasis(model, α*Ecut, kcoords, ksymops)
    scfres_fine = self_consistent_field(basis_fine, tol=tol,
                                        is_converged=DFTK.ScfConvergenceDensity(tol),
                                        callback=info->nothing)
    ψ_fine = scfres_fine.ψ
    err_fine = abs.(hcat((scfres_ref.eigenvalues .- scfres_fine.eigenvalues)...))

    ψp_ref, _ = DFTK.interpolate_blochwave(ψp_fine, basis_fine, basis_ref)
    ψ_fine_ref, _ = DFTK.interpolate_blochwave(ψ_fine, basis_fine, basis_ref)
    kpt = basis_fine.kpoints[1]

    ψ_ref = scfres_ref.ψ
    occ = scfres_ref.occupation
    occ_bands = [n for n in 1:length(occ[1]) if occ[1][n] != 0.0]
    N = length(occ_bands)
    err_norm = sqrt(2*(N - sum(abs2, [dot(ψp_ref[1][:,i], ψ_ref[1][:,j])
                                for i in 1:N, j in 1:N])))
    err_norm_fine = sqrt(2*(N - sum(abs2, [dot(ψ_fine_ref[1][:,i], ψ_ref[1][:,j])
                                     for i in 1:N, j in 1:N])))

    # nrj
    global nrj_list
    nrj = abs(sum(values(Ep_fine)) - sum(values(scfres_ref.energies)))

    # nrj
    global ρ_list
    ρnorm = norm(ρp_fine.fourier - DFTK.interpolate_density(scfres_ref.ρ, basis_fine).fourier)

    global err2_list, err3_list, err_rr_list, err_norm_list
    global err_norm_fine_list, err_fine_list
    push!(err2_list, err2[1,1])
    push!(err3_list, err3[1,1])
    push!(err_rr_list, err_rr[1,1])
    push!(err_fine_list, err_fine[1,1])
    push!(err_norm_list, err_norm)
    push!(err_norm_fine_list, err_norm_fine)
    push!(nrj_list, nrj)
    push!(ρ_list, ρnorm)
end

display(hcat(scfres_ref.eigenvalues...))

figure(figsize=(10,10))
subplot(221)
semilogy(α_list, err2_list, label="2")
semilogy(α_list, err3_list, label="3")
semilogy(α_list, err_fine_list, label="fine")
semilogy(α_list, err_rr_list, label="rr")
legend()

subplot(222)
semilogy(α_list, err_norm_list, label="vep1")
semilogy(α_list, err_norm_fine_list, label="vep1 fine")
legend()

subplot(223)
semilogy(α_list, nrj_list, label="nrj")
legend()

subplot(224)
semilogy(α_list, ρ_list, label="density")
legend()

