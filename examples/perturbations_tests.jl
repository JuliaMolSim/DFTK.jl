# Tests functions for perturbation theory
# We test several value of the ratio Ecut_fine/Ecut, the relative gain, ...
#
# Perturbation routines are in perturbations.jl
#

using DataFrames
using GLM
using PyPlot

"""
Perturbation for several values of the ratio α = Ecut_fine/Ecut
"""
function test_perturbation_ratio(Ecut, Ecut_ref, α_max, compute_forces, file)
    """
    Ecut: coarse grid Ecut
    Ecut_ref: Ecut for the reference solution
    α_max: max ratio
    compute_forces: if true, compute forces for the reference, coarse grid and
    fine grid (highly increase computation time)
    """

    ### reference solution
    println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol))
    Etot_ref = sum(values(scfres_ref.energies))
    egval_ref = scfres_ref.eigenvalues
    ρ_ref = scfres_ref.ρ.fourier[1,2,3]
    if compute_forces
        forces_ref = forces(scfres_ref)
    end

    ### solution on a coarse grid to apply perturbation
    println("---------------------------\nSolution for Ecut = $(Ecut)")
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
    scfres = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
    Etot = sum(values(scfres.energies))

    ### lists to save data for plotting
    α_list = vcat(collect(1:0.1:3), collect(3.5:0.5:α_max))
    Ep_list = []
    E_fine_list = []
    egvalp2_list = []
    egvalp3_list = []
    egvalp_rr_list = []
    egval_fine_list = []
    ρp_list = []
    ρ_fine_list = []
    if compute_forces
        forcesp_list = []
        forces_fine_list = []
    end

    ### test perturbation for several ratio
    for α in α_list
        println("---------------------------\nEcut_fine = $(α) * Ecut")

        # full scf on basis_fine
        basis_fine = PlaneWaveBasis(model, α*Ecut, kcoords, ksymops)
        scfres_fine = self_consistent_field(basis_fine, tol=tol,
                                            is_converged=DFTK.ScfConvergenceDensity(tol))
        push!(E_fine_list, sum(values(scfres_fine.energies)))
        push!(egval_fine_list, scfres_fine.eigenvalues)
        push!(ρ_fine_list, scfres_fine.ρ.fourier[1,2,3])
        if compute_forces
            forces_fine = forces(scfres_fine)
            push!(forces_fine_list, forces_fine)
            display(forces_fine)
        end

        # perturbation
        Ep_fine, ψp_fine, ρp_fine, egvalp2, egvalp3, egvalp_rr, forcesp = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, compute_forces)
        push!(Ep_list, sum(values(Ep_fine)))
        push!(egvalp2_list, deepcopy(egvalp2))
        push!(egvalp3_list, deepcopy(egvalp3))
        push!(egvalp_rr_list, deepcopy(egvalp_rr))
        push!(ρp_list, ρp_fine.fourier[1,2,3])
        if compute_forces
            push!(forcesp_list, forcesp)
            display(forcesp)
        end

    end

    ### Plotting results
    figure(figsize=(20,20))
    tit = "Average shift : $(avg)
    Ne = $(Ne), kpts = $(length(basis.kpoints)), Ecut_ref = $(Ecut_ref), Ecut = $(Ecut)
    kcoords = $(kcoords)"
    suptitle(tit)

    # plot energy relative error
    subplot(221)
    title("Relative energy error for α = Ecut_fine/Ecut")
    error_list = abs.((Ep_list .- Etot_ref)/Etot_ref)
    error_fine_list = abs.((E_fine_list .- Etot_ref)/Etot_ref)
    semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
    semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
    semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
    xlabel("α")
    legend()

    # plot eigenvalue relative error
    subplot(222)
    title("Relative error on the first egval[1][1] for α = Ecut_fine/Ecut")
    egvalp211 = [egvalp2_list[i][1][1] for i in 1:length(α_list)]
    egvalp311 = [egvalp3_list[i][1][1] for i in 1:length(α_list)]
    egvalp_rr11 = [egvalp_rr_list[i][1][1] for i in 1:length(α_list)]
    egval_fine11 = [egval_fine_list[i][1][1] for i in 1:length(α_list)]
    egval11_ref = egval_ref[1][1]
    error1_list = abs.((egvalp211 .- egval11_ref)/egval11_ref)
    error2_list = abs.((egvalp311 .- egval11_ref)/egval11_ref)
    error_rr_list = abs.((egvalp_rr11 .- egval11_ref)/egval11_ref)
    error_fine_list = abs.((egval_fine11 .- egval11_ref)/egval11_ref)
    semilogy(α_list, error1_list, "-+", label = "perturbation from Ecut = $(Ecut), order 2")
    semilogy(α_list, error2_list, "-+", label = "perturbation from Ecut = $(Ecut), order 3")
    semilogy(α_list, error_rr_list, "-+", label = "perturbation from Ecut = $(Ecut)\nwith Rayleigh coef")
    semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
    semilogy(α_list, [error1_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
    xlabel("α")
    legend()

    # plot density error at one point in particular
    subplot(223)
    title("Error on the density in Fourier space")
    error_list = abs.(ρp_list .- ρ_ref)
    error_fine_list = abs.(ρ_fine_list .- ρ_ref)
    semilogy(α_list, error_list, "-+", label="perturbation from Ecut = $(Ecut)")
    semilogy(α_list, error_fine_list, "-+", label="full solution for Ecut_fine = α * Ecut")
    semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
    legend()

    if compute_forces
        #  plot forces error
        subplot(224)
        title("Error on the norm of the forces for α = Ecut_fine/Ecut")
        error_list = norm.([forcesp - forces_ref for forcesp in forcesp_list])
        error_fine_list = norm.([forces_fine - forces_ref for forces_fine in forces_fine_list])
        semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
        semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        legend()
    end

    savefig("first_order_perturbation_silicon_$(Ne)e_kpoints$(length(basis.kpoints))_Ecut_ref$(Ecut_ref)_Ecut$(Ecut)_avg$(avg)_$(file).pdf")
end

"""
Perturbation for several values of the coarse Ecut
"""
function test_perturbation_coarsegrid(α, Ecut_min, Ecut_max)
    """
    α: ratio to compute the fine grid
    Ecut_min, Ecut_max: interval for the different coarse grid
    """

    ### reference solution
    Ecut_ref = 3*Ecut_max
    println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
    scfres_ref = self_consistent_field(basis_ref, tol=1e-12)
    Etot_ref = sum(values(scfres_ref.energies))
    egval_ref = scfres_ref.eigenvalues

    Ecut_list = range(Ecut_min, Ecut_max, length=Int(Ecut_max/Ecut_min))
    Ep_list = []
    E_coarse_list = []
    for Ecut in Ecut_list
        println("---------------------------\nEcut = $(Ecut)")
        # full scf on coarse
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
        scfres = self_consistent_field(basis, tol=1e-12)
        push!(E_coarse_list, sum(values(scfres.energies)))

        # perturbation
        Ep_fine, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut)
        push!(Ep_list, sum(values(Ep_fine)))
    end

    ##### Plotting results
    figure(figsize=(20,10))
    tit = "Average shift : $(avg)
    Ne = $(Ne), kpts = $(length(kcoords)), Ecut_ref = $(Ecut_ref),
    kcoords = $(kcoords)"
    suptitle(tit)

    # size of the discretization grid
    N_list = sqrt.(2 .* Ecut_list)

    # plot energy error
    subplot(121)
    title("Difference with the reference energy Ecut = $(Ecut_ref)")
    error_list = abs.(Ep_list .- Etot_ref)
    error_coarse_list = abs.(E_coarse_list .- Etot_ref)
    semilogy(N_list, error_list, "-+", label = "perturbation")
    semilogy(N_list, error_coarse_list, "-+", label = "coarse grid")
    xlabel("Nc")
    legend()

    # plot energy relative error
    subplot(122)
    title("Relative error between perturbed and non-perturbed")
    error_list = abs.((Ep_list .- Etot_ref) ./ (E_coarse_list .- Etot_ref))
    loglog(N_list, error_list, "-+")
    xlabel("Nc")
    legend()

    # plot slope
    error_list_slope = error_list[7:12]
    Nc = N_list[7:12]
    data = DataFrame(X=log.(Nc), Y=log.(error_list_slope))
    ols = lm(@formula(Y ~ X), data)
    Nc_slope = N_list[5:13]
    slope = exp(coef(ols)[1]) .* Nc_slope .^ coef(ols)[2]
    loglog(Nc_slope, 1.5 .* slope, "--", label = "slope $(coef(ols)[2])")
    legend()

    savefig("first_order_perturbation_silicon_$(Ne)e_kpoints$(length(kcoords))_Ecut_ref$(Ecut_ref)_alpha$(α).pdf")

    ### Return results
    Ecut_list, N_list, Ep_list, E_coarse_list
end


