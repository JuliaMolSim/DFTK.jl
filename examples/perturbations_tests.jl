# Tests functions for perturbation theory
# We test several value of the ratio Ecut_fine/Ecut, the relative gain, ...
#
# Perturbation routines are in perturbations.jl
#

using DataFrames
using GLM
using PyPlot
using HDF5

"""
Perturbation for several values of the ratio α = Ecut_fine/Ecut
"""
function test_perturbation_ratio(Ecut, Ecut_ref, α_max, compute_forces)
    """
    Ecut: coarse grid Ecut
    Ecut_ref: Ecut for the reference solution
    α_max: max ratio
    compute_forces: if true, compute forces for the reference, coarse grid and
    fine grid (at the moment, highly increase computation time)
    """

    h5open("perturbation_tests.h5", "r+") do file

        ### compute the reference solution
        println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
        basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
        scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                           is_converged=DFTK.ScfConvergenceDensity(tol))
        Etot_ref = sum(values(scfres_ref.energies))
        egval_ref = scfres_ref.eigenvalues
        ρ_ref = scfres_ref.ρ
        if compute_forces
            ff = forces(scfres_ref)
            forces_ref = Array.(ff[1])
        end

        ### solution on a coarse grid to which we want to apply perturbation
        println("---------------------------\nSolution for Ecut = $(Ecut)")
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
        scfres = self_consistent_field(basis, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol))
        Etot = sum(values(scfres.energies))

        ### lists to save data for plotting
        Ep_list = []                    # perturbed energy
        E_fine_list = []                # energy for full scf on the fine grid
        egvalp2_list = []               # 2nd order perturbed eigenvalues
        egvalp3_list = []               # 3rd order perturbed eigenvales
        egvalp_rr_list = []             # Rayleigh-Ritz egval with perturbed egvectors
        egval_fine_list = []            # eigenvalues for full scf on the fine grid
        ρp_list = []                    # norm (perturbed density - ref density)
        ρ_fine_list = []                # norm (density on fine grid - ref density)
        if compute_forces
            forcesp_list = []           # perturbed forces
            forces_fine_list = []       # forces on the fine grid
        end

        # number of kpoints
        nk = length(egval_ref)
        # number of eigenvalues computed at each kpoint
        nel = length(egval_ref[1])

        ### test perturbation for several ratio Ecut_fine / Ecut
        for α in α_list
            println("---------------------------\nEcut_fine = $(α) * Ecut")

            # full scf on basis_fine
            basis_fine = PlaneWaveBasis(model, α*Ecut, kcoords, ksymops)
            scfres_fine = self_consistent_field(basis_fine, tol=tol,
                                                is_converged=DFTK.ScfConvergenceDensity(tol))

            # interpolate ρ_ref to compare with ρ / ρp on the fine grid
            ρref_fine = DFTK.interpolate_density(ρ_ref, basis_fine)

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_ref_fourier"] = ρref_fine.fourier
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_ref_real"] = ρref_fine.real

            # complete data on fine grid
            push!(E_fine_list, sum(values(scfres_fine.energies)))
            push!(egval_fine_list, scfres_fine.eigenvalues)
            push!(ρ_fine_list, norm(scfres_fine.ρ.fourier - ρref_fine.fourier))

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_fourier"] = scfres_fine.ρ.fourier
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_real"] = scfres_fine.ρ.real

            if compute_forces
                forces_fine = forces(scfres_fine)
                push!(forces_fine_list, Array.(forces_fine[1]))
            end

            # perturbation
            Ep_fine, ψp_fine, ρp_fine, egvalp2, egvalp3, egvalp_rr, forcesp = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, compute_forces)

            # complete data for perturbation
            push!(Ep_list, sum(values(Ep_fine)))
            push!(egvalp2_list, deepcopy(egvalp2))
            push!(egvalp3_list, deepcopy(egvalp3))
            push!(egvalp_rr_list, deepcopy(egvalp_rr))
            push!(ρp_list, norm(ρp_fine.fourier - ρref_fine.fourier))

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_fourier"] = ρp_fine.fourier
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_real"] = ρp_fine.real

            if compute_forces
                push!(forcesp_list, Array.(forcesp[1]))
            end

        end

        ### Plotting results and saving objects to HDF5 file

        # plot energy relative error
        error_list = abs.((Ep_list .- Etot_ref)/Etot_ref)
        error_fine_list = abs.((E_fine_list .- Etot_ref)/Etot_ref)

        # save data
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/error"] = error_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/error_fine"] = error_fine_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/E_fine"] = Float64.(E_fine_list)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/Ep"] = Float64.(Ep_list)

        # plot eigenvalue relative error
        egvalp211 = [egvalp2_list[i][1][1] for i in 1:length(α_list)]
        egvalp311 = [egvalp3_list[i][1][1] for i in 1:length(α_list)]
        egvalp_rr11 = [egvalp_rr_list[i][1][1] for i in 1:length(α_list)]
        egval_fine11 = [egval_fine_list[i][1][1] for i in 1:length(α_list)]
        egval11_ref = egval_ref[1][1]
        error1_list = abs.((egvalp211 .- egval11_ref)/egval11_ref)
        error2_list = abs.((egvalp311 .- egval11_ref)/egval11_ref)
        error_rr_list = abs.((egvalp_rr11 .- egval11_ref)/egval11_ref)
        error_fine_list = abs.((egval_fine11 .- egval11_ref)/egval11_ref)

        # save data
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error2"] = error1_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error3"] = error2_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error_rr"] = error_rr_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error_fine"] = error_fine_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/egvalp2"] = reshape(hcat(hcat(egvalp2_list...)...), nel, nk, length(α_list))
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/egvalp3"] = reshape(hcat(hcat(egvalp3_list...)...), nel, nk, length(α_list))
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/egvalp_rr"] = reshape(hcat(hcat(egvalp_rr_list...)...), nel, nk, length(α_list))
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/egval_fine"] = reshape(hcat(hcat(egval_fine_list...)...), nel, nk, length(α_list))


        # plot density error at one point in particular
        error_list = ρp_list
        error_fine_list = ρ_fine_list

        # save data
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error"] = Float64.(error_list)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error_fine"] = Float64.(error_fine_list)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_fourier"] = ρ_ref.fourier
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_real"] = ρ_ref.real

        if compute_forces
            #  plot forces error
            error_list = norm.([forcesp - forces_ref for forcesp in forcesp_list])
            error_fine_list = norm.([forces_fine - forces_ref for forces_fine in forces_fine_list])

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/error"] = error_list
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/error_fine"] = error_fine_list
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/forces_ref"] = hcat(forces_ref...)
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/forcesp"] = reshape(hcat(hcat(forcesp_list...)...), 3, 2, length(α_list))
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/forces_fine"] = reshape(hcat(hcat(forces_fine_list...)...), 3, 2, length(α_list))
        end

    end
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


