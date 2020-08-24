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
function test_perturbation_ratio(Ecut, Ecut_ref, compute_forces)
    """
    Ecut: coarse grid Ecut
    Ecut_ref: Ecut for the reference solution
    compute_forces: if true, compute forces for the reference, coarse grid and
    fine grid (at the moment, highly increase computation time)
    """

    h5open(filename, "r+") do file

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
        E_p_list = []                    # perturbed energy
        E_fine_list = []                 # energy for full scf on the fine grid
        egval_p2_list = []               # 2nd order perturbed eigenvalues
        egval_p3_list = []               # 3rd order perturbed eigenvales
        egval_p_rr_list = []             # Rayleigh-Ritz egval with perturbed egvectors
        egval_fine_list = []             # eigenvalues for full scf on the fine grid
        ρ_p_list = []                    # norm (perturbed density - ref density)
        ρ_fine_list = []                 # norm (density on fine grid - ref density)
        if compute_forces
            forces_p_list = []           # perturbed forces
            forces_fine_list = []        # forces on the fine grid
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

            # interpolate ρ_ref to compare with ρ / ρ_p on the fine grid
            ρ_ref_fine = DFTK.interpolate_density(ρ_ref, basis_fine)
            push!(E_fine_list, sum(values(scfres_fine.energies)))
            push!(egval_fine_list, scfres_fine.eigenvalues)
            push!(ρ_fine_list, norm(scfres_fine.ρ.fourier - ρ_ref_fine.fourier))

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_ref_fourier"] = ρ_ref_fine.fourier
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_ref_real"] = ρ_ref_fine.real
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_fourier"] = scfres_fine.ρ.fourier
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_real"] = scfres_fine.ρ.real

            if compute_forces
                forces_fine = forces(scfres_fine)
                push!(forces_fine_list, Array.(forces_fine[1]))
            end

            # perturbation
            E_p, ψ_p, ρ_p, egval_p2, egval_p3, egval_p_rr, forces_p = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, compute_forces)

            # complete data for perturbation
            push!(E_p_list, sum(values(E_p)))
            push!(egval_p2_list, deepcopy(egval_p2))
            push!(egval_p3_list, deepcopy(egval_p3))
            push!(egval_p_rr_list, deepcopy(egval_p_rr))
            push!(ρ_p_list, norm(ρ_p.fourier - ρ_ref_fine.fourier))

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_fourier"] = ρ_p.fourier
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_real"] = ρ_p.real

            if compute_forces
                push!(forces_p_list, Array.(forces_p[1]))
            end

        end

        ### Plotting results and saving objects to HDF5 file

        # plot energy relative error
        error_list = abs.((E_p_list .- Etot_ref)/Etot_ref)
        error_fine_list = abs.((E_fine_list .- Etot_ref)/Etot_ref)

        # save data
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/error"] = error_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/error_fine"] = error_fine_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/E_fine"] = Float64.(E_fine_list)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/E_p"] = Float64.(E_p_list)

        # plot eigenvalue relative error
        egval_p211 = [egval_p2_list[i][1][1] for i in 1:length(α_list)]
        egval_p311 = [egval_p3_list[i][1][1] for i in 1:length(α_list)]
        egval_p_rr11 = [egval_p_rr_list[i][1][1] for i in 1:length(α_list)]
        egval_fine11 = [egval_fine_list[i][1][1] for i in 1:length(α_list)]
        egval11_ref = egval_ref[1][1]
        error1_list = abs.((egval_p211 .- egval11_ref)/egval11_ref)
        error2_list = abs.((egval_p311 .- egval11_ref)/egval11_ref)
        error_rr_list = abs.((egval_p_rr11 .- egval11_ref)/egval11_ref)
        error_fine_list = abs.((egval_fine11 .- egval11_ref)/egval11_ref)

        # save data
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/error2"] = error1_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/error3"] = error2_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/error_rr"] = error_rr_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/error_fine"] = error_fine_list
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/egval_ref"] = hcat(egval_ref...)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/egval_p2"] = reshape(hcat(hcat(egval_p2_list...)...), nel, nk, length(α_list))
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/egval_p3"] = reshape(hcat(hcat(egval_p3_list...)...), nel, nk, length(α_list))
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/egval_p_rr"] = reshape(hcat(hcat(egval_p_rr_list...)...), nel, nk, length(α_list))
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/egval/egval_fine"] = reshape(hcat(hcat(egval_fine_list...)...), nel, nk, length(α_list))


        # plot density error at one point in particular
        error_list = ρ_p_list
        error_fine_list = ρ_fine_list

        # save data
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error"] = Float64.(error_list)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error_fine"] = Float64.(error_fine_list)
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_fourier"] = ρ_ref.fourier
        file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_real"] = ρ_ref.real

        if compute_forces
            #  plot forces error
            error_list = norm.([forces_p - forces_ref for forces_p in forces_p_list])
            error_fine_list = norm.([forces_fine - forces_ref for forces_fine in forces_fine_list])

            # save data
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/error"] = error_list
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/error_fine"] = error_fine_list
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/forces_ref"] = hcat(forces_ref...)
            file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/forces_p"] = reshape(hcat(hcat(forces_p_list...)...), 3, 2, length(α_list))
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
    E_p_list = []
    E_coarse_list = []
    for Ecut in Ecut_list
        println("---------------------------\nEcut = $(Ecut)")
        # full scf on coarse
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
        scfres = self_consistent_field(basis, tol=1e-12)
        push!(E_coarse_list, sum(values(scfres.energies)))

        # perturbation
        E_p_fine, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut;
                                  compute_egval=false)
        push!(E_p_list, sum(values(E_p_fine)))
    end

    h5open(filename, "w") do file
        file["nk"] = length(kcoords)
        file["Ecut_ref"] = Ecut_ref
        file["Etot_ref"] = Etot_ref
        file["Ecut_list"] = Float64.(Ecut_list)
        file["E_p_list"] = Float64.(E_p_list)
        file["E_coarse_list"] = Float64.(E_coarse_list)
    end

    ### Return results
    Ecut_list, N_list, E_p_list, E_coarse_list
end


