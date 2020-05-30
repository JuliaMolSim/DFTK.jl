using HDF5
using JLD
using PyPlot

h5open("perturbation_tests.h5") do file
    Ecut_ref = 30
    nk = 1
    α_list = read(file["alpha"])
    nel = read(file["nel"])
    Ecut_list = read(file["Ecut_list"])

    for Ecut in Ecut_list
        # create figure
        figure(figsize=(20,20))
        tit = "kpts = $(nk), Ecut_ref = $(Ecut_ref), Ecut = $(Ecut)"
        suptitle(tit)

        # plot energy relative error
        subplot(221)
        title("Relative energy error for α = Ecut_fine/Ecut")
        error_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/error"])
        semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        error_fine_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/energy/error_fine"])
        semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
        xlabel("α")
        legend()


        # plot eigenvalue relative error
        subplot(222)
        title("Relative error on the first egval[1][1] for α = Ecut_fine/Ecut")
        error2_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error2"])
        semilogy(α_list, error2_list, "-+", label = "perturbation from Ecut = $(Ecut), order 2")
        semilogy(α_list, [error2_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        error3_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error3"])
        semilogy(α_list, error3_list, "-+", label = "perturbation from Ecut = $(Ecut), order 3")
        error_rr_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error_rr"])
        semilogy(α_list, error_rr_list, "-+", label = "perturbation from Ecut = $(Ecut)\nwith Rayleigh coef")
        error_fine_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/error_fine"])
        semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
        xlabel("α")
        legend()

        # plot density error at one point in particular
        subplot(223)
        title("Error on the density in Fourier space")
        error_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error"])
        semilogy(α_list, error_list, "-+", label="perturbation from Ecut = $(Ecut)")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        error_fine_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error_fine"])
        semilogy(α_list, error_fine_list, "-+", label="full solution for Ecut_fine = α * Ecut")
        legend()


        # plot forces error
        subplot(224)
        title("Error on the norm of the forces for α = Ecut_fine/Ecut")
        error_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/error"])
        semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        error_fine_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/forces/error_fine"])
        semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
        legend()

        savefig("perturbations_Ecutref$(Ecut_ref)_Ecut$(Ecut)_nk$(nk).pdf")
    end
end
