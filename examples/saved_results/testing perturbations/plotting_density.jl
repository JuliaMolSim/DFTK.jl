using HDF5
using JLD
using PyPlot

h5open("perturbation_tests.h5") do file
    Ecut_ref = 120
    nk = 1
    α_list = read(file["alpha"])
    Ecut_list = read(file["Ecut_list"])

    for Ecut in Ecut_list
        # create figure
        figure(figsize=(20,20))
        tit = "kpts = $(nk), Ecut_ref = $(Ecut_ref), Ecut = $(Ecut)"

        subplot(221)
        title("Error on the density in Fourier space")
        error_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error"])
        semilogy(α_list, error_list, "-+", label="perturbation from Ecut = $(Ecut)")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        error_fine_list = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/norm_error_fine"])
        semilogy(α_list, error_fine_list, "-+", label="full solution for Ecut_fine = α * Ecut")
        legend()

        subplot(222)
        error_list = []
        error_fine_list = []
        ρref = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_fourier"])
        for α in α_list
            ρ = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_fourier"])
            ρp = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_fourier"])
            push!(error_fine_list, abs.(ρ[2,1,1] - ρref[2,1,1]))
            push!(error_list, abs.(ρp[2,1,1] - ρref[2,1,1]))
        end
        title("Error on the coefficient [2,1,1]")
        semilogy(α_list, error_list, "-+", label="full solution for Ecut_fine = α * Ecut")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        semilogy(α_list, error_fine_list, "-+", label="full solution for Ecut_fine = α * Ecut")
        legend()

        subplot(223)
        title("Density on the axis [:,1,1]")
        ρref = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_real"])[:,1,1]
        plot(range(0,1; length=length(ρref)), ρref, "r-", label="reference")
        for α in α_list
            ρ = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_real"])[:,1,1]
            ρp = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_real"])[:,1,1]
            plot(range(0,1; length=length(ρ)), ρ, "x-")
            plot(range(0,1; length=length(ρp)), ρp, "+-")
        end

        subplot(224)
        title("Density on the axis [:,1,1]")
        ρref = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/rho_ref_real"])[:,1,1]
        plot(range(0,1; length=length(ρref)), ρref, "r-", label="reference")
        for α in α_list
            ρ = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rho_real"])[:,1,1]
            ρp = read(file["Ecutref$(Ecut_ref)_nk$(nk)/Ecut$(Ecut)/density/alpha_$(α)/rhop_real"])[:,1,1]
            plot(range(0,1; length=length(ρ)), ρ, "x-")
            plot(range(0,1; length=length(ρp)), ρp, "+-")
        end

        savefig("density_Ecutref$(Ecut_ref)_Ecut$(Ecut)_nk$(nk).pdf")
    end
end
