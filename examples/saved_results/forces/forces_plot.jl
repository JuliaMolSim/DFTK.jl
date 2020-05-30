using DFTK
using HDF5
using JLD
using PyPlot

h5open("forces_rHF.h5", "r") do file
    Ecut_list = read(file["Ecut_list"])
     key_list = read(file["key_list"])

    # plotting
    figure(figsize=(20,10))
    subplot(121)
    title("Local")
    subplot(122)
    title("Nonlocal")
    pos = Dict("DFTK.TermAtomicLocal" => 121, "DFTK.TermAtomicNonlocal" => 122)
    for key in key_list
        println(key)
        if key != "DFTK.TermEwald"
            subplot(pos[key])
            k = 0
            complete_plot = []
            F = read(file["F/$(key)"])
            Fref = read(file["Fref/$(key)"])
            for Ecut in Ecut_list
                k += 1
                toplot = F[1,1,k] - Fref[1,1]
                push!(complete_plot, abs(toplot))
                if toplot > 0
                    semilogy(Ecut, abs(toplot), "or", label="bigger than ref")
                else
                    semilogy(Ecut, abs(toplot), "ob", label="smaller than ref")
                end
            end
            semilogy(Ecut_list, complete_plot, "-", label=key)
        end
    end
    savefig("forces_rHF.pdf")
end
