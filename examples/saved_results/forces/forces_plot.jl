using DFTK
using StaticArrays
using JLD
using PyPlot

@load "./forces_rHF.jld"

# plotting
figure(figsize=(20,10))
subplot(121)
title("Local")
subplot(122)
title("Nonlocal")
pos = Dict("DFTK.TermAtomicLocal" => 121, "DFTK.TermAtomicNonlocal" => 122)
for key in key_list
    if (Fref[key] != nothing) && key != "DFTK.TermEwald"
        subplot(pos[key])
        k = 0
        for Ecut in Ecut_list
            k += 1
            toplot = F[key][k][1][1][1] - Fref[key][1][1][1]
            if toplot > 0
                plot(Ecut, toplot, "or", label="bigger than ref")
            else
                plot(Ecut, toplot, "ob", label="smaller than ref")
            end
        end
    end
end

savefig("forces_rHF.pdf")
