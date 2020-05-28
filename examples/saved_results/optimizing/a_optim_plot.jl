using JLD
using PyPlot

@load "./optim_nonperturbed.jld"
@load "./optim_perturbed.jld"

figure(figsize=(20,10))

# time VS accuracy
subplot(121)
title("time vs accuracy")
xlabel("time (s)")
ylabel("error")
semilogy(time_list, err_list, "+-", label="Full SCF on fine grid")
semilogy(timep_list, errp_list, "x-", label="Perturbed SCF")
legend()

# Ecut VS accuracy
subplot(122)
title("Ecut vs accuracy")
xlabel("Ecut")
ylabel("error")
semilogy(Ecut_list, err_list, "+-", label="Full SCF for given Ecut")
semilogy(Ecutp_list, errp_list, "x-", label="Perturbed SCF with Efine = 2.5 * Ecut")
legend()

savefig("compare.pdf")
