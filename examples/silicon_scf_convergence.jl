## Convergence tests on silicon for different lattice constants (higher close the gap)
## see Cancès/Kemlin/Levitt, Convergence analysis of direct minimization and self-consistent iterations, TODO arxiv link
## Some parameter values have been reduced for faster automatic testing

using DFTK
using PyPlot
using Printf

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 15 # 30 in the paper
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
tol = 1e-10
diagtol = 1e-12
maxiter = 20 # 100 in the paper; a high value is important to see the divergence of the simple damping algorithms

global resids = []
function my_callback(info)
    global resids
    neval = info.neval
    if neval == 1
        resids = []
    end
    err = norm(info.ρout.fourier - info.ρin.fourier)
    println(info.neval, " ", err)
    push!(resids, err)
end
my_isconverged=info -> norm(info.ρout.fourier - info.ρin.fourier) < tol
opts = (callback=my_callback, is_converged=my_isconverged, max_iter=maxiter, tol=tol, diagtol=diagtol)

global errs = []
global gaps = []
## is (10.26, 11.2, 11.405) in the paper
global as = (10.26, 11.405)
for a in as
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    model = model_LDA(lattice, atoms)
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    res = self_consistent_field(basis; opts...)
    gap = res.eigenvalues[1][5] - res.eigenvalues[1][4]
    errs_anderson = copy(resids)
    self_consistent_field(basis; solver=scf_damping_solver(1), opts...)
    errs_1 = copy(resids)
    self_consistent_field(basis; solver=scf_damping_solver(.5), opts...)
    errs_05 = copy(resids)
    self_consistent_field(basis; solver=scf_damping_solver(.2), opts...)
    errs_02 = copy(resids)
    self_consistent_field(basis; solver=scf_damping_solver(.1), opts...)
    errs_01 = copy(resids)
    global errs
    global gaps
    push!(errs, [errs_anderson, errs_1, errs_05, errs_02, errs_01])
    push!(gaps, gap)
end
# @save "res.jld" as errs gaps

for ia = 1:length(as)
    figure()
    semilogy(errs[ia][1], "-x", label="Anderson")
    semilogy(errs[ia][2], "-x", label="Damping β=1")
    semilogy(errs[ia][3], "-x", label="Damping β=0.5")
    semilogy(errs[ia][4], "-x", label="Damping β=0.2")
    semilogy(errs[ia][5], "-x", label="Damping β=0.1")
    xlabel("n")
    ylabel("residual")
    ylim([1e-10, 10])
    g = @sprintf "%1.1e" gaps[ia]
    title("a=$(as[ia])   gap=$g")
    legend(loc="lower right")
    PyPlot.savefig("silicon_a=$(as[ia]).pdf")
end
