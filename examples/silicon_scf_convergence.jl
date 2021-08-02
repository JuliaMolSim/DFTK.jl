## Convergence tests on silicon for different lattice constants (higher close the gap)
## see Cancès/Kemlin/Levitt, Convergence analysis of direct minimization and self-consistent iterations, https://arxiv.org/abs/2004.09088
## Some parameter values have been reduced for faster automatic testing

using DFTK
using Plots
using Printf
using LinearAlgebra

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 15  # 30 in the paper
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
tol = 1e-10
diagtol = 1e-12
# 100 in the paper; a high value is important to see the divergence of the simple damping algorithms
# Very reduced here for fast automated testing
maxiter = 5

global resids = []
function my_callback(info)
    info.stage == :finalize && return
    global resids
    info.n_iter == 1 && (resids = [])
    err = norm(info.ρout - info.ρin)
    println(info.n_iter, " ", err)
    push!(resids, err)
end
my_isconverged = info -> norm(info.ρout - info.ρin) < tol
opts = (callback=my_callback, is_converged=my_isconverged, maxiter=maxiter, tol=tol,
        determine_diagtol=info -> diagtol)

global errs = []
global gaps = []
## is (10.26, 11.2, 11.405) in the paper
global as = (10.26, 11.405)
for a in as
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    model = model_LDA(lattice, atoms)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
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
    p = plot(errs[ia][1], label="Anderson", m=true, yaxis=:log, reuse=false)
    plot!(errs[ia][2], label="Damping β=1", m=true, yaxis=:log)
    plot!(errs[ia][3], label="Damping β=0.5", m=true, yaxis=:log)
    plot!(errs[ia][4], label="Damping β=0.2", m=true, yaxis=:log)
    plot!(errs[ia][5], label="Damping β=0.1", m=true, yaxis=:log)
    yaxis!("residual")
    xaxis!("n")
    g = @sprintf "%1.1e" gaps[ia]
    title!("a=$(as[ia])   gap=$g")
    ylims!(1e-10, 10)
    display(p)
end
