using DFTK
using KrylovKit
using PyPlot
using Random
using PseudoArcLengthContinuation
const PALC = PseudoArcLengthContinuation
using Parameters, Setfield

# Calculation parameters
kgrid = [1, 1, 1]       # k-Point grid
Ecut = 1000 # energy cutoff in Hartree
C = 0
α = 2
lattice = [[1 0 0.]; [0 0 0]; [0 0 0]]
f(x) = .2*(x-a/2)^2 # potential
f(x) = cos(4π*x)
n_electrons = 1
# We add the needed terms
terms = [Kinetic(),
         ExternalFromReal(X -> f(X[1])),
         PowerNonlinearity(C, α),
         ]
model = Model(lattice; n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless)  # "spinless fermions"
basis = PlaneWaveBasis(model, Ecut)
xgrid = range(-1/2, 1/2, length=prod(basis.fft_size))

diagtol = 1e-8
toldf = diagtol
δ_findiff = toldf^(1/3)
tol_newton = 1e-6

function update_C!(basis, C)
    inonlinear = findfirst(t -> isa(t, DFTK.TermPowerNonlinearity), basis.terms)
    basis.terms[inonlinear] = DFTK.TermPowerNonlinearity(basis, C, α)
end

function F(ρ, p)
    ρr = reshape(ρ, basis.fft_size)
    update_C!(basis, p.C)
    E, ham = energy_hamiltonian(basis, nothing, nothing; ρ=from_real(basis, ρr))
    res = DFTK.next_density(ham, tol=diagtol)
    vec(res.ρ.real) - ρ
end
function dF(ρ, p)
    ρr = reshape(ρ, basis.fft_size)
    update_C!(basis, p.C)
    E, ham = energy_hamiltonian(basis, nothing, nothing; ρ=from_real(basis, ρr))
    res = DFTK.next_density(ham, tol=diagtol)
    function dFfun(dρ)
        dρr = reshape(dρ, basis.fft_size)
        # dV = DFTK.apply_xc_kernel(basis, res.ρ.real, dρr)
        dV = α*p.C*dρr
        ψ = [res.ψ[1][:, 1]]
        eig = [res.eigenvalues[1][1:1]]
        dρ_out = apply_χ0(ham, dV, ψ, res.εF, eig)
        vec(dρ_out - dρr)
    end

    dFfun
end
function dFt(ρ, p)
    ρr = reshape(ρ, basis.fft_size)
    update_C!(basis, p.C)
    E, ham = energy_hamiltonian(basis, nothing, nothing; ρ=from_real(basis, ρr))
    res = DFTK.next_density(ham, tol=diagtol)
    function dFtfun(dV)
        dVr = reshape(dV, basis.fft_size)
        dρ = apply_χ0(ham, dVr, res.ψ, res.εF, res.eigenvalues; cgtol=toldf)
        dV_out = α*p.C*dρ
        # dV_out = DFTK.apply_xc_kernel(basis, res.ρ.real, dρ)
        vec(dV_out - dVr)
    end

    dFtfun
end

# centered difference have an accuracy of ε^2 + tol/ε, where tol is the precision of f
# best is ε=tol^1/3, which has an accuracy of tol^2/3
function D(f, x, p, dx; ε=1e-6)
    (f(x + ε*dx, p) - f(x - ε*dx, p))/(2ε)
end
d2F(x,p,dx1,dx2) = D((z, p0) -> dF(z, p0)(dx1), x, p, dx2, ε=toldf^(1/3))
d3F(x,p,dx1,dx2, dx3) = D((z, p0) -> d2F(z, p0, dx1, dx2), x, p, dx3, ε=toldf^(2/9))

C_beg = 0.
C_end = -6.

opt_newton = PALC.NewtonPar(tol = tol_newton, maxIter = 20, eigsolver = EigArpack(v0=randn(prod(basis.fft_size))), linsolver=GMRESKrylovKit())
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.2, ds = -0.01, pMax = C_beg, pMin = C_end,
	                  detectBifurcation = 2, plotEveryNsteps = 10, newtonOptions = opt_newton,
	                  maxSteps = 100, precisionStability = 1e-6, nInversion = 4, dsminBisection = 1e-7, maxBisectionSteps = 25)

ρ0 = zeros(prod(basis.fft_size))
p0 = (; C=0.)
ρ0 = F(ρ0, p0)

sol_start, _, _ = newton( F, ρ0, p0, opt_newton)
printSolution = (x, p) -> sum(xgrid .* x)
plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...)
args = (printSolution=printSolution,
        plotSolution=plotSolution,
        plot=false,
        verbosity=3)

br, _ = @time PALC.continuation(
    F, dF, sol_start, p0, (@lens _.C), opts_br; args...)

branches = []
push!(branches, br)
for i = 1:2
    br2, _ = continuation(F, dF, d2F, d3F, br, i, setproperties(opts_br, ds = -0.01, detectBifurcation=1); Jt=dFt, δ=δ_findiff, args...)
    br3, _ = continuation(F, dF, d2F, d3F, br, i, setproperties(opts_br, ds = 0.01, detectBifurcation=1); Jt=dFt, δ=δ_findiff, args...)
    push!(branches, br2)
    push!(branches, br3)
end
Plots.plot([branches...])
# PALC.computeNormalForm1d(F, dF, d2F, d3F, br, 1; Jt=dFt, verbose = true, δ=1e-4)
