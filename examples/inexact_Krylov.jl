# test inexact GMRES for linear response
using ASEconvert
using DFTK
using JLD2
using LinearMaps
using LinearAlgebra
using Printf
using Dates
using Random
using ForwardDiff

disable_threading()

println("------ Setting up model ... ------")
repeat  = 2
mixing  = KerkerMixing()
tol     = 1e-12
Ecut    =15
kgrid   =(1, 3, 3)

system = ase.build.bulk("Al", cubic=true) * pytuple((repeat, 1, 1))
system = pyconvert(AbstractSystem, system)
system = attach_psp(system; Al="hgh/pbe/Al-q3")
model = model_PBE(system, temperature=0.001, symmetries=false)
basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
println(show(stdout, MIME("text/plain"), basis))
println("------ Running SCF ... ------")
DFTK.reset_timer!(DFTK.timer)
scfres = self_consistent_field(basis; tol=tol, mixing=mixing)
println(DFTK.timer)

println("------ Computing rhs ... ------")
ρ, ψ, ham, basis, occupation, εF, eigenvalues = scfres.ρ, scfres.ψ, scfres.ham, scfres.basis, scfres.occupation, scfres.εF, scfres.eigenvalues
num_kpoints = length(basis.kpoints)
positions = model.positions
lattice = model.lattice
atoms = model.atoms
R = [zeros(3) for pos in positions]
Random.seed!(1234)
for iR in 1:length(R)
    R[iR] = -ones(3) + 2 * rand(3)
end
function V1(ε)
    T = typeof(ε)
    pos = positions + ε * R
    modelV = Model(Matrix{T}(lattice), atoms, pos; model_name="potential",
        terms=[DFTK.AtomicLocal(), DFTK.AtomicNonlocal()], symmetries=false)
    basisV = PlaneWaveBasis(modelV; Ecut, kgrid)
    jambon = Hamiltonian(basisV)
    DFTK.total_local_potential(jambon)
end
δV = ForwardDiff.derivative(V1, 0.0)
println("||δV|| = ", norm(δV))
flush(stdout)
DFTK.reset_timer!(DFTK.timer)
δρ0 = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV; tol=1e-16)
println(DFTK.timer)
println("||δρ0|| = ", norm(δρ0))

# setup's for running inexact GMRES for linear response
adaptive=true # other options: "D10", "D100", "D10_n" for fixed CG tolerances
CG_tol_scale_choice="hdmd" # other options: "agr", "hdmd", "grt", with increasingly tighter CG tolerances (i.e., gives more accurate results)
precon = false
restart = 20
maxiter = 100
tol = 1e-9

apply_χ0_info = DFTK.get_apply_χ0_info(ham, ψ, occupation, εF, eigenvalues; CG_tol_type= (adaptive == true) ? CG_tol_scale_choice : "plain")
CG_tol_scale = apply_χ0_info.CG_tol_scale
Nocc_ks = [length(CG_tol_scale[ik]) for ik in 1:num_kpoints]
Nocc = sum(Nocc_ks)

# The actual computations are only several lines of code
# most of the code here is for printing, debugging, and saving intermediate results
# maybe they should be wrapped in chi0.jl and use a verbose flag to control printing
normδV_all = Float64[]
tol_sternheimer_all = Float64[]
CG_niters_all = Vector{Vector{Int64}}[]
CG_xs_all = Vector{Vector{Any}}[]

inds = [0, 1, 1] # i, ik, n

function sternheimer_callback(CG_niters, CG_xs)
    function callback(info)
        if inds[3] > Nocc_ks[inds[2]]
            inds[3] = 1
            inds[2] += 1
        end
        CG_niters[inds[2]][inds[3]] = info.res.n_iter
        push!(CG_xs[inds[2]], info.res.x)
        inds[3] += 1
    end
end

pack(δρ) = vec(δρ)
unpack(δρ) = reshape(δρ, size(ρ))

function operators_a(tol_sternheimer)
    function eps_fun(δρ)
        δρ = unpack(δρ)
        δV = apply_kernel(basis, δρ; ρ)

        inds[1] += 1
        inds[2:3] = [1, 1]
        push!(normδV_all, norm(DFTK.symmetrize_ρ(basis, δV)))
        if adaptive == true && CG_tol_scale_choice == "grt"
            tol_sternheimer = tol_sternheimer ./ (2*normδV_all[end])
        end
        push!(tol_sternheimer_all, tol_sternheimer)
        CG_niters = [zeros(Int64, Nocc_ks[i]) for i in 1:num_kpoints]
        CG_xs = [ [] for _ in 1:num_kpoints ]

        println("---- τ_CG's used for each Sternheimer equation (row) of each k-point (column) ----")
        τ_CG_table = [max.(0.5*eps(Float64), tol_sternheimer ./ CG_tol_scale[ik]) for ik in 1:num_kpoints]
        @printf("| %-7s ", "k-point")
        for n in 1:maximum(Nocc_ks)
            @printf("| %-8d ", n)
        end
        @printf("|\n")
        for (k, row) in enumerate(τ_CG_table)
            @printf("| %-7d ", k)
            for τ in row[1:end]
                @printf("| %-8.2e ", τ)
            end
            @printf("|\n")
        end
        @printf("| %-10s | %-10s | %-10s | %-10s |\n", "τ_i", "min τ_CG", "mean τ_CG", "max τ_CG")
        @printf("| %-10.3e | %-10.3e | %-10.3e | %-10.3e |\n\n", tol_sternheimer, minimum(reduce(vcat, τ_CG_table)), exp(sum(log.(reduce(vcat, τ_CG_table)))/Nocc), maximum(reduce(vcat, τ_CG_table)))
        flush(stdout)

        t1 = Dates.now()
        χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV; tol=tol_sternheimer, callback=sternheimer_callback(CG_niters,CG_xs), apply_χ0_info=apply_χ0_info)
        t2 = Dates.now()

        push!(CG_niters_all, CG_niters)
        push!(CG_xs_all, CG_xs)
        println("no. CG iters for each Sternheimer equation (row) of each k-point (column):")
        @printf("| %-7s ", "k-point")
        for n in 1:maximum(Nocc_ks)
            @printf("| %-3d ", n)
        end
        @printf("|\n")
        for (k, row) in enumerate(CG_niters)
            @printf("| %-7d ", k)
            for niters in row[1:end]
                @printf("| %-3d ", niters)
            end
            @printf("|\n")
        end
        @printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n", "min", "mean", "max", "sum", "total")
        @printf("| %-10d | %-10.3f | %-10d | %-10d | %-10d |\n\n", minimum(reduce(vcat, CG_niters)), sum(reduce(vcat, CG_niters)) / Nocc, maximum(reduce(vcat, CG_niters)), sum(reduce(vcat, CG_niters)), sum(sum.(sum(CG_niters_all))))
        println("χ0Time = ", canonicalize(t2 - t1), ", time now: ", Dates.format(t2, "yyyy-mm-dd HH:MM:SS"))
        flush(stdout)
        #println("ave CG iters = ", sum(reduce(vcat, CG_niters)) / Nocc)

        if precon
            return pack(DFTK.mix_density(mixing, basis, δρ - χ0δV))
        else
            return pack(δρ - χ0δV)
        end
    end
    return LinearMap(eps_fun, prod(size(δρ0)))
end

println("----- running GMRES: tol=", tol, ", restart=", restart, ", adaptive=", adaptive, ", CG_tol_scale_choice=", CG_tol_scale_choice, " -----")
Pδρ0 = δρ0
if precon
    Pδρ0 = DFTK.mix_density(mixing, basis, δρ0)
end
println("||Pδρ0|| = ", norm(Pδρ0))
# if the first argument of DFTK.gmres is a function, then each iteration the effective matrix is changed (here, due to inexactly computed mat-vec products)
# if the first argument of DFTK.gmres is a matrix, then the matrix is fixed
DFTK.reset_timer!(DFTK.timer)
if adaptive == "D10"
    results_a = DFTK.gmres(operators_a(tol / 10), pack(Pδρ0); restart=restart, tol=tol, verbose=1, debug=true, maxiter=maxiter)
elseif adaptive == "D10_n"
    results_a = DFTK.gmres(operators_a(tol / 10 / norm(Pδρ0)), pack(Pδρ0); restart=restart, tol=tol, verbose=1, debug=true, maxiter=maxiter)
elseif adaptive == "D100"
    results_a = DFTK.gmres(operators_a(tol / 100), pack(Pδρ0); restart=restart, tol=tol, verbose=1, debug=true, maxiter=maxiter)
elseif adaptive == true
    results_a = DFTK.gmres(operators_a, pack(Pδρ0); restart=restart, tol=tol, verbose=1, debug=true, maxiter=maxiter)
else
    error("Invalid adaptive choice")
end
println(DFTK.timer)


# define the "exact" application of the dielectric adjoint operator
# by using very tight CG tolerances
# this is also how `eps_fun` should have looked like after removing all the printing and debugging code...
function eps_fun_exact(δρ)
    normδρ = norm(δρ)
    δρ = δρ ./ normδρ
    δρ = unpack(δρ)
    δV = apply_kernel(basis, δρ; ρ)

    χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV; tol=1e-16)
    pack(δρ - χ0δV) .* normδρ
end

println("true residual = ", norm(eps_fun_exact(results_a.x[:, end]) - pack(δρ0)), "\n")