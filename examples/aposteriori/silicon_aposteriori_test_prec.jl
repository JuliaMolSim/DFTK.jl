# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# when the error we look at is the basis error : P* is computed for a reference
# Ecut_ref and then we measure the error P-P* and the residual obtained for
# smaller Ecut (currently, only Nk = 1 kpt only is supported)
#
#            !!! NOT OPTIMIZED YET, WE USE PLAIN DENSITY MATRICES !!!
#

# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")
include("aposteriori_callback.jl")
include("newton.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# define different models
## nonlocal potential
model = model_atomic(lattice, atoms)
## local potential : beware, no gap between 1st and 2nd eigenvalue
#  model = Model(lattice; atoms=atoms,
#                      terms=[Kinetic(), AtomicLocal()],
#                      n_electrons=4)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-10
tol_krylov = 1e-12
Ecut_ref = 50           # kinetic energy cutoff in Hartree

println("--------------------------------")
println("reference computation")
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
Typ = typeof(scfres_ref.ρ.real[1])
## number of kpoints
Nk = length(basis_ref.kpoints)
## number of eigenvalue/eigenvectors we are looking for
filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
occupation = [filled_occ * ones(Typ, N) for ik = 1:Nk]
φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end
gap = scfres_ref.eigenvalues[1][N+1] - scfres_ref.eigenvalues[1][N]
println(gap)

## with non local potential
T_list_1 = 0.01:0.01:0.04
T_list_2 = 0.05:0.05:1
T_list_3 = 1.1:0.1:2
T_list = vcat(T_list_1, T_list_2, T_list_3)

## with local potential
#  T_list_1 = 1:1:3
#  T_list_2 = 3.2:0.2:6
#  T_list_3 = 7:1:10
#  T_list = vcat(T_list_1, T_list_2, T_list_3)


## svd lists
svd_min_list = []
svd_max_list = []
normop_list = []

for T in T_list
    println("--------------------------------")
    println("T = $(T)")

    Pk = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    for ik = 1:length(Pk)
        Pk[ik].mean_kin = [T for i in 1:N]
    end

    println("Computing operator norms...")
    normop_invΩ, svd_min_Ω, svd_max_Ω = compute_normop_invΩ(basis_ref, φ_ref, occupation;
                                                            tol_krylov=tol_krylov, Pks=Pk, change_norm=true)
    append!(normop_list, normop_invΩ)
    append!(svd_min_list, svd_min_Ω)
    append!(svd_max_list, svd_max_Ω)

end

println("--------------------------------")
println("Computing operator norms, T = meankin...")
Pk = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
for ik = 1:length(Pk)
    DFTK.precondprep!(Pk[ik], φ_ref[ik])
end
normop_invΩ_meankin, svd_min_Ω_meankin, svd_max_Ω_meankin = compute_normop_invΩ(basis_ref, φ_ref, occupation;
                                                                                tol_krylov=tol_krylov, Pks=Pk, change_norm=true)
mk = Pk[1].mean_kin
println("--------------------------------")
println("Computing operator norms, T = meanpot...")
function LinearAlgebra.Diagonal(opnl::DFTK.NonlocalOperator)
    [dot(p, opnl.D * p) for p in eachrow(opnl.P)]
end
Pk = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
for ik = 1:length(Pk)
    non_local_op = [op for op in scfres_ref.ham.blocks[ik].operators if (op isa DFTK.NonlocalOperator)][1]
    Pk[ik].mean_kin = [abs.(Diagonal(non_local_op)) .+ real(dot(x, Pk[ik].kin .* x)) for x in eachcol(φ_ref[ik])]
end
normop_invΩ_meanpot, svd_min_Ω_meanpot, svd_max_Ω_meanpot = compute_normop_invΩ(basis_ref, φ_ref, occupation;
                                                                                tol_krylov=tol_krylov, Pks=Pk, change_norm=true)

figure()
title("Comparing conditioning of M^-1/2ΩM^-1/2 for M = (T-Δ) as T varies
      mean_kin = $(mk)")
# T varies
plot(T_list, svd_max_list, "x:", label="max svd")
plot(T_list, svd_min_list, "x:", label="min svd")
plot(T_list, svd_max_list ./ svd_min_list, "x-", label="cond")
plot(T_list, normop_list, "x--", label="normop")

# T = mean_kin
plot(T_list, [svd_max_Ω_meankin/svd_min_Ω_meankin for T in T_list], "-", label="cond, T = mean_kin")
plot(T_list, [normop_invΩ_meankin for T in T_list], "--", label="normop, T = mean_kin")

#  # T = mean_pot
plot(T_list, [svd_max_Ω_meanpot/svd_min_Ω_meanpot for T in T_list], "-", label="cond, T = mean_pot")
plot(T_list, [normop_invΩ_meanpot for T in T_list], "--", label="normop, T = mean_pot")

xlabel("T")
legend()
