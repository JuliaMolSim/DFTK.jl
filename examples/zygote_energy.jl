using DFTK
using Zygote
import FiniteDiff

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
# model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn]) # xc not working yet (mutating)
# terms = [
#     Kinetic(),
#     AtomicLocal(),
#     AtomicNonlocal(),
#     Ewald(),
#     PspCorrection(),
#     Entropy(),
#     Hartree()
# ]
# model = Model(lattice; atoms=atoms, terms=terms, symmetries=false)
model = model_DFT(lattice, atoms, [], symmetries=false)
kgrid = [1, 1, 1]
Ecut = 7
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-13))
ψ = scfres.ψ
occupation = scfres.occupation

function total_energy(ρ)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
    sum(values(getfield(energies, :energies)))
end

total_energy(scfres.ρ)
# Zygote.gradient(total_energy, scfres.ρ) # ERROR

# Zygote.gradient(x -> sum(values(Dict(1=>x))), 1.0) # ERROR
# using OrderedCollections
# Zygote.gradient(x -> sum(values(OrderedDict{Int,Float64}(1=>x))), 1.0) ERROR
#
# # Zygote has issues with Dicts?
# https://github.com/FluxML/Zygote.jl/issues/760

# Approach 2: Try direct access to term energies

function total_energy2(ρ)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
total_energy2(scfres.ρ)
g1 = Zygote.gradient(total_energy2, scfres.ρ)[1] # works

# check against finite differences
g2 = FiniteDiff.finite_difference_gradient(total_energy2, scfres.ρ)
sum(abs, g1 - g2)  # 3.7460628693848023e-7

using BenchmarkTools
@btime total_energy2(scfres.ρ)  # 215.649 μs (278 allocations: 772.42 KiB)
@btime Zygote.gradient(total_energy2, scfres.ρ)[1];  # 2.044 ms (3413 allocations: 3.03 MiB)
@btime FiniteDiff.finite_difference_gradient(total_energy2, scfres.ρ);  # 4.559 s (4463509 allocations: 11.78 GiB)

# also try E w.r.t. ψ
total_energy_psi(ψ) = sum([DFTK.ene_ops(term, ψ, occupation; ρ=scfres.ρ).E for term in basis.terms])
Zygote.gradient(total_energy_psi, ψ)
@btime total_energy_psi(ψ);  # 192.225 μs (291 allocations: 776.19 KiB)
@btime Zygote.gradient(total_energy_psi, ψ);  # 1.823 ms (3464 allocations: 3.11 MiB)


# E w.r.t. basis

typeof.(basis.terms)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[1], ψ, occupation)[1], basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[2], ψ, occupation)[1], basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[3], ψ, occupation)[1], basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[4], ψ, occupation)[1], basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[5], ψ, occupation)[1], basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[6], ψ, occupation; ρ=scfres.ρ)[1], basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[7], ψ, occupation; ρ=scfres.ρ)[1], basis)
total_energy_basis(basis) = sum(DFTK.ene_ops(term, ψ, occupation; ρ=scfres.ρ).E for term in basis.terms)
total_energy_basis(basis) # -4.807121625456233
Zygote.gradient(total_energy_basis, basis) # seems to work
# TODO verify result


# basis w.r.t. lattice parameter

Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
function make_basis(a)
    lattice = a / 2 * [[0. 1. 1.];
                       [1. 0. 1.];
                       [1. 1. 0.]]
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model = model_DFT(lattice, atoms, [], symmetries=false)
    kgrid = [1, 1, 1]
    Ecut = 7
    PlaneWaveBasis(model, Ecut; kgrid=kgrid)
end

f(a) = real(sum(make_basis(a).opFFT * real(zeros(20, 20, 20))))
f(a)
Zygote.gradient(f, a)
#ERROR: Compiling Tuple{DFTK.var"##PlaneWaveBasis#59", Nothing, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, MPI.Comm, Type{PlaneWaveBasis}, Model{Float64}, Int64, Vector{StaticArrays.SVector{3, Rational{Int64}}}, Vector{Vector{Tuple{StaticArrays.SMatrix{3, 3, Int64, 9}, StaticArrays.SVector{3, Float64}}}}, Vector{Tuple{StaticArrays.SMatrix{3, 3, Int64, 9}, StaticArrays.SVector{3, Float64}}}}: try/catch is not supported.

Zygote.gradient(x -> sum(Mat3(x)), zeros(3,3))
# ERROR: Need an adjoint for constructor StaticArrays.SMatrix{3, 3, Float64, 9}. Gradient is of type FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}


