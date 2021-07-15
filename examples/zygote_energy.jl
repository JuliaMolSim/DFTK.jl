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
model = model_DFT(lattice, atoms, [])
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
