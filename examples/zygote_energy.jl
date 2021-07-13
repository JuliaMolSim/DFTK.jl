using DFTK
using Zygote
import FiniteDiff

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn])
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
Zygote.gradient(total_energy, scfres.ρ)

Zygote.gradient(x -> sum(values(Dict(1=>x))), 1.0)

using OrderedCollections
Zygote.gradient(x -> sum(values(OrderedDict{Int,Float64}(1=>x))), 1.0)

# https://github.com/FluxML/Zygote.jl/issues/760 ?
