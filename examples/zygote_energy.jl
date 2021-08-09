using DFTK
using Zygote
import FiniteDiff

a = 10.26
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
function make_model(a)
    lattice = a / 2 * [[0. 1. 1.];
                       [1. 0. 1.];
                       [1. 1. 0.]]
    terms = [
        Kinetic(),
        AtomicLocal(),
        # AtomicNonlocal(),
        # Ewald(),
        # PspCorrection(),
        # Entropy(),
        # Hartree()
    ]
    Model(lattice; atoms=atoms, terms=terms, temperature=1e-3)
end
# kgrid = [2, 2, 2]
kgrid = [1, 1, 1]
Ecut = 7
make_basis(model::Model) = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
make_basis(a::Real) = make_basis(make_model(a))
basis = make_basis(a)

scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))
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

function HF_energy(basis, ψ, occupation, ρ)
    # TODO ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
HF_energy(ρ) = HF_energy(basis, ψ, occupation, ρ) # only for debug purposes, TODO delete
HF_energy(scfres.ρ)
g1 = Zygote.gradient(HF_energy, scfres.ρ)[1] # works

# check against finite differences
g2 = FiniteDiff.finite_difference_gradient(HF_energy, scfres.ρ)
sum(abs, g1 - g2)  # 3.7460628693848023e-7

using BenchmarkTools
@btime HF_energy(scfres.ρ)  # 215.649 μs (278 allocations: 772.42 KiB)
@btime Zygote.gradient(HF_energy, scfres.ρ)[1];  # 2.044 ms (3413 allocations: 3.03 MiB)
@btime FiniteDiff.finite_difference_gradient(HF_energy, scfres.ρ);  # 4.559 s (4463509 allocations: 11.78 GiB)

# also try E w.r.t. ψ
HF_energy_psi(ψ) = HF_energy(basis, ψ, occupation, scfres.ρ)
Zygote.gradient(HF_energy_psi, ψ)
@btime HF_energy_psi(ψ);  # 192.225 μs (291 allocations: 776.19 KiB)
@btime Zygote.gradient(HF_energy_psi, ψ);  # 1.823 ms (3464 allocations: 3.11 MiB)


# E w.r.t. basis 

typeof.(basis.terms)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[1], ψ, occupation).E, basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[2], ψ, occupation; ρ=scfres.ρ).E, basis)

HF_energy(basis::PlaneWaveBasis) = HF_energy(basis, ψ, occupation, scfres.ρ)
HF_energy(basis) # -4.807121625456233
g = Zygote.gradient(HF_energy, basis)[1];
dump(g; maxdepth=2)
# TODO verify result
# look at forces


# model w.r.t. lattice parameter
make_model(a)
Zygote.gradient(a -> make_model(a).recip_cell_volume, a) # (-0.2686157095138732,)
FiniteDiff.finite_difference_derivative(a -> make_model(a).recip_cell_volume, a) # -0.2686157095506202

make_basis(a).G_to_r_normalization # 0.06085677788055191
Zygote.gradient(a -> make_basis(a).G_to_r_normalization, a)  # (-0.0088971897486187,)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).G_to_r_normalization, a)  # -0.008897189749284017

Zygote.gradient(a -> make_basis(a).dvol, a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).dvol, a)

Zygote.gradient(a -> make_basis(a).model.recip_cell_volume, a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).model.recip_cell_volume, a)

Zygote.gradient(a -> make_basis(a).r_to_G_normalization, a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).r_to_G_normalization, a)

# diff through term construction (pre-computations)

# Kinetic
Zygote.gradient(a -> sum(make_basis(a).terms[1].kinetic_energies[1]), a)
FiniteDiff.finite_difference_derivative(a -> sum(make_basis(a).terms[1].kinetic_energies[1]), a)

# AtomicLocal 
Zygote.gradient(a -> make_basis(a).terms[2].potential[1], a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).terms[2].potential[1], a)

# Kinetic + AtomicLocal
Zygote.gradient(a -> HF_energy(make_basis(a)), a)
FiniteDiff.finite_difference_derivative(a -> HF_energy(make_basis(a)), a)

# TODO compute_density

Zygote.gradient(t -> sum(real(DFTK._accumulate_over_symmetries(t*scfres.ρ[:,:,:,1], basis, basis.ksymops[1]))), 1.0) # (470.99047570146837 - 0.0im,)
FiniteDiff.finite_difference_derivative(t -> sum(real(DFTK._accumulate_over_symmetries(t*scfres.ρ[:,:,:,1], basis, basis.ksymops[1]))), 1.0) # 470.99047570106745
FiniteDiff.finite_difference_derivative(t -> sum(real(DFTK.accumulate_over_symmetries!(zeros(20,20,20), t*scfres.ρ[:,:,:,1], basis, basis.ksymops[1]))), 1.0) # 470.99047570106745

Zygote.gradient(t -> sum(compute_density(basis, ψ + t*ψ, occupation)), 0.0) # (474.05406899236243 + 1.4210854715202004e-14im,)
FiniteDiff.finite_difference_derivative(t -> sum(compute_density(basis, ψ + t*ψ, occupation)), 0.0) # 474.0540689933782

# TODO Currently wrong below
Zygote.gradient(a -> sum(compute_density(make_basis(a), ψ, occupation)), a) # -3.125002852258521,
FiniteDiff.finite_difference_derivative(a -> sum(compute_density(make_basis(a), ψ, occupation)), a) # -69.3061504470064
FiniteDiff.finite_difference_derivative(a -> sum(DFTK._autodiff_compute_density(make_basis(a), ψ, occupation)), a) # -69.3061504470064
# TODO  debug compute_density w.r.t. basis
Zygote.gradient(basis -> sum(compute_density(basis, ψ, occupation)), basis)
using ChainRulesCore
tang = Tangent{typeof(basis)}(;r_to_G_normalization=1.0)
FiniteDiff.finite_difference_derivative(t -> sum(compute_density(basis + t*tang, ψ, occupation)), 0.0)


function HF_energy_recompute(basis, ψ, occupation)
    ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
HF_energy_recompute(basis, ψ, occupation)
Zygote.gradient(a -> HF_energy_recompute(make_basis(a), ψ, occupation), a) # -8.569624864733145,
FiniteDiff.finite_difference_derivative(a -> HF_energy_recompute(make_basis(a), ψ, occupation), a) # -0.22098990093995188
# TODO find error

