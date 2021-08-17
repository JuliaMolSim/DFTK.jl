using DFTK
using Zygote
import FiniteDiff

# NOTE: this snippet works on Zygote v0.6.17 but breaks on v0.6.18 (suspicion: due to `basis` argument in PlaneWaveBasis rrule)

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

function HF_energy_debug(basis, ψ, occupation, ρ)
    # TODO ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
HF_energy_debug(basis::PlaneWaveBasis) = HF_energy_debug(basis, ψ, occupation, scfres.ρ)
HF_energy_debug(ρ) = HF_energy_debug(basis, ψ, occupation, ρ) # only for debug purposes, TODO delete
HF_energy_debug(scfres.ρ)
g1 = Zygote.gradient(HF_energy_debug, scfres.ρ)[1] # works

# check against finite differences
g2 = FiniteDiff.finite_difference_gradient(HF_energy_debug, scfres.ρ)
sum(abs, g1 - g2)  # 3.7460628693848023e-7

using BenchmarkTools
@btime HF_energy_debug(scfres.ρ)  # 215.649 μs (278 allocations: 772.42 KiB)
@btime Zygote.gradient(HF_energy_debug, scfres.ρ)[1];  # 2.044 ms (3413 allocations: 3.03 MiB)
@btime FiniteDiff.finite_difference_gradient(HF_energy_debug, scfres.ρ);  # 4.559 s (4463509 allocations: 11.78 GiB)

# also try E w.r.t. ψ
HF_energy_psi(ψ) = HF_energy_debug(basis, ψ, occupation, scfres.ρ)
Zygote.gradient(HF_energy_psi, ψ)
@btime HF_energy_psi(ψ);  # 192.225 μs (291 allocations: 776.19 KiB)
@btime Zygote.gradient(HF_energy_psi, ψ);  # 1.823 ms (3464 allocations: 3.11 MiB)


# E w.r.t. basis 

typeof.(basis.terms)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[1], ψ, occupation).E, basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[2], ψ, occupation; ρ=scfres.ρ).E, basis)

# diff through term construction (pre-computations)

# Kinetic + AtomicLocal
Zygote.gradient(a -> HF_energy_debug(make_basis(a)), a)
FiniteDiff.finite_difference_derivative(a -> HF_energy_debug(make_basis(a)), a)

# TODO move these over to test/chainrules.jl
kpt = basis.kpoints[1]
kpt.mapping
x = rand(ComplexF64,259)
y = rand(20,20,20)
w = rand(ComplexF64,20,20,20)

function f(a)
    # a = real(a)
    basis = make_basis(a)
    sum(abs2, r_to_G(basis, G_to_r(basis, kpt, a*x) .* w) .* w)
end
Zygote.gradient(f, a)
FiniteDiff.finite_difference_derivative(f, a)
# looks wrong, TODO

using ChainRulesCore
g2 = Zygote.gradient(basis -> real(sum(DFTK._compute_partial_density(basis, kpt, ψ[1], occupation[1]) .* y)), basis)[1]

tang = Tangent{typeof(basis)}(;r_to_G_normalization=1.0)
FiniteDiff.finite_difference_derivative(t -> real(sum(DFTK._compute_partial_density(basis + t*tang, kpt, ψ[1], occupation[1]) .* y)), 0.0)
g2.r_to_G_normalization
# looks good

tang = Tangent{typeof(basis)}(;G_to_r_normalization=1.0)
FiniteDiff.finite_difference_derivative(t -> real(sum(DFTK._compute_partial_density(basis + t*tang, kpt, ψ[1], occupation[1]) .* y)), 0.0)
g2.G_to_r_normalization
# looks good

function HF_energy(basis, ψ, occupation)
    ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
Zygote.gradient(a -> HF_energy(make_basis(a), ψ, occupation), a) # -0.22098988721348034,
FiniteDiff.finite_difference_derivative(a -> HF_energy(make_basis(a), ψ, occupation), a) # -0.22098988731612818

# TODO HF forces. This needs diff w.r.t. keyword arg Model(...; atoms) which is not supported by ChainRules so far.
