using DFTK
using Zygote
import FiniteDiff


a = 10.26
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/He-q2"))
atoms = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
function make_model(a)
    lattice = a / 2 * [[0. 1. 1.];
                       [1. 0. 1.];
                       [1. 1. 0.]]
    terms = [
        Kinetic(),
        AtomicLocal(),
        # AtomicNonlocal(),
        Ewald(),
        # PspCorrection(),
        # Entropy(),
        Hartree()
    ]
    Model(lattice, atoms, positions; terms, temperature=1e-3)
end
kgrid = [1, 1, 1]
Ecut = 7
make_basis(model::Model) = PlaneWaveBasis(model; Ecut, kgrid)
make_basis(a::Real) = make_basis(make_model(a))
basis = make_basis(a)

scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))
ψ = scfres.ψ
occupation = scfres.occupation

function total_energy(ρ)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
    sum(values(energies))
end

total_energy(scfres.ρ)
# Zygote.gradient(total_energy, scfres.ρ) # ERROR

# Zygote.gradient(x -> sum(values(Dict(1=>x))), 1.0) # ERROR
# using OrderedCollections
# Zygote.gradient(x -> sum(values(OrderedDict{Int,Float64}(1=>x))), 1.0) ERROR
#
# # Zygote has issues with Dicts?
# https://github.com/FluxML/Zygote.jl/issues/760

# TODO move these over to test/chainrules.jl
kpt = basis.kpoints[1]
kpt.mapping
x = rand(ComplexF64,259)
y = rand(20,20,20)
w = rand(ComplexF64,20,20,20)
function f(a)
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

function hellmann_feynman_energy(basis, ψ, occupation)
    ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, basis, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
Zygote.gradient(a -> hellmann_feynman_energy(make_basis(a), ψ, occupation), a) # 0.501817333084525
FiniteDiff.finite_difference_derivative(a -> hellmann_feynman_energy(make_basis(a), ψ, occupation), a) # 0.501817332882851

# TODO make more terms reverse-mode differentiable

