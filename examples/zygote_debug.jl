using DFTK
using Zygote
using FiniteDiff
using ForwardDiff
setup_threading(n_blas=1)

# Specify structure for silicon lattice
a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si,        Si        ]
positions = [ones(3)/8, -ones(3)/8]
terms = [
    Kinetic(),
    AtomicLocal(),
    AtomicNonlocal(),
    Ewald(),
    PspCorrection(),
    Hartree(),
]
model = Model(lattice, atoms, positions; terms, symmetries=false)
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), kshift=(0, 0, 0))

function energy_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    # scfres.energies.total
    sum(values(scfres.energies))
end
energy_from_basis(basis)
Zygote.gradient(energy_from_basis, basis)

function forces_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    compute_forces(scfres)[1][1]
end
forces_from_basis(basis)
Zygote.gradient(forces_from_basis, basis)

function eigenvalues_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    sum(sum, scfres.eigenvalues)
end
Zygote.gradient(eigenvalues_from_basis, basis)

# Comparison to FiniteDiff

function basis_from_lattice(lattice)
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        Ewald(),
        PspCorrection(),
        Hartree(),
    ]
    model = Model(lattice, atoms, positions; terms, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), kshift=(0, 0, 0))
end

function energy_from_a(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    basis = basis_from_lattice(lattice)
    energy_from_basis(basis)
end
FiniteDiff.finite_difference_derivative(energy_from_a, a)
ForwardDiff.derivative(energy_from_a, a)
Zygote.withgradient(energy_from_a, a)
# TODO: there seems to be a factor 2 in Zygote

using LinearAlgebra
using ChainRulesCore

density_from_lattice(lattice) = compute_density(basis_from_lattice(lattice), scfres.ψ, scfres.occupation)
rho = density_from_lattice(lattice)
rho_zg, back = rrule_via_ad(Zygote.ZygoteRuleConfig(), density_from_lattice, lattice)
jvp = FiniteDiff.finite_difference_derivative(ε -> density_from_lattice(lattice + ε*lattice), 0.0)
dot(lattice, back(rho)[2]) # correct
dot(jvp, rho)


∂energies = Tangent{Any}(energies = Tangent{Any, NamedTuple{(:first, :second), Tuple{ZeroTangent, Float64}}}[Tangent{Any}(first = ZeroTangent(), second = 1.0), Tangent{Any}(first = ZeroTangent(), second = 1.0), Tangent{Any}(first = ZeroTangent(), second = 1.0), Tangent{Any}(first = ZeroTangent(), second = 1.0), Tangent{Any}(first = ZeroTangent(), second = 1.0), Tangent{Any}(first = ZeroTangent(), second = 1.0)],)
# (; E, H), energy_hamiltonian_pullback =
#         rrule_via_ad(config, energy_hamiltonian, basis, scfres.ψ, scfres.occupation, scfres.ρ)
energy_hamiltonian_from_lattice(lattice) = energy_hamiltonian(basis_from_lattice(lattice), scfres.ψ, scfres.occupation, scfres.ρ)
(; E, H) = energy_hamiltonian_from_lattice(lattice)
(; E, H), energy_hamiltonian_pullback =
        rrule_via_ad(Zygote.ZygoteRuleConfig(), energy_hamiltonian_from_lattice, lattice)
energy_hamiltonian_pullback((; E=∂energies, H=NoTangent())) # result is same as grad(energy_from_lattice)
# TODO debug
