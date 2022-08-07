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

# function forces_from_basis(basis)
#     is_converged = DFTK.ScfConvergenceDensity(1e-8)
#     scfres = self_consistent_field(basis; is_converged)
#     compute_forces(scfres)[1][1]
# end
# forces_from_basis(basis)
# Zygote.gradient(forces_from_basis, basis)
#
# function eigenvalues_from_basis(basis)
#     is_converged = DFTK.ScfConvergenceDensity(1e-8)
#     scfres = self_consistent_field(basis; is_converged)
#     sum(sum, scfres.eigenvalues)
# end
# Zygote.gradient(eigenvalues_from_basis, basis)

using LinearAlgebra
using ChainRulesCore

scfres = self_consistent_field(basis; tol=1e-8)
density_from_lattice(lattice) = compute_density(basis_from_lattice(lattice), scfres.ψ, scfres.occupation)
rho = density_from_lattice(lattice)
rho_zg, back = rrule_via_ad(Zygote.ZygoteRuleConfig(), density_from_lattice, lattice)
jvp = FiniteDiff.finite_difference_derivative(ε -> density_from_lattice(lattice + ε*lattice), 0.0)
dot(lattice, back(rho)[2]) # correct
dot(jvp, rho)


make_lattice(a) = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
f(a) = sum(values(energy_hamiltonian(basis_from_lattice(make_lattice(a)), scfres.ψ, scfres.occupation, scfres.ρ).E))
FiniteDiff.finite_difference_derivative(f, a)
Zygote.gradient(f, a) # works

using Infiltrator
infiltrate_deriv(x) = x
function ChainRulesCore.rrule(::typeof(infiltrate_deriv), x)
    function infiltrate_pb(dx)
        @infiltrate
        return NoTangent(), dx
    end
    return x, infiltrate_pb
end

function hellmann_feynman_energy(scfres, lattice)
    basis = basis_from_lattice(lattice)
    ρ = DFTK.compute_density(basis, scfres.ψ, scfres.occupation)
    energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ)
    energies.total
end
# ForwardDiff.derivative(a -> hellmann_feynman_energy(scfres, make_lattice(a)), a)
FiniteDiff.finite_difference_derivative(a -> hellmann_feynman_energy(scfres, make_lattice(a)), a)
Zygote.gradient(a -> hellmann_feynman_energy(scfres, make_lattice(a)), a) # correct

f2(a) = sum(density_from_lattice(make_lattice(a)))
FiniteDiff.finite_difference_derivative(f2, a)
ForwardDiff.derivative(f2, a)
Zygote.gradient(f2, a) # correct

f3(ρ) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ).E.total
f3(ρ; k) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ).E[k]
for k in keys(scfres.energies)
    println(k)
    f4(h) = f3(scfres.ρ + h*scfres.ρ; k)
    g1 = FiniteDiff.finite_difference_derivative(f4, 0.0)
    g2 = ForwardDiff.derivative(f4, 0.0)
    g3 = only(Zygote.gradient(f4, 0.0)) # correct
    println(" FinDiff: ", g1, " ForwardDiff: ", g2, " Zygote: ", g3)
end
# Kinetic
#  FinDiff: 0.0 ForwardDiff: 0.0 Zygote: 0.0
# AtomicLocal
#  FinDiff: -2.1687148211557647 ForwardDiff: -2.1687148211339324 Zygote: -2.1687148211339324
# AtomicNonlocal
#  FinDiff: 0.0 ForwardDiff: 0.0 Zygote: 0.0
# Ewald
#  FinDiff: 0.0 ForwardDiff: 0.0 Zygote: 0.0
# PspCorrection
#  FinDiff: 0.0 ForwardDiff: 0.0 Zygote: 0.0
# Hartree
#  FinDiff: 1.2590922290980915 ForwardDiff: 1.259092229051408 Zygote: 1.259092229051417

# Suspicion: This might be something with double-counting of kwargs
f6(x; kwargs...) = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; kwargs..., ρ=kwargs[:ρ]).E.total
f7(ρ) = f6(0.0; ρ=ρ)
FiniteDiff.finite_difference_derivative(h -> f7(scfres.ρ + h*scfres.ρ), 0.0)
Zygote.gradient(h -> f7(scfres.ρ + h*scfres.ρ), 0.0) # incorrect by factor 2

f1(; kwargs...) = kwargs[:x]
f2(; kwargs...) = f1(; kwargs..., x=kwargs[:x])
f3(x) = f2(; x)
FiniteDiff.finite_difference_derivative(f3, 0.0) # 1.0
ForwardDiff.derivative(f3, 0.0) # 1.0
Zygote.gradient(f3, 0.0) # (2.0,) aha!
# --> opened a new issue https://github.com/FluxML/Zygote.jl/issues/1284
