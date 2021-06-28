# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual
using DFTK

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model = model_DFT(lattice, atoms, []; symmetries=false)
    kgrid = [4, 4, 4] # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 7          # kinetic energy cutoff in Hartree
    PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])
end

a = 10.26
scfres = self_consistent_field(make_basis(a), tol=1e-4)

function compute_energy(scfres_ref, a)
    basis = make_basis(a)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(a) = compute_energy(scfres, a)
compute_energy(10.26)

import FiniteDiff
FiniteDiff.finite_difference_derivative(compute_energy, 10.26) # -0.4347657610031856 

using ForwardDiff
ForwardDiff.derivative(compute_energy, 10.26) # -0.434770331446876

using BenchmarkTools
@btime compute_energy(10.26)                                           # 101.814 ms (89326 allocations: 48.94 MiB)
@btime FiniteDiff.finite_difference_derivative(compute_energy, 10.26)  # 224.734 ms (178656 allocations: 97.87 MiB)
@btime ForwardDiff.derivative(compute_energy, 10.26)                   # 340.684 ms (1556202 allocations: 146.99 MiB)
