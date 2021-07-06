# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual
using DFTK

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn]; symmetries=false)
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
FiniteDiff.finite_difference_derivative(compute_energy, 10.26) # -1.215583767344458

using ForwardDiff
ForwardDiff.derivative(compute_energy, 10.26) # -1.2155837670108651

using BenchmarkTools
@btime compute_energy(10.26)                                           #  91.082 ms (  88919 allocations:  65.00 MiB)
@btime FiniteDiff.finite_difference_derivative(compute_energy, 10.26)  # 244.758 ms ( 177842 allocations: 129.99 MiB)
@btime ForwardDiff.derivative(compute_energy, 10.26)                   # 206.069 ms (1555666 allocations: 177.43 MiB)
