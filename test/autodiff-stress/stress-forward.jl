# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual
using DFTK

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        Ewald(),
        PspCorrection()
    ]
    model = Model(lattice; atoms=atoms, terms=terms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15          # kinetic energy cutoff in Hartree
    PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])
end

a = 10.26
scfres = self_consistent_field(make_basis(a), tol=1e-8)

function compute_energy(scfres_ref, a)
    basis = make_basis(a)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(a) = compute_energy(scfres, a)
compute_energy(10.26)

import FiniteDiff
FiniteDiff.finite_difference_derivative(compute_energy, 10.26) # -2.940653844187964e9 

using ForwardDiff
ForwardDiff.derivative(compute_energy, 10.26) # -2.940653844103271e9

using BenchmarkTools
@btime compute_energy(10.26)                                           # 19.513 ms ( 60004 allocations:  8.15 MiB)
@btime FiniteDiff.finite_difference_derivative(compute_energy, 10.26)  # 39.317 ms (120012 allocations: 16.29 MiB)
@btime ForwardDiff.derivative(compute_energy, 10.26)                   # 80.757 ms (543588 allocations: 31.91 MiB)

