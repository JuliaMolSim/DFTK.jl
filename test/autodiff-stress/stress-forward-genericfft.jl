# Hellmann-Feynman stress
# via ForwardDiff & generic arithmetic (FourierTransforms.jl)
# (disregarding performance)
using DFTK
using Test

# for generic FourierTransforms.jl (TODO replace by FFTW later)
using DoubleFloats
using GenericLinearAlgebra

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    terms = [
        Kinetic(),
        AtomicLocal(),
        # AtomicNonlocal(),
        Ewald(),
        PspCorrection()
    ]
    model = Model(lattice; atoms=atoms, terms=terms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15          # kinetic energy cutoff in Hartree
    PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])
end

a = 10.26
basis = make_basis(a)

# scfres = self_consistent_field(basis, tol=1e-8) # LoadError: Unable to find non-fractional occupations that have the correct number of electrons. You should add a temperature.
# try a bogus tolerance for debugging
scfres = self_consistent_field(basis, tol=1e9)

function compute_energy(scfres_ref, a)
    basis = make_basis(a)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(a) = compute_energy(scfres, a)
compute_energy(10.26)

import FiniteDiff
FiniteDiff.finite_difference_derivative(compute_energy, 10.26) # -11.113131188820518 

using ForwardDiff
ForwardDiff.derivative(compute_energy, 10.26) # -11.113131188299548

using BenchmarkTools
@btime compute_energy(10.26)                                           # 14.294 ms (60112 allocations: 9.65 MiB)
@btime FiniteDiff.finite_difference_derivative(compute_energy, 10.26)  # 29.582 ms (120228 allocations: 19.30 MiB)
@btime ForwardDiff.derivative(compute_energy, 10.26)                   # 26.178 ms (70669 allocations: 14.51 MiB)

