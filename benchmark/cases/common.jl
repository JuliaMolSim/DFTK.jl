using AtomsIO
using BenchmarkTools
using DFTK
using PseudoPotentialData

# TODO other benchmarks to include:
#  - Hamiltonian application
#  - Diagonalisation
#  - Response
#
#  - A function to automatically add the standard benchmarks for a particular structure ?


function bm_scf_3steps(basis)
    # TODO Problem here is that benchmarktools runs a warmup
    Random.seed!(101)
    self_consistent_field(basis; tol=1e-10, callback=identity, maxiter=3)
end

function bm_scf_full(basis; tol=1e-6)
    # TODO Problem here is that benchmarktools runs a warmup
    # TODO Use setup / teardown integration of BenchmarkTools to collect and store timings using TimerOutputs

    Random.seed!(234)
    println("scf_full")
    self_consistent_field(basis; tol, callback=identity)
end

function setup_dummy_scfres(basis)
    Random.seed!(42)
    self_consistent_field(basis; maxiter=1, callback=identity)
end

bm_compute_forces(scfres) = compute_forces_cart(scfres)
