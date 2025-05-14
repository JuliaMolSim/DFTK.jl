using AtomsIO
using BenchmarkTools
using DFTK
using PseudoPotentialData
using Random

function setup_dummy_scfres(basis)
    Random.seed!(42)
    self_consistent_field(basis; maxiter=1, callback=identity)
end

#
# Benchmarks
#
function bm_scf(basis; tol=1e-6, kwargs...)
    # TODO Problem here is that benchmarktools runs a warmup
    # TODO Use setup / teardown integration of BenchmarkTools to collect and store timings using TimerOutputs
    Random.seed!(101)
    self_consistent_field(basis; tol, callback=identity, kwargs...)
end

bm_compute_forces(scfres) = compute_forces_cart(scfres)

# TODO other benchmarks to include:
#  - Hamiltonian application
#  - Diagonalisation
#  - Response

#
# Benchmark collections
#
function add_default_benchmarks!(SUITE, basis, scfres)
    SUITE["scf_3steps"]     = @benchmarkable bm_scf($basis; maxiter=3)  evals=1 samples=1
    SUITE["scf_full"]       = @benchmarkable bm_scf($basis)             evals=1 samples=1
    SUITE["compute_forces"] = @benchmarkable bm_compute_forces($scfres) evals=1 samples=3
end
