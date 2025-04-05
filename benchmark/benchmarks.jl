using BenchmarkTools
using DFTK
using PseudoPotentialData

setup_threading()

# For inspiration see
# https://github.com/gridap/Gridap.jl/blob/master/benchmark/benchmarks.jl
# https://github.com/gridap/Gridap.jl/blob/master/benchmark/README.md

# TODO:
#  - Hamiltonian application
#  - Diagonalisation
#  - Functionality test
#  - Response

const SUITE = BenchmarkGroup()

# TODO Just dummy to get started
SUITE["fullrun"] = BenchmarkGroup(["insulator", "SCF", "LDA"])
SUITE["fullrun"]["silicon"] = @benchmarkable let
    a = 10.26  # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    model  = model_DFT(lattice, atoms, positions; functionals=LDA())
    basis  = PlaneWaveBasis(model; Ecut=35, kgrid=[8, 8, 8])
    scfres = self_consistent_field(basis, tol=1e-8, callback=identity)
end


# TODO Use setup / teardown integration of BenchmarkTools to collect and store timings using TimerOutputs
