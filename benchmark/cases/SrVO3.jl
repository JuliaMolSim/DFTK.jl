# This is a highly symmetric structure with a dense k-point mesh and multiple atomic species.
# It is useful to measure the cumulative time of the iterative solver.

module SrVO3
include("common.jl")

const SUITE = BenchmarkGroup(["insulator", "PBE", "high_symmetry"])
function setup_basis()
    system = load_system(joinpath(@__DIR__, "SrVO3.extxyz"))
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf")
    model  = model_DFT(system; pseudopotentials, functionals=PBE(),
                               temperature=1e-2, smearing=Smearing.Gaussian())
    PlaneWaveBasis(model; Ecut=72, kgrid=(10, 10, 10))
end

basis  = setup_basis()
scfres = setup_dummy_scfres(basis)

# Assemble benchmarks
SUITE["setup"] = @benchmarkable setup_basis() evals=1 samples=3
add_default_benchmarks!(SUITE, basis, scfres)
end
