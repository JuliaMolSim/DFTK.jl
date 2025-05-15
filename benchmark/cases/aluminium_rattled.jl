# This is a large 64-atoms Al supercell with manually rattled atoms. It is a Gamma-point
# calculation with low symmetry and a large number of electrons. Particularly useful
# to measure force timings.

module aluminium_rattled
include("common.jl")

const SUITE = BenchmarkGroup(["metal", "PBE", "gamma_only", "low_symmetry"])
function setup_basis()
    system = load_system(joinpath(@__DIR__, "Al27_rattled.extxyz"))
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf")
    model  = model_DFT(system; pseudopotentials, functionals=PBE(),
                               temperature=1e-4, smearing=Smearing.Gaussian())
    PlaneWaveBasis(model; Ecut=64, kgrid=(1, 1, 1))
end

basis  = setup_basis()
scfres = setup_dummy_scfres(basis)

# Assemble benchmarks
SUITE["setup"] = @benchmarkable setup_basis() evals=1 samples=3
add_default_benchmarks!(SUITE, basis, scfres)

end
