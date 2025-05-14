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

SUITE["setup"]          = @benchmarkable evals=1 samples=3 setup_basis()
SUITE["scf_3steps"]     = @benchmarkable evals=1 samples=1 bm_scf_3steps($basis)
SUITE["scf_full"]       = @benchmarkable evals=1 samples=1 bm_scf_full($basis)
SUITE["compute_forces"] = @benchmarkable evals=1 samples=1 bm_compute_forces($scfres)
end
