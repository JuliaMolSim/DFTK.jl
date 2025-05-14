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

SUITE["setup"]          = @benchmarkable evals=1 samples=3 setup_basis()
SUITE["scf_3steps"]     = @benchmarkable evals=1 samples=1 bm_scf_3steps($basis)
SUITE["scf_full"]       = @benchmarkable evals=1 samples=1 bm_scf_full($basis)
SUITE["compute_forces"] = @benchmarkable evals=1 samples=1 bm_compute_forces($scfres)
end
