# Simple system for some super rapid testing

module silicon
include("common.jl")

const SUITE = BenchmarkGroup(["insulator", "LDA", "fast"])
function setup_basis()
    a = 10.26  # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    model = model_DFT(lattice, atoms, positions; functionals=LDA())
    PlaneWaveBasis(model; Ecut=35, kgrid=(8, 8, 8))
end

basis  = setup_basis()
scfres = setup_dummy_scfres(basis)

SUITE["setup"]          = @benchmarkable setup_basis()                 evals=1 samples=3
SUITE["scf_3steps"]     = @benchmarkable bm_scf_3steps($basis)         evals=1 samples=1
SUITE["scf_full"]       = @benchmarkable bm_scf_full($basis; tol=1e-8) evals=1 samples=1
SUITE["compute_forces"] = @benchmarkable bm_compute_forces($scfres)    evals=1 samples=1
end
