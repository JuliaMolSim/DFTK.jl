# Simple system for some super rapid testing

module silicon
include("common.jl")

const SUITE = BenchmarkGroup(["insulator", "LDA", "fast"])
function setup_basis(; symmetries=true, kgrid=(8, 8, 8))
    a = 10.26  # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    model = model_DFT(lattice, atoms, positions; functionals=LDA(), symmetries)
    PlaneWaveBasis(model; Ecut=35, kgrid)
end

basis  = setup_basis()
scfres = setup_dummy_scfres(basis)

# Assemble benchmarks
SUITE["setup"] = @benchmarkable setup_basis() evals=1 samples=3
add_default_benchmarks!(SUITE, basis, scfres)

# Response benchmark
scfres_nosym = setup_dummy_scfres(setup_basis(; symmetries=false, kgrid=(4, 4, 4)))
perturb = setup_atomic_perturbation(scfres_nosym)
SUITE["response"] = @benchmarkable bm_response($scfres_nosym, $perturb) evals=1 samples=1

end
