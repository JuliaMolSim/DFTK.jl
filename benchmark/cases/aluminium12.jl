# This is a 12-atom Al supercell without any rattling.

module aluminium12
include("common.jl")

const SUITE = BenchmarkGroup(["metal", "PBE", "fast"])
function setup_basis(; symmetries=true, kgrid=(3, 7, 7))
    system = load_system(joinpath(@__DIR__, "Al12.extxyz"))
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf")
    model  = model_DFT(system; pseudopotentials, functionals=PBE(), symmetries,
                               temperature=1e-3, smearing=Smearing.Gaussian())
    PlaneWaveBasis(model; Ecut=30, kgrid)
end

#=
basis   = setup_basis()
scfres  = setup_dummy_scfres(basis)
=#

# Response benchmark
scfres_nosym = setup_dummy_scfres(setup_basis(; symmetries=false, kgrid=(1, 2, 2));
                                  mixing=KerkerMixing())
perturb = setup_atomic_perturbation(scfres_nosym)
SUITE["response"] = @benchmarkable(
    bm_response($scfres_nosym, $perturb; tol=1e-8),
    evals=1, samples=1
)

end
