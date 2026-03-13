# # Exact exchange and Hybrid DFT
#
# This is a quick sketch how to run a simple hybrid DFT calculation using DFTK.
#
# !!! warning "Preliminary implementation"
#     Hybrid-DFT is not yet performance-optimised and has only seen rudimentary testing
#     so far. Further only Gamma point calculations are supported and the interface is not
#     yet considered stable. We appreciate any issues, bug reports or PRs.
#
using AtomsBuilder
using DFTK
using PseudoPotentialData

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
system = bulk(:Si)

# First perform a PBE calculation to get a good starting point
model  = model_DFT(system; pseudopotentials, functionals=PBE())
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis; tol=1e-6);
nothing  # hide

# Then run PBE0, see also [PBE0](@ref) and [HybridFunctional](@ref) for more documentation.
model  = model_DFT(system; pseudopotentials, functionals=PBE0())
basis  = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
scfres = self_consistent_field(basis; tol=1e-4, scfres.ρ, scfres.ψ, scfres.occupation,
                               scfres.eigenvalues, solver=DFTK.scf_damping_solver());
nothing  # hide
