include("../../testcases.jl")
using DFTK

abinitpseudos = [joinpath(@__DIR__, "Si-q4-pade.abinit.hgh")]
pspmap = Dict(14 => "hgh/lda/si-q4", )

atoms = [Element(14, load_psp(pspmap[14])) => silicon.positions]
model = model_dft(silicon.lattice, :lda_xc_teter93, atoms)

DFTK.run_abinit_scf(model, @__DIR__;
                    abinitpseudos=abinitpseudos, pspmap=pspmap,
                    Ecut=15, kgrid=[4, 4, 4], n_bands=6, tol=1e-10,
                    iscf=3) # Use Anderson mixing instead of minimisation
