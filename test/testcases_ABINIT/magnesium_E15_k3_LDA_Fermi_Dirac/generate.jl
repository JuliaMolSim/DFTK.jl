include("../../testcases.jl")
using PyCall
using DFTK

abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/12mg.2.hgh")]
pspmap = Dict(12 => "hgh/lda/mg-q2", )

atoms = [Element(12, load_psp(pspmap[12])) => magnesium.positions]
model = model_dft(magnesium.lattice, :lda_xc_teter93, atoms,
                  temperature=0.01, smearing=Smearing.FermiDirac())

DFTK.run_abinit_scf(model, @__DIR__;
                    abinitpseudos=abinitpseudos, pspmap=pspmap,
                    Ecut=15, kgrid=[3, 3, 3], n_bands=10, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimisation
