include("../../testcases.jl")
using PyCall
using DFTK

atoms = [ElementPsp(:Mg, psp=load_psp("hgh/lda/mg-q2")) => magnesium.positions]
model = model_DFT(magnesium.lattice, atoms, :lda_xc_teter93,
                  temperature=0.01, smearing=Smearing.FermiDirac())

abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/12mg.2.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=15, kgrid=[3, 3, 3], n_bands=10, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimization
