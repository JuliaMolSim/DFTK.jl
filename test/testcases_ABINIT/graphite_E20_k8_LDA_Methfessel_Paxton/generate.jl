include("../../testcases.jl")
using PyCall
using DFTK
import DFTK.units: Ǎ

# Note: This is not exactly the minimum-energy structure
a = 1.228Ǎ
b = 2.12695839Ǎ
c = 7Ǎ
lattice = [[a a 0]; [-b b 0]; [0 0 c]]
C = ElementPsp(:C, psp=load_psp("hgh/lda/c-q4"))
atoms = [C => [[0, 0, 1/4], [0, 0, 3/4], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]]

model = model_DFT(lattice, atoms, :lda_xc_teter93, temperature=0.01,
                  smearing=Smearing.MethfesselPaxton2())
abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/6c.4.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=20, kgrid=[8, 8, 8], n_bands=12, tol=1e-10,
                    iscf=3) # Use Anderson mixing instead of minimisation
