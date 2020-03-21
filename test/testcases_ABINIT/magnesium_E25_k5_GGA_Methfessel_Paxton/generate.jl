include("../../testcases.jl")
using PyCall
using DFTK

atoms = [ElementPsp(:Mg, psp=load_psp("hgh/lda/mg-q2")) => magnesium.positions]
model = model_DFT(magnesium.lattice, atoms, [:gga_x_pbe, :gga_c_pbe],
                  temperature=0.01, smearing=Smearing.MethfesselPaxton2())

abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/12mg.2.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=25, kgrid=[5, 5, 5], n_bands=10, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimisation
