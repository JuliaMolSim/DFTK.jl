include("../../testcases.jl")
using PyCall
using DFTK

atoms = [ElementPsp(12, load_psp("hgh/lda/mg-q2")) => magnesium.positions]
model = model_dft(magnesium.lattice, [:gga_x_pbe, :gga_c_pbe], atoms,
                  temperature=0.01, smearing=Smearing.MethfesselPaxton2())

abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/12mg.2.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=15, kgrid=[3, 3, 3], n_bands=10, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimisation
