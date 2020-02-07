include("../../testcases.jl")
using PyCall
using DFTK

abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/12mg.2.hgh")]
pspmap = Dict(12 => "hgh/lda/mg-q2", )

atoms = [Element(12, load_psp(pspmap[12])) => magnesium.positions]
model = model_dft(magnesium.lattice, [:gga_x_pbe, :gga_c_pbe], atoms,
                  temperature=0.01, smearing=Smearing.MethfesselPaxton2())

DFTK.run_abinit_scf(model, @__DIR__;
                    abinitpseudos=abinitpseudos, pspmap=pspmap,
                    Ecut=25, kgrid=[5, 5, 5], n_bands=10, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimisation
