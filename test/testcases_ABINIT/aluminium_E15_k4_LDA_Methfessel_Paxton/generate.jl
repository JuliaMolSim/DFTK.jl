include("../../testcases.jl")
using PyCall
using DFTK

atoms = [ElementPsp(:Al, psp=load_psp("hgh/lda/al-q3")) => aluminium_primitive.positions]
model = model_LDA(aluminium_primitive.lattice, atoms, temperature=0.01,
                  smearing=Smearing.MethfesselPaxton2(), spin_polarisation=:collinear)

abinitpseudos = [joinpath(pyimport("abipy.data").__path__[end], "hgh_pseudos/13al.3.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=15, kgrid=[4, 4, 4], n_bands=8, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimisation
