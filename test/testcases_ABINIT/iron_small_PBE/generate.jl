include("../../testcases.jl")
using PyCall
using DFTK

atoms = [ElementPsp(:Fe, psp=load_psp("hgh/lda/Fe-q8.hgh")) => iron_bcc.positions]
model = model_PBE(iron_bcc.lattice, atoms, temperature=iron_bcc.temperature,
                  smearing=Smearing.FermiDirac(), spin_polarization=:collinear)

abinitpseudos = [joinpath(@__DIR__, "Fe-q8-lda.abinit.hgh")]  # Note: Wrong pseudo is on purpose
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=20, kgrid=[4, 4, 4], nshiftk=1, shiftk=[0.5, 0.5, 0.5],
                    spinat=[0 0 4.0], n_bands=10, tol=1e-10, iscf=13, nstep=100)
# iscf == 13: Use Anderson mixing on density
