include("../../testcases.jl")
using PyCall
using DFTK

atoms = [ElementPsp(:Fe, psp=load_psp("hgh/pbe/Fe-q16.hgh")) => iron_bcc.positions]
model = model_PBE(iron_bcc.lattice, atoms, temperature=iron_bcc.temperature,
                  smearing=Smearing.FermiDirac(), spin_polarization=:collinear)

abinitpseudos = [joinpath(@__DIR__, "Fe-q16-pbe.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=50, kgrid=[11, 11, 11],
                    spinat=[0 0 4.0], n_bands=14, tol=1e-10, iscf=13, nstep=100)
