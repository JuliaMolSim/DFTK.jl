include("../../testcases.jl")
using PyCall
using DFTK
using LinearAlgebra

atoms = [ElementPsp(:O, psp=load_psp("hgh/pbe/O-q6.hgh")) => o2molecule.positions]
model = model_PBE(o2molecule.lattice, atoms, temperature=0.02,
                  smearing=Smearing.Gaussian(), spin_polarization=:collinear)

abinitpseudos = [joinpath(@__DIR__, "O-q6-pbe.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=13, kgrid=[1, 1, 1], spinat=[0 0 1.0 0 0 1.0],
                    n_bands=8, tol=1e-10, nstep=100)
