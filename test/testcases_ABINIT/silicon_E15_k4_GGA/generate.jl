include("../../testcases.jl")
using DFTK

atoms = [ElementPsp(:Si, psp=load_psp("hgh/pbe/si-q4")) => silicon.positions]
model = model_DFT(silicon.lattice, atoms, [:gga_c_pbe, :gga_x_pbe])

abinitpseudos = [joinpath(@__DIR__, "Si-q4-pbe.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=15, kgrid=[4, 4, 4], n_bands=6, tol=1e-10,
                    iscf=3) # Use Anderson mixing instead of minimization
