include("../../testcases.jl")
using DFTK

atoms = [ElementPsp(14, load_psp("hgh/pbe/si-q4")) => silicon.positions]
model = model_dft(silicon.lattice, [:gga_c_pbe, :gga_x_pbe], atoms)

abinitpseudos = [joinpath(@__DIR__, "Si-q4-pbe.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=25, kgrid=[3, 3, 3], n_bands=10, tol=1e-10,
                    iscf=3) # Use Anderson mixing instead of minimisation
