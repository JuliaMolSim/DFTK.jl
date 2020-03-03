include("../../testcases.jl")
using DFTK

atoms = [ElementPsp(14, load_psp("hgh/lda/si-q4")) => silicon.positions]
model = model_dft(silicon.lattice, :lda_xc_teter93, atoms)

abinitpseudos = [joinpath(@__DIR__, "Si-q4-pade.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=15, kgrid=[4, 4, 4], n_bands=6, tol=1e-10,
                    iscf=3) # Use Anderson mixing instead of minimisation
