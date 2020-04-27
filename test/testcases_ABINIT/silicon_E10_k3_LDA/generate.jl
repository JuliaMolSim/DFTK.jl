include("../../testcases.jl")
using DFTK

atoms = [ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4")) => silicon.positions]
model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93)

abinitpseudos = [joinpath(@__DIR__, "Si-q4-pade.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=10, kgrid=[3, 3, 3], n_bands=6, tol=1e-10,
                    iscf=3)  # Use Anderson mixing instead of minimization
