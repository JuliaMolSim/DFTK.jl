include("../../testcases.jl")
using DFTK

atoms = [ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4")) => silicon.positions]
model = model_DFT(silicon.lattice, atoms, [:lda_x, :lda_c_vwn])

abinitpseudos = [joinpath(@__DIR__, "Si-q4-pade.abinit.hgh")]
DFTK.run_abinit_scf(model, @__DIR__; abinitpseudos=abinitpseudos,
                    Ecut=25, kgrid=[3, 3, 3], n_bands=10, tol=1e-10,
                    iscf=3) # Use Anderson mixing instead of minimization
