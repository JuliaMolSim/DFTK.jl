using LinearAlgebra
using DFTK

# LiCu lattice
lattice = Matrix(Diagonal([9.124450004207551, 9.891041973775085, 8.270832367071698]))

Li = ElementPsp(:Li, psp=load_psp("hgh/lda/li-q3.hgh"))
Cu = ElementPsp(:Cu, psp=load_psp("hgh/lda/cu-q11.hgh"))
atoms = [Li => [[0.015272, 0.0, 0.0], [0.332794, 0.0, 0.5],
                [0.811602, 0.264698, 0.5], [0.811602, 0.735301, 0.5]],
         Cu => [[0.518469, 0.248638, 0.0], [0.017306, 0.499999, 0.0],
                [0.518469, 0.751363, 0.0], [0.307825, 0.499999, 0.5]]]

model = model_LDA(lattice, atoms, temperature=0.001)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 5           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-12, solver=scf_nlsolve_solver(m=10))
scfres.energies
