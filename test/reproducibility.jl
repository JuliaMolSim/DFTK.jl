@testitem "Reproducibility of SCF runs" setup=[TestCases] begin
using DFTK
silicon = TestCases.silicon

model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions; functionals=LDA())
Ecut = 15
kgrid = [2, 2, 2]

basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres1 = self_consistent_field(basis; tol=1e-7)

scfres2 = self_consistent_field(basis; tol=1e-7)

# Should be exactly equal if the computation is reproducible, no need for epsilons.
@assert scfres1.history_Etot == scfres2.history_Etot
@assert scfres1.history_Δρ == scfres2.history_Δρ
end
