@testitem "Reproducibility of seeded SCF runs" setup=[TestCases] begin
using DFTK
silicon = TestCases.silicon

model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions; functionals=LDA())
Ecut = 15
kgrid = [2, 2, 2]

basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres1 = self_consistent_field(basis; tol=1e-7)

# Use seed from scfres1 for reproducibility
scfres2 = self_consistent_field(basis; tol=1e-7, scfres1.seed)

# Should be exactly equal if the computation is reproducible, no need for epsilons.
@assert scfres1.history_Etot == scfres2.history_Etot
@assert scfres1.history_Δρ == scfres2.history_Δρ
@assert scfres1.ψ == scfres2.ψ
@assert scfres1.ρ == scfres2.ρ
end
