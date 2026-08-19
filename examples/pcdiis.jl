using DFTK
using PseudoPotentialData
using AtomsBuilder

pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
model_lda = model_DFT(bulk(:C); functionals=LDA(), pseudopotentials)
model_hf  = model_HF(bulk(:C); pseudopotentials)

basis_lda = PlaneWaveBasis(model_lda; Ecut=24, kgrid=[1, 1, 1]);
basis_hf = PlaneWaveBasis(model_hf; Ecut=24, kgrid=[1, 1, 1]);

scfres_pre = self_consistent_field(basis_lda; 
                                   tol=1e-8,
                                   seed=1234,
                                   nbandsalg=FixedBands(;n_bands_converge=24))
ψ_start = deepcopy(scfres_pre.ψ)
o_start = deepcopy(scfres_pre.occupation)

cut = Int(model_hf.n_electrons / 2) + 3

for (i, ψk) in enumerate(ψ_start) 
    ψ_start[i] = ψk[:,1:cut] 
end

for (i, o) in enumerate(o_start) 
    o_start[i] = o[1:cut] 
end

scfres_pcdiis = self_consistent_field(basis_hf; 
                                      tol=1e-8,
                                      maxiter=50,
                                      ρ = scfres_pre.ρ,
                                      ψ = ψ_start,
                                      occupation = o_start,
                                      mixing = SimpleMixing(), #mixing interferes with Pcdiis!
                                      solver=DFTK.ScfPcdiisSolver(;m_start=1, reference=scfres_pre.ψ))