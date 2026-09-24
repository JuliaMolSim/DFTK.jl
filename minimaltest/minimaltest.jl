using DFTK
using PseudoPotentialData
using AtomsBuilder
using LinearAlgebra

#We take very (very) crude parameters
pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
model_lda = model_DFT(bulk(:C); functionals=LDA(), pseudopotentials)
model_hf  = model_HF(bulk(:C); pseudopotentials)

basis_lda = PlaneWaveBasis(model_lda; Ecut=24, kgrid=[1, 1, 1]);
basis_hf = PlaneWaveBasis(model_hf; Ecut=24, kgrid=[1, 1, 1]);

scfres_pre = self_consistent_field(basis_lda; 
                                   tol=1e-8,
                                   seed=1234,
                                   nbandsalg=FixedBands(;n_bands_converge=24))
ψ_ref = deepcopy(scfres_pre.ψ)
scfres_start = deepcopy(scfres_pre)
scfres_damp = deepcopy(scfres_pre)

for (i, ψk) in enumerate(scfres_start.ψ) 
    scfres_start.ψ[i] = ψk[:,1:7] 
    scfres_damp.ψ[i] = ψk[:,1:4] 
end
for (i, ψk) in enumerate(ψ_ref) 
    ψ_ref[i] = ψk[:,1:end] 
end
for (i, occ) in enumerate(scfres_start.occupation) 
    scfres_start.occupation[i] = occ[1:7] 
    scfres_damp.occupation[i] = occ[1:4] 
end

#t_dm = @elapsed begin
#    for i in 1:10
#    global scfres_dm = direct_minimization(basis_hf;
#        ψ = scfres_damp.ψ,
#        tol = 1e-8,
#        iterations = 50)
#    end
#end
#t_damp = @elapsed begin
#    for i in 1:10
#    global scfres_damp = self_consistent_field(basis_hf;
#        tol = 1e-8,
#        maxiter = 50,
#        ψ = scfres_start.ψ,
#        ρ = scfres_start.ρ,
#        occupation = scfres_start.occupation,
#        #mixing = SimpleMixing(),
#        solver = DFTK.ScfDampingSolver(),  # default damping ≈0.8; pass e.g. ScfDampingSolver(1.0) to disable damping
#        seed = 1234)
#    end
#end
#
#t_pcdiis = @elapsed begin
#    for i in 1:10
#    global scfres_pcdiis = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_start.ψ,
#                               ρ = scfres_start.ρ,
#                               occupation = scfres_start.occupation,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=1, reference=ψ_ref), 
#                               seed=1234)
#    end
#end
#println("Direct minimization: $(t_dm) s, $(scfres_dm.energies.total) Ha")
#println("Damping: $(t_damp) s, $(scfres_damp.energies.total) Ha")
#println("Pcdiis: $(t_pcdiis) s, $(scfres_pcdiis.energies.total) Ha")

#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_pre.ψ,
#                               ρ = scfres_pre.ρ,
#                               occupation = scfres_pre.occupation,
#                               solver=DFTK.ScfDampingSolver(), 
#                               seed=1234)
#
#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_start.ψ,
#                               ρ = scfres_start.ρ,
#                               occupation = scfres_start.occupation,
#                               solver=DFTK.ScfDampingSolver(), 
#                               seed=1234)

#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_pre.ψ,
#                               ρ = scfres_pre.ρ,
#                               occupation = scfres_pre.occupation,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(m_start=1), 
#                               seed=1234)
#
#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_start.ψ,
#                               ρ = scfres_start.ρ,
#                               occupation = scfres_start.occupation,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=1, reference=ψ_ref), 
#                               seed=1234)

#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_pre.ψ,
#                               ρ = scfres_pre.ρ,
#                               occupation = scfres_pre.occupation,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=1, reference=ψ_ref), 
#                               seed=1234)
#
#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_pre.ψ,
#                               ρ = scfres_pre.ρ,
#                               occupation = scfres_pre.occupation,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=1), 
#                               seed=1234)

#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=1, ψ_ref), 
#                               seed=1234)
#

#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_pre.ψ,
#                               ρ = scfres_pre.ρ,
#                               occupation = scfres_pre.occupation,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=2, reference=ψ_ref), 
#                               seed=1234)

#scfres = self_consistent_field(basis_lda; 
#                               tol=1e-8,
#                               maxiter=50,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfPcdiisSolver(;m_start=2, ψ_ref, errorfactor=1e6), 
#                               seed=1234)

#scfres = self_consistent_field(basis_hf; 
#                               tol=1e-8,
#                               maxiter=50,
#                               ψ = scfres_start.ψ,
#                               ρ = scfres_start.ρ,
#                               occupation = scfres_start.occupation,
#                               solver=ScfDampingSolver(), 
#                               seed=1234)
#
#scfres = self_consistent_field(basis_lda; 
#                               tol=1e-8,
#                               maxiter=50,
#                               solver=ScfDampingSolver(), 
#                               seed=1234)
#

#scfres = self_consistent_field(basis_lda; 
#                               tol=1e-8,
#                               maxiter=50,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfAndersonSolver(), 
#                               seed=1234)
#
#scfres = self_consistent_field(basis_lda; 
#                               tol=1e-8,
#                               maxiter=50,
#                               mixing = SimpleMixing(),
#                               solver=DFTK.ScfAndersonDensitySolver(), 
#                               seed=1234)
