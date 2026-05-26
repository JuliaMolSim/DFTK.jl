using DFTK
using TimerOutputs: reset_timer!, print_timer
using NPZ
using LinearAlgebra
using PseudoPotentialData
using AtomsBuilder
using GLMakie

pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf")

Al = ElementPsp(:Al, pd_pbe_family)

atoms = [Al, Al, Al, Al]

a = 7.653  # in bohr
lattice = a * [[1.0 0.0 0.0];
               [0.0 1.0 0.0];
               [0.0 0.0 1.0]]

positions = [
    [0.00, 0.00, 0.00],
    [0.50, 0.50, 0.00],
    [0.50, 0.00, 0.50],
    [0.00, 0.50, 0.50],
]

model_hf = model_HF(lattice,
		    atoms,
		    positions
)

nelec=model_hf.n_electrons
nelec=floor(Int,nelec/2)
nelecbig=4*nelec
pad=3

model_lda = model_LDA(lattice,
		      atoms,
		      positions,
)

Ecut = 26

basis_hf  = PlaneWaveBasis(model_hf; Ecut=Ecut, kgrid=[1, 1, 1])
basis_lda  = PlaneWaveBasis(model_lda; Ecut=Ecut, kgrid=[1, 1, 1])

scfres_lda = self_consistent_field(basis_lda;
			           seed=1234,
			           nbandsalg=FixedBands(; n_bands_converge=nelecbig));

ψ_ref = scfres_lda.ψ
ψ_start = deepcopy(ψ_ref)
occ_lda = scfres_lda.occupation
eig_lda = scfres_lda.eigenvalues


for ik in 1:length(ψ_ref)
	ψ_ref[ik] = ψ_ref[ik][:,1:nelecbig]
	ψ_start[ik] = ψ_start[ik][:,1:nelec+pad]
	occ_lda[ik] = occ_lda[ik][begin:nelec+pad] 
	eig_lda[ik] = eig_lda[ik][begin:nelec+pad] 
end

ρ_lda = scfres_lda.ρ

etol = 1e-10 / 27.211

reset_timer!(DFTK.timer)
scfres_hf = self_consistent_field(basis_hf;
				  is_converged = ScfConvergenceEnergy(etol),
				  ρ = deepcopy(ρ_lda),
				  ψ = deepcopy(ψ_start),
				  occupation = deepcopy(occ_lda),
				  eigenvalues = deepcopy(eig_lda),
       			       	  seed=1234,
       			       	  mixing=SimpleMixing(),
				  solver=DFTK.scf_pcdiis_solver(;ψ_ref=ψ_ref));
print_timer(DFTK.timer)

reset_timer!(DFTK.timer)
scfres_hf = self_consistent_field(basis_hf;
				  is_converged = ScfConvergenceEnergy(etol),
				  ρ = deepcopy(ρ_lda),
				  ψ = deepcopy(ψ_start),
				  occupation = deepcopy(occ_lda),
				  eigenvalues = deepcopy(eig_lda),
       			       	  seed=1234,
       			       	  mixing=SimpleMixing(),
				  solver=DFTK.scf_damping_solver());
print_timer(DFTK.timer)

reset_timer!(DFTK.timer)
scfres_hf = self_consistent_field(basis_hf;
				  is_converged = ScfConvergenceEnergy(etol),
				  ρ = deepcopy(ρ_lda),
				  ψ = deepcopy(ψ_start),
				  occupation = deepcopy(occ_lda),
				  eigenvalues = deepcopy(eig_lda),
       			       	  seed=1234,
       			       	  mixing=SimpleMixing(),
				  solver=DFTK.scf_anderson_solver());
print_timer(DFTK.timer)
