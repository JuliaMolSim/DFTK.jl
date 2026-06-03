using DFTK
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra
using PseudoPotentialData
using AtomsBuilder

pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf")

Si = ElementPsp(:Si, pd_pbe_family)
atoms = [Si, Si]

# Silicon FCC conventional cell in bohr (experimental: 10.263 bohr = 5.431 Å)
a = 10.263
lattice = a / 2 * [[ 0.0  1.0  1.0];
                   [ 1.0  0.0  1.0];
                   [ 1.0  1.0  0.0]]

# Two-atom FCC primitive cell (diamond structure)
positions = [
    [0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25],
]

model_hf = model_HF(lattice,
                    atoms,
                    positions,
                    symmetries = false)

nelec     = model_hf.n_electrons
nbocc     = floor(Int, nelec / 2)
nbbig     = 4 * nbocc

Ecut      = 24
basis_hf  = PlaneWaveBasis(model_hf;  Ecut=Ecut, kgrid=[1, 1, 1])

Etol = 1e-10 / 27.211 #ΔE convergence threshold of 1e-10 eV

# ── PCDIIS solver with ψ_ref provided ─────────────────────────────────────────
scfres_hf = self_consistent_field(basis_hf;
                                  maxiter      = 1,
                                  seed         = 1234,
                                  nbandsalg    = FixedBands(;n_bands_converge=nbbig))

scfres_hf = self_consistent_field(basis_hf;
   			                      is_converged = ScfConvergenceEnergy(Etol),
                                  seed         = 1234,
                                  ρ            = scfres_hf.ρ,
                                  ψ            = [ψk[:, begin:nbocc+3] for ψk in scfres_hf.ψ],
                                  occupation   = [ok[begin:nbocc+3] for ok in scfres_hf.occupation],
                                  eigenvalues  = [ek[begin:nbocc+3] for ek in scfres_hf.eigenvalues],
                                  mixing       = SimpleMixing(),
                                  solver       = DFTK.scf_pcdiis_solver(;nb=nbbig, ψ_ref=scfres_hf.ψ))

# ── PCDIIS solver with ψ_ref set automatically ────────────────────────────────
#Note: Here the number of bands is kept high throughout the whole calculation!
scfres_hf = self_consistent_field(basis_hf;
   			                      is_converged = ScfConvergenceEnergy(Etol),
                                  seed         = 1234,
                                  nbandsalg    = FixedBands(;n_bands_converge=nbbig),
                                  mixing       = SimpleMixing(),
                                  solver       = DFTK.scf_pcdiis_solver(;nb=nbbig))

# ── Damping solver ───────────────────────────────────────────────────────────
scfres_hf = self_consistent_field(basis_hf;
   			                      is_converged = ScfConvergenceEnergy(Etol),
                                  seed         = 1234,
                                  mixing       = SimpleMixing(),
                                  solver       = DFTK.scf_damping_solver())

