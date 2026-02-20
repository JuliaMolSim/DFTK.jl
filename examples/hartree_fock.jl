# Very basic setup, useful for testing
using DFTK
using PseudoPotentialData

pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
Si = ElementPsp(:Si, pd_pbe_family)

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model  = model_DFT(lattice, atoms, positions; functionals=PBE())
basis  = PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis; tol=1e-6)

# Run Hartree-Fock
model  = model_HF(lattice, atoms, positions)
basis  = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
scfres = self_consistent_field(basis;
                               solver=DFTK.scf_damping_solver(damping=1.0),
                               is_converged=ScfConvergenceEnergy(1e-7), 
                               scfres.ρ, scfres.ψ, scfres.eigenvalues, # all three needed: ρ, ψ, eigenvalues
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=5e-4))
