# Very basic setup, useful for testing
using DFTK

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model  = model_PBE(lattice, atoms, positions)
basis  = PlaneWaveBasis(model; Ecut=30, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis; tol=1e-6)

model  = model_HF(lattice, atoms, positions)
basis  = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
scfres = self_consistent_field(basis;
                               solver=DFTK.scf_damping_solver(1.0),
                               tol=1e-10, scfres.ρ, scfres.ψ,
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=5e-4))
