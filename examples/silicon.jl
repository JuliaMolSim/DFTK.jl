# Very basic setup, useful for testing
using DFTK

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
magnetic_moments = [1.0, -1.0]

model  = model_LDA(lattice, atoms, positions; magnetic_moments)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
ρ = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis, tol=1e-3; ρ)
