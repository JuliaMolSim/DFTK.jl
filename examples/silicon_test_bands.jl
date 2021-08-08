# Very basic setup, useful for testing
using DFTK
setup_threading()

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

super = load_atoms(ase_atoms(lattice, atoms) * (2, 1, 2))
atoms = [Si => last(only(super))]

model = model_LDA(lattice, atoms, temperature=1e-3)
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])

println("Brillouin.jl")
scfres = self_consistent_field(basis, tol=1e-8)
plot_bandstructure(scfres, kline_density=5)
