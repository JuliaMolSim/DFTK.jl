using DFTK
using AMDGPU

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

architecture = DFTK.GPU(AMDGPU.ROCArray)
model = Model(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1), architecture)
println("basis done")

psi = DFTK.random_orbitals(basis, basis.kpoints[1], 6)
println("orbital done")

ham = Hamiltonian(basis)
println("ham done")

res = ham * psi
println("apply done")
