using DFTK
using AMDGPU
using PseudoPotentialData
using AbstractFFTs
using LinearAlgebra

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

architecture = DFTK.GPU(AMDGPU.ROCArray)
model = Model(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1), architecture)
println("basis done")

psi = [DFTK.random_orbitals(basis, basis.kpoints[1], 6)]
println("orbital done")

ham = Hamiltonian(basis)
println("ham done")

X = ComplexF64.(roc(Float64.(randn(12, 12, 12))));
P = AbstractFFTs.plan_fft(X);
Y = copy(X)
mul!(Y, P, X)
println("fft prepared")

res = ham * psi
println("apply done")
