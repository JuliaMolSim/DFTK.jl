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

if false
    # Full problem
    model  = model_PBE(lattice, atoms, positions)
    basis  = PlaneWaveBasis(model; Ecut=20, kgrid=(3, 3, 3), architecture)
    scfres = self_consistent_field(basis; tol=1e-2, solver=scf_damping_solver())
elseif false
    # Slightly reduced problem
    model = model_atomic(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=(1, 1, 1), architecture)
else
    # Simplest failing problem
    model = Model(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=(1, 1, 1), architecture)
end

println("basis done")
psi = DFTK.random_orbitals(basis, basis.kpoints[1], 6)
@show typeof(psi)
println("psi done")

ham = Hamiltonian(basis)
println("ham done")

res = ham * psi
println("apply done")

if false  # Do full eigensolve
    eigres = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, 6);
    println("eigensolve done")
end
