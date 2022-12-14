using DFTK
using CUDA

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
model = model_DFT(lattice, atoms, positions, []; temperature=1e-3)

# If available, use CUDA to store DFT quantities and perform main computations
architecture = has_cuda() ? DFTK.GPU(CuArray) : DFTK.CPU()

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(1, 1, 1), architecture)
scfres = self_consistent_field(basis; tol=1e-2,
                               solver=scf_damping_solver(),
                               mixing=KerkerMixing())
