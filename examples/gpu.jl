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

if has_cuda()
    # Use CUDA to store DFT quantities and perform main computations
    # For this we set the array_type for storing DFT quantities to a GPU array type
    array_type = CuArray
else
    array_type = Array  # Keep using the CPU
end

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(1, 1, 1), array_type)
scfres = self_consistent_field(basis; tol=1e-3,
                               solver=scf_anderson_solver(),
                               mixing=KerkerMixing())
