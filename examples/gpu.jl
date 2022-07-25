using DFTK
using CUDA
using MKL
setup_threading(n_blas=1)

a = 10.263141334305942  # Lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8];
terms_LDA = [Kinetic(), AtomicLocal(), AtomicNonlocal()]

# Setup an LDA model and discretize using
# a single k-point and a small `Ecut` of 5 Hartree.
mod = Model(lattice, atoms, positions; terms=terms_LDA,symmetries=false)
basis = PlaneWaveBasis(mod; Ecut=30, kgrid=(1, 1, 1))
basis_gpu = PlaneWaveBasis(mod; Ecut=30, kgrid=(1, 1, 1), array_type = CuArray)


DFTK.reset_timer!(DFTK.timer)
scfres = self_consistent_field(basis; solver=scf_damping_solver(1.0), is_converged=DFTK.ScfConvergenceDensity(1e-3))
println(DFTK.timer)

DFTK.reset_timer!(DFTK.timer)
scfres_gpu = self_consistent_field(basis_gpu; solver=scf_damping_solver(1.0), is_converged=DFTK.ScfConvergenceDensity(1e-3))
println(DFTK.timer)
