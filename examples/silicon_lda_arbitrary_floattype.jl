using DFTK
using DoubleFloats

# Calculation parameters
kgrid = [3, 3, 3]
Ecut = 15  # Hartree
T = Double64  # Try Float32, Double32, BigFloat (very slow!)

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Setup LDA model and discretization
model = model_DFT(Array{T}(lattice), atoms, [:lda_x, :lda_c_vwn])
basis = PlaneWaveBasis(model, Ecut, kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-6)

# Print obtained energies
println()
display(scfres.energies)
@assert eltype(sum(values(scfres.energies))) == T
