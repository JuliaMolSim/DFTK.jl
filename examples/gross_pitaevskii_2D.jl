## 2D Gross-Pitaevskii equation, with magnetic field, and with
## multiple electrons (of course, it doesn't make physical sense, but
## why not)

## This is pretty WIP, and only serves as a very rough demo. Nothing
## has been checked properly, so do not use for any serious purposes.

using DFTK
using StaticArrays
using Plots

Ecut = 100
# Nonlinearity : energy C ∫ρ^α
C = 500.0
α = 2

# Unit cell. Having one lattice vectors as zero means a 2D system
a = 10
lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]]

f(x, y, z) = x^2 + y^2  # potential

Apot(x, y, z) = .2 * @SVector [y, -x, 0]
Apot(X) = Apot(X...)

n_electrons = 1  # increase this for fun
# We add the needed terms
terms = [Kinetic(),
         ExternalFromReal(X -> f(X...)),
         PowerNonlinearity(C, α),
         Magnetic(Apot),
]
model = Model(lattice; n_electrons=n_electrons,
              terms=terms, spin_polarization=:spinless)  # "spinless fermions"
basis = PlaneWaveBasis(model, Ecut)

n_bands_scf = model.n_electrons
scfres = direct_minimization(basis, x_tol=1e-8, f_tol=-1, g_tol=-1)
println()
display(scfres.energies)

heatmap(scfres.ρ.real[:,:,1], c=:blues, clims=[0, Inf])
