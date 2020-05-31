# # 2D Gross-Pitaevskii equation

# We solve the 2D Gross-Pitaevskii equation with a magnetic field. This is similar to the 1D case, but with an extra magnetic field.

using DFTK, StaticArrays, Plots

# Unit cell. Having one lattice vectors as zero means a 2D system
a = 10
lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]]

# Confining scalar potential, and magnetic potential
pot(x, y, z) = (x-a/2)^2 + (y-a/2)^2  # potential
Apot(x, y, z) = .2 * @SVector [y-a/2, -(x-a/2), 0]
Apot(X) = Apot(X...)


# Parameters
Ecut = 40
C = 500.0
α = 2
n_electrons = 1  # increase this for fun

# Add all the needed terms, and run the model
terms = [Kinetic(),
         ExternalFromReal(X -> pot(X...)),
         PowerNonlinearity(C, α),
         Magnetic(Apot),
]
model = Model(lattice; n_electrons=n_electrons,
              terms=terms, spin_polarization=:spinless)  # "spinless fermions"
basis = PlaneWaveBasis(model, Ecut)
scfres = direct_minimization(basis, tol=1e-8)
heatmap(scfres.ρ.real[:,:,1], c=:blues)
