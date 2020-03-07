using DFTK
using Plots

# Example to reproduce the silicon results of the Cohen-Bergstresser paper
# DOI 10.1103/PhysRev.141.789

Ecut = 15           # kinetic energy cutoff in Hartree
n_bands = 8         # number of bands to plot in the bandstructure
T = Float64         # Floating-point type for computation

# Setup silicon lattice using lattice constant from paper (stored in DFTK)
Si = ElementCohenBergstresser(:Si)
lattice = Si.lattice_constant / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Model, discretisation and Hamiltonian
model = Model(Matrix{T}(lattice); atoms=atoms, terms=[Kinetic(), AtomicLocal()])
basis = PlaneWaveBasis(model, Ecut)

# Diagonalise (at the Gamma point) to find Fermi level used in Cohen-Bergstresser,
# then compute the bands
ham = Hamiltonian(basis)
eigres = diagonalise_all_kblocks(DFTK.lobpcg_hyper, ham, 6)
εF = find_fermi_level(basis, eigres.λ)

ρ0 = guess_density(basis)  # Just dummy, has no meaning in this model
p = plot_bandstructure(basis, ρ0, n_bands, εF=εF, kline_density=15)
ylims!(p, (-5, 6))
println()
display(p)
