using DFTK
using Plots

# This example plots the free-electron bands for a 1D system.
# TODO Convert into an example in the docs
#      show how one could add a potential to this and see the bands change.

lattice = diagm([5., 0, 0])
model   = Model(lattice; n_electrons=4, terms=[Kinetic()])
basis   = PlaneWaveBasis(model; Ecut=300, kgrid=(1, 1, 1));

n_bands = 6
ρ0 = guess_density(basis)  # Just dummy, has no meaning in this model
p = plot_bandstructure(basis, ρ0, n_bands, kline_density=15)
