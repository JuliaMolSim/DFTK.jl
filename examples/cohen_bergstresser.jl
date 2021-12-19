# # Cohen-Bergstresser model
#
# This example considers the Cohen-Bergstresser model[^CB1966],
# reproducing the results of the original paper. This model is particularly
# simple since its linear nature allows one to get away without any
# self-consistent field calculation.
#
# [^CB1966]: M. L. Cohen and T. K. Bergstresser Phys. Rev. **141**, 789 (1966) DOI [10.1103/PhysRev.141.789](https://doi.org/10.1103/PhysRev.141.789)

# We build the lattice using the tabulated lattice constant from the original paper,
# stored in DFTK:
using DFTK

Si = ElementCohenBergstresser(:Si)
lattice = Si.lattice_constant / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
atoms = [Si => [ones(3)/8, -ones(3)/8]];

# Next we build the rather simple model and discretise it with moderate `Ecut`:
model = Model(lattice; atoms=atoms, terms=[Kinetic(), AtomicLocal()])
basis = PlaneWaveBasis(model, Ecut=10.0, kgrid=(1, 1, 1));

# We diagonalise at the Gamma point to find a Fermi level ...
ham = Hamiltonian(basis)
eigres = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, 6)
εF = DFTK.fermi_level(basis, eigres.λ)

# ... and compute and plot 8 bands:
using Plots
using Unitful

p = plot_bandstructure(basis; n_bands=8, εF, kline_density=10, unit=u"eV")
ylims!(p, (-5, 6))
