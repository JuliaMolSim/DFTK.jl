# # Densities of states (DOS)
# In this example, we'll plot the DOS, local DOS, and projected DOS of Silicon.
# DOS computation only supports finite temperature.

using DFTK
using Unitful
using Plots
using PseudoPotentialData

## Define the geometry and pseudopotential
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.0];
                   [1 0 1.0];
                   [1 1 0.0]]
Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
atoms = [Si, Si]
positions = [ones(3) / 8, -ones(3) / 8]

## Run SCF
model = model_LDA(lattice, atoms, positions; temperature=5e-3)
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4], symmetries_respect_rgrid=true)
scfres = self_consistent_field(basis, tol=1e-8)

## Plot the DOS
plot_dos(scfres)

## Plot the local DOS along one direction
plot_ldos(scfres; n_points=100, ldos_xyz=[:, 10, 10])

## Plot the projected DOS
p = plot_pdos(scfres; iatom=1, label="3S", εrange=(-0.3, 0.5))
plot_pdos(scfres; p, iatom=1, label="3P", εrange=(-0.3, 0.5))
