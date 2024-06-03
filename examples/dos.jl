# # Densities of states (DOS)
# In this example, we'll plot the DOS, local DOS, and  projected DOS of Silicon.
# DOS computation only supports finite temperature.
# Projected DOS only supports PspUpf.

using DFTK
using Unitful
using Plots
using LazyArtifacts

## Define the geometry and pseudopotential
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.0];
    [1 0 1.0];
    [1 1 0.0]]
Si = ElementPsp(:Si; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Si.upf"))
atoms = [Si, Si]
positions = [ones(3) / 8, -ones(3) / 8]

## Run SCF
model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4], symmetries_respect_rgrid=true)
scfres = self_consistent_field(basis, tol=1e-8)

## Plot the DOS
plot_dos(scfres; smearing=DFTK.Smearing.FermiDirac(), temperature = 5e-3)

## Plot the local DOS about a single axis
plot_ldos(scfres; smearing=DFTK.Smearing.FermiDirac(), temperature = 5e-3, n_points=100, ldos_xyz=[:, 10, 10])

## Plot the projected DOS
plot_pdos(scfres; smearing=DFTK.Smearing.FermiDirac(), temperature=5e-3, εrange=(-0.3, 0.5))
