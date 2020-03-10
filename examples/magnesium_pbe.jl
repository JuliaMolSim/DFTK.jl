using DFTK
using Plots
Plots.pyplot()  # Use PyPlot backend for unicode support

# Calculation parameters
kgrid = [4, 4, 4]        # k-Point grid
Ecut = 15                # kinetic energy cutoff in Hartree
supercell = [1, 1, 1]    # Lattice supercell
n_bands = 8              # Number of bands for SCF and plotting
Tsmear = 0.01            # Smearing temperature in Hartree

# Setup magnesium lattice (constants in Bohr)
a = 3.0179389193174084
b = 5.227223542397263
c = 9.773621942589742
lattice = [[-a -a  0]; [-b  b  0]; [0   0 -c]]
Mg = ElementPsp(:Mg, psp=load_psp("hgh/pbe/Mg-q2"))
atoms = [Mg => [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Mg => [s.frac_coords for s in pystruct.sites]]

# Setup PBE model with Methfessel-Paxton smearing and its discretisation
model = model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe];
                  temperature=Tsmear,
                  smearing=DFTK.Smearing.MethfesselPaxton1())
basis = PlaneWaveBasis(model, Ecut, kgrid=kgrid)

# Run SCF
scfres = self_consistent_field(basis, n_bands=n_bands)

# Print obtained energies and plot bands
println()
display(scfres.energies)
p = plot_bandstructure(scfres, n_bands)

# Plot DOS
εs = range(minimum(minimum(scfres.eigenvalues)) - 1,
           maximum(maximum(scfres.eigenvalues)) + 1, length=1000)
Ds = DOS.(εs, Ref(basis), Ref(scfres.eigenvalues), T=Tsmear*4,
          smearing=DFTK.Smearing.MethfesselPaxton1())
q = plot(εs, Ds, label="DOS")
vline!(q, [scfres.εF], label="εF")

gui(plot(p, q))
