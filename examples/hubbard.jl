# # Hubbard correction (DFT+U)
# In this example, we'll plot the DOS and projected DOS of Nickel Oxide 
# with and without the Hubbard term correction.

using DFTK
using PseudoPotentialData
using Unitful
using UnitfulAtomic
using Plots

# Define the geometry and pseudopotential
a = 7.9  # Nickel Oxide lattice constant in Bohr
lattice = a * [[ 1.0  0.5  0.5];
               [ 0.5  1.0  0.5];
               [ 0.5  0.5  1.0]]
pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Ni = ElementPsp(:Ni, pseudopotentials)
O  = ElementPsp(:O, pseudopotentials)
atoms = [Ni, O, Ni, O]
positions = [zeros(3), ones(3) / 4, ones(3) / 2, ones(3) * 3 / 4]
magnetic_moments = [2, 0, -1, 0]
 
# First, we run an SCF and band computation without the Hubbard term
model = model_DFT(lattice, atoms, positions; temperature=5e-3, functionals=PBE(), magnetic_moments)
basis = PlaneWaveBasis(model; Ecut=32, kgrid=[2, 2, 2])
scfres = self_consistent_field(basis; tol=1e-10, ρ=guess_density(basis, magnetic_moments))
bands = compute_bands(scfres, MonkhorstPack(4, 4, 4))
lowest_unocc_band = findfirst(ε -> ε-bands.εF > 0, bands.eigenvalues[1])
band_gap = bands.eigenvalues[1][lowest_unocc_band] - bands.eigenvalues[1][lowest_unocc_band-1]

# Then we plot the DOS and the PDOS for the relevant 3D (pseudo)atomic projector
εF = bands.εF
width = 5.0u"eV"
εrange = (εF - austrip(width), εF + austrip(width))
p = plot_dos(bands; εrange, colors=[:red, :red])
plot_pdos(bands; p, iatom=1, label="3D", colors=[:yellow, :orange], εrange)

# To perform and Hubbard computation, we have to define the Hubbard manifold and associated constant
U = 10u"eV"
manifold = OrbitalManifold([1,3], Ni.psp, 2, 1)

# Run SCF with a DFT+U setup, notice the `extra_terms` keyword argument, setting up the Hubbard +U term.
model = model_DFT(lattice, atoms, positions; extra_terms=[Hubbard(manifold, U)],
                  functionals=PBE(), temperature=5e-3, magnetic_moments)
basis = PlaneWaveBasis(model; Ecut=32, kgrid=[2, 2, 2] )
scfres = self_consistent_field(basis; tol=1e-10, ρ=guess_density(basis, magnetic_moments))

# Run band computation
bands_hub = compute_bands(scfres, MonkhorstPack(4, 4, 4))
lowest_unocc_band = findfirst(ε -> ε-bands_hub.εF > 0, bands_hub.eigenvalues[1])
band_gap = bands_hub.eigenvalues[1][lowest_unocc_band] - bands_hub.eigenvalues[1][lowest_unocc_band-1]

# With the electron localization introduced by the Hubbard term, the band gap has now opened, 
# reflecting the experimental insulating behaviour of Nickel Oxide.
εF = bands_hub.εF
εrange = (εF - austrip(width), εF + austrip(width))
p = plot_dos(bands_hub; p, colors=[:blue, :blue], εrange)
plot_pdos(bands_hub; p, iatom=1, label="3D", colors=[:green, :purple], εrange)
