# # Hubbard correction (DFT+U)
# In this example, we'll plot the DOS and projected DOS of Nickel Oxide 
# with and without the Hubbard term correction.

using DFTK
using Printf
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
Ni, O = ElementPsp(:Ni, pseudopotentials), ElementPsp(:O, pseudopotentials)
atoms = [Ni, O, Ni, O]
positions = [zeros(3), ones(3) / 4, ones(3) / 2, ones(3) * 3 / 4]
magnetic_moments = [2, 0, 2, 0]
 
# First, we run an SCF and NSCF without the Hubbard term
model = model_DFT(lattice, atoms, positions; temperature=5e-3,
                  functionals = PBE(), magnetic_moments=magnetic_moments)
basis = PlaneWaveBasis(model; Ecut = 15, kgrid = [2, 2, 2] )
scfres = self_consistent_field(basis; tol=1e-10, ρ=guess_density(basis, magnetic_moments))
bands = compute_bands(scfres, MonkhorstPack(2, 2, 2); ρ=scfres.ρ)

εF = bands.εF
width = 5.0u"eV"
εrange = (εF - austrip(width), εF + austrip(width))
band_gap = bands.eigenvalues[1][25] - bands.eigenvalues[1][24]

# Then we plot the DOS and the PDOS for the relevant 3D (pseudo)atomic projector
p = plot_dos(bands; εrange, temperature=2e-3, colors=[:red, :red])
p = plot_pdos(bands; p, temperature=2e-3, iatom=1, label="3D", colors=[:yellow, :orange], εrange)

# To perform and Hubbard computation, we have to define the Hubbard manifold and associated constant
U = 10u"eV"
manifold = DFTK.OrbitalManifold(;species=:Ni, label="3D")

# Run SCF
model = model_DFT(lattice, atoms, positions; extra_terms=[DFTK.Hubbard(manifold, U)],
                  functionals = PBE(), temperature=5e-3, magnetic_moments=magnetic_moments)
basis = PlaneWaveBasis(model; Ecut = 15, kgrid = [2, 2, 2] )
scfres = self_consistent_field(basis; tol=1e-10, ρ=guess_density(basis, magnetic_moments))

# Run NSCF
bands_hub = compute_bands(scfres, MonkhorstPack(2, 2, 2); ρ=scfres.ρ)

εF = bands_hub.εF
εrange = (εF - austrip(width), εF + austrip(width))
band_gap = bands_hub.eigenvalues[1][26] - bands_hub.eigenvalues[1][25]

# With the electron localization introduced by the Hubbard term, the band gap has now opened, 
# reflecting the experimental insulating behaviour of Nickel Oxide.
p = plot_dos(bands_hub; p, colors=[:blue, :blue], temperature=2e-3, εrange)
plot_pdos(bands_hub; p, temperature=2e-3, iatom=1, label="3D", colors=[:green, :purple], εrange)

