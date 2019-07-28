using PyCall
using DFTK
using Libxc: Functional
using Profile
# using ProfileView
using StatProfilerHTML

mg = pyimport("pymatgen")
symmetry = pyimport("pymatgen.symmetry")
elec_structure = pyimport("pymatgen.electronic_structure")
plotter = pyimport("pymatgen.electronic_structure.plotter")

#
# Calculation parameters
#
kgrid = [1,1,1]
Ecut = 25  # Hartree
n_bands = 10
kline_density = 20


#
# Setup silicon structure in pymatgen
#
a = 5.431020504 * mg.units.ang_to_bohr
A = mg.ArrayWithUnit(a / 2 .* [[0 1 1.];
                               [1 0 1.];
                               [1 1 0.]], "bohr")
lattice = mg.lattice.Lattice(A)
recip_lattice = lattice.reciprocal_lattice
structure = mg.Structure(lattice, ["Si", "Si"], [ones(3)/8, -ones(3)/8])

# Get k-Point mesh for Brillouin-zone integration
spgana = symmetry.analyzer.SpacegroupAnalyzer(structure)
bzmesh = spgana.get_ir_reciprocal_mesh(kgrid)
kpoints = [mp[1] for mp in bzmesh]
kweigths = [mp[2] for mp in bzmesh]
kweigths = kweigths / sum(kweigths)

#
# SCF calculation in DFTK
#
# Construct basis: transpose is required, since pymatgen uses rows for the
# lattice vectors and DFTK uses columns
grid_size = DFTK.determine_grid_size(A', Ecut, kpoints=kpoints) * ones(Int, 3)
basis = PlaneWaveBasis(A', grid_size, Ecut, kpoints, kweigths)

# Setup model for silicon and list of silicon positions
Si = Species(mg.Element("Si").number, psp=load_psp("si-pade-q4.hgh"))
composition = [Si => [s.frac_coords for s in structure.sites if s.species_string == "Si"]]
n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in composition)

# Construct Hamiltonian
ham = Hamiltonian(basis, pot_local=build_local_potential(basis, composition...),
                  pot_nonlocal=build_nonlocal_projectors(basis, composition...),
                  pot_hartree=PotHartree(basis),
                  pot_xc=nothing)

# Build a guess density and run the SCF
ρ = guess_gaussian_sad(basis, composition...)

Profile.clear()
@profile self_consistent_field(ham, Int(n_electrons / 2 + 2), n_electrons, ρ=ρ, tol=1e-6,
                      lobpcg_prec=PreconditionerKinetic(ham, α=0.1))
# ProfileView.view()
statprofilehtml()
