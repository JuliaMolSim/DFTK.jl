using PyCall
using DFTK
mg = pyimport("pymatgen")
symmetry = pyimport("pymatgen.symmetry")
elec_structure = pyimport("pymatgen.electronic_structure")
plotter = pyimport("pymatgen.electronic_structure.plotter")

#
# Calculation parameters
#
kgrid = [3, 3, 3]
Ecut = 5  # Hartree
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
structure = mg.Structure(lattice, ["Si", "Si"], [ones(3)/8, -ones(3)/8])

#
# Model and basis setup in DFTK
#
n_electrons = 8
model = Model(A', n_electrons)  # free-electron model

# Setup basis (with uniform k-Point mesh)
kpoints, ksymops = bzmesh_uniform(kgrid)
kweights = ones(length(kpoints)) / length(kpoints)
fft_size = determine_grid_size(A', Ecut)
basis = PlaneWaveModel(model, fft_size, Ecut, kpoints, kweights, ksymops)

ham = Hamiltonian(basis)

STOP


#
# Band structure calculation in DFTK
#
# Get the kpoints at which the band structure should be computed
symm_kpath = symmetry.bandstructure.HighSymmKpath(structure)
kpoints, klabels = symm_kpath.get_kpoints(kline_density, coords_are_cartesian=false)
println("Computing bands along kpath:\n     $(join(symm_kpath.kpath["path"][1], " -> "))")

# TODO Maybe think about some better mechanism here:
#      This kind of feels implicit, since it also replaces the kpoints
#      from potential other references to the ham or PlaneWaveBasis object.
set_kpoints!(ham.basis, kpoints)

# Compute bands:
band_data = lobpcg(ham, n_bands, prec=PreconditionerKinetic(ham, α=0.5),
                   interpolate_kpoints=false)
if ! band_data.converged
    println("WARNING: Not all k-points converged.")
end

#
# Band structure plotting in pymatgen
#
# Transform band_data to datastructure used in pymatgen
eigenvals_spin_up = Matrix{eltype(band_data.λ[1])}(undef, n_bands, length(kpoints))
for (ik, λs) in enumerate(band_data.λ)
    eigenvals_spin_up[:, ik] = λs
end
eigenvals = Dict(elec_structure.core.Spin.up => eigenvals_spin_up)

labels_dict = Dict{String, Vector{eltype(kpoints[1])}}()
for (ik, k) in enumerate(kpoints)
    if length(klabels[ik]) > 0
        labels_dict[klabels[ik]] = k
    end
end

efermi = 0.5  # Just an invented number
bs = elec_structure.bandstructure.BandStructureSymmLine(
    kpoints, eigenvals, lattice.recip_lattice, efermi,
    labels_dict=labels_dict, coords_are_cartesian=true
)

# Plot resulting bandstructure object
bsplot = plotter.BSPlotter(bs)
plt = bsplot.get_plot()
plt.autoscale()
plt.savefig("silicon_free_electron.pdf")
plt.show()
