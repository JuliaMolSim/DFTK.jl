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
Ecut = 15  # Hartree
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
kpoints = [recip_lattice.get_cartesian_coords(mp[1]) for mp in bzmesh]
kweigths = [mp[2] for mp in bzmesh]
kweigths = kweigths / sum(kweigths)

#
# SCF calculation in DFTK
#
# Construct basis: transpose is required, since pymatgen uses rows for the
# lattice vectors and DFTK uses columns
grid_size = DFTK.determine_grid_size(A', Ecut, kpoints=kpoints) * ones(Int, 3)
basis = PlaneWaveBasis(A', grid_size, Ecut, kpoints, kweigths)

# Construct a local pseudopotential
hgh = load_psp("si-pade-q4.hgh")
positions = [s.frac_coords for s in structure.sites]
psp_local = build_local_potential(basis, positions,
                                  G -> DFTK.eval_psp_local_fourier(hgh, basis.recip_lattice * G))
psp_nonlocal = PotNonLocal(basis, :Si => positions, :Si => hgh)
n_electrons = 8  # In a Silicon psp model, the number of electrons per unit cell is 8

# Construct a Hamiltonian (Kinetic + local psp + nonlocal psp + Hartree)
ham = Hamiltonian(basis, pot_local=psp_local,
                  pot_nonlocal=psp_nonlocal,
                  pot_hartree=PotHartree(basis))

ρ = guess_gaussian_sad(basis, :Si => positions, :Si => mg.Element("Si").number,
                       :Si => hgh.Zion)
scfres = self_consistent_field(ham, Int(n_electrons / 2 + 2), n_electrons, ρ=ρ, tol=1e-5,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1))
ρ, pot_hartree_values, pot_xc_values = scfres


# TODO Some routine to compute this properly
efermi = 0.5

#
# Band structure calculation in DFTK
#
# Get the kpoints at which the band structure should be computed
symm_kpath = symmetry.bandstructure.HighSymmKpath(structure)
kpoints, klabels = symm_kpath.get_kpoints(kline_density, coords_are_cartesian=true)
println("Computing bands along kpath:\n     $(join(symm_kpath.kpath["path"][1], " -> "))")


# TODO Maybe think about some better mechanism here:
#      This kind of feels implicit, since it also replaces the kpoints
#      from potential other references to the ham or PlaneWaveBasis object.
kweigths = ones(length(kpoints)) ./ length(kpoints)
set_kpoints!(ham.basis, kpoints, kweigths)

# TODO This is super ugly, but needed, since the PotNonLocal implicitly
#      stores information per k-Point at the moment
if ham.pot_nonlocal !== nothing
    psp_nonlocal = PotNonLocal(ham.basis, :Si => positions, :Si => hgh)
else
    psp_nonlocal = nothing
end
ham = Hamiltonian(ham.basis, pot_local=ham.pot_local, pot_nonlocal=psp_nonlocal,
                  pot_hartree=ham.pot_hartree, pot_xc=ham.pot_xc)


# Compute bands:
band_data = lobpcg(ham, n_bands, pot_hartree_values=pot_hartree_values, tol=1e-5,
                   pot_xc_values=pot_xc_values, prec=PreconditionerKinetic(ham, α=0.5))
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

bs = elec_structure.bandstructure.BandStructureSymmLine(
    kpoints, eigenvals, recip_lattice, efermi,
    labels_dict=labels_dict, coords_are_cartesian=true
)

# Plot resulting bandstructure object
bsplot = plotter.BSPlotter(bs)
plt = bsplot.get_plot()
plt.autoscale()
plt.savefig("silicon_noXC.pdf")
plt.legend()
plt.show()
