using PyCall
using DFTK
using Libxc: Functional
using Printf
mg = pyimport("pymatgen")
symmetry = pyimport("pymatgen.symmetry")
elec_structure = pyimport("pymatgen.electronic_structure")
plotter = pyimport("pymatgen.electronic_structure.plotter")

#
# Calculation parameters
#
kgrid = [4, 4, 4]
Ecut = 15  # Hartree
n_bands = 8
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
# SCF calculation in DFTK
#
# Setup model for silicon and the list of silicon positions
Si = Species(mg.Element("Si").number, psp=load_psp("si-pade-q4.hgh"))
composition = [Si => [s.frac_coords for s in structure.sites if s.species_string == "Si"]]
n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in composition)

# Get k-Point mesh for Brillouin-zone integration
# Note: The transpose is required, since pymatgen uses rows for the
# lattice vectors and DFTK uses columns
kpoints, ksymops = bzmesh_ir_wedge(kgrid, A', composition...)
kweights = [length(symops) for symops in ksymops]
kweights = kweights / sum(kweights)

# Construct plane-wave basis
grid_size = determine_grid_size(A', Ecut, kpoints=kpoints) * ones(Int, 3)
basis = PlaneWaveBasis(A', grid_size, Ecut, kpoints, kweights, ksymops)

# Construct Hamiltonian
ham = Hamiltonian(basis, pot_local=build_local_potential(basis, composition...),
                  pot_nonlocal=build_nonlocal_projectors(basis, composition...),
                  pot_hartree=PotHartree(basis),
                  pot_xc=PotXc(basis, :lda_xc_teter93)
                 )

# Build a guess density and run the SCF
ρ = guess_gaussian_sad(basis, composition...)
scfres = self_consistent_field(ham, Int(n_electrons / 2 + 2), n_electrons, ρ=ρ, tol=1e-6,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                               n_conv_check=Int(n_electrons / 2),)

energies = scfres.energies
energies[:Ewald] = energy_nuclear_ewald(basis.lattice, composition...)
energies[:PspCorrection] = energy_nuclear_psp_correction(basis.lattice, composition...)
println("\nEnergy breakdown:")
for key in sort([keys(energies)...]; by=S -> string(S))
    @printf "    %-20s%-10.7f\n" string(key) energies[key]
end
@printf "\n    %-20s%-15.12f\n\n" "total" sum(values(energies))


# TODO Some routine to compute this properly
efermi = 0.5

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

# TODO This is super ugly, but needed, since the PotNonLocal implicitly
#      stores information per k-Point at the moment
if ham.pot_nonlocal !== nothing
    pot_nonlocal = build_nonlocal_projectors(ham.basis, composition...)
else
    pot_nonlocal = nothing
end
ham = Hamiltonian(ham.basis, pot_local=ham.pot_local, pot_nonlocal=pot_nonlocal,
                  pot_hartree=ham.pot_hartree, pot_xc=ham.pot_xc)


# Compute bands:
band_data = lobpcg(ham, n_bands, pot_hartree_values=scfres.pot_hartree_values, tol=1e-5,
                   pot_xc_values=scfres.pot_xc_values,
                   prec=PreconditionerKinetic(ham, α=0.5))
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
    kpoints, eigenvals, lattice.reciprocal_lattice, efermi,
    labels_dict=labels_dict, coords_are_cartesian=true
)

# Plot resulting bandstructure object
bsplot = plotter.BSPlotter(bs)
plt = bsplot.get_plot()
plt.autoscale()
plt.savefig("silicon_lda.pdf")
plt.legend()
plt.show()
