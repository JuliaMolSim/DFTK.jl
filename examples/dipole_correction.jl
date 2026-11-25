# # Dipole corrections
#
# An asymmetric, dipolar slab has different electrostatic potentials on its two
# sides. Ordinary periodic boundary conditions force these two vacuum levels to be
# equal, which introduces an artificial electric field. The Bengtsson dipole
# correction adds a potential to counteract this effect plus an associated energy
# correction ([Phys. Rev. B 59, 12301 (1999)](https://doi.org/10.1103/PhysRevB.59.12301)).
# We illustrate the dipole correction using a layer of water molecules. 

using DFTK
using LinearAlgebra
using PseudoPotentialData
using Statistics
using Unitful
using UnitfulAtomic

lateral = 6.0   # Distance of water molecules
width   = 30.0  # Width of the slab
lattice = [lateral 0 0;
           0 lateral 0;
           0 0   width]

pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
H = ElementPsp(:H, pseudopotentials)
O = ElementPsp(:O, pseudopotentials)

bond_length = austrip(0.958u"Å")
half_angle = deg2rad(104.45) / 2
bond_y = bond_length * sin(half_angle)
bond_z = bond_length * cos(half_angle)
center = [lateral, lateral, width] / 2
cartesian_positions = [
    center,                                            # O
    center + [0,  bond_y, -bond_z],                    # H
    center + [0, -bond_y, -bond_z],                    # H
]
positions = [r ./ [lateral, lateral, width] for r in cartesian_positions]
atoms = [O, H, H];

# We now solve the problem without corrections

model = model_DFT(lattice, atoms, positions; functionals=LDA(), extra_terms=[])
basis = PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])
uncorrected = self_consistent_field(basis; tol=1e-6)
uncorrected.energies

# and with dipole corrections:

model = model_DFT(lattice, atoms, positions;
                  functionals=LDA(), extra_terms=[DipoleCorrection()])
basis = PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])
corrected = self_consistent_field(basis; tol=1e-6)
corrected.energies

# By default the discontinuity is along the third coordinate direction and at
# fractional coordinate zero, which is in the vacuum region for this geometry.
# Use `position` to move it and `width` to control the width of the cubic
# smoothening used to make the sawtooth continuous. The entire smoothing region
# must remain in vacuum.
#
# To get some more insights let us plot the behaviour of the potential
# along the third coordinate direction, i.e. perpendicular to the water surface.

using Plots

function perpendicular_potential(scfres)
    ## 
    potential = DFTK.total_local_potential(scfres.ham)
    averaged = dropdims(mean(potential; dims=(1, 2)); dims=(1, 2))
    averaged[:, 1]  # First (and here only) spin component
end

z = vec([r[3] for r in r_vectors(corrected.basis)[1, 1, :]]) * width
p = plot(; xlabel="z (Bohr)", ylabel="Laterally averaged local potential (Hartree)",
         legend=:topright)
plot!(p, z, perpendicular_potential(uncorrected); label="Uncorrected")
plot!(p, z, perpendicular_potential(corrected); label="w. dipole correction")

# Away from the dipolar layer and the smoothed discontinuity, the corrected vacuum
# potentials have two different plateaus. The two work-function references are the
# respective vacuum levels minus the common Fermi level.

function vacuum_level(scfres, interval)
    z_fractional = vec([r[3] for r in r_vectors(scfres.basis)[1, 1, :]])
    selected = findall(z -> first(interval) ≤ z ≤ last(interval), z_fractional)
    mean(perpendicular_potential(scfres)[selected])
end
vacuum_left  = vacuum_level(corrected, (0.15, 0.25))
vacuum_right = vacuum_level(corrected, (0.75, 0.85))

println("Vacuum level (Left):   ", vacuum_left,
        "   Work function:  ", vacuum_left - corrected.εF, " Hartree")
println("Vacuum level (Right):  ", vacuum_right,
        "   Work function:  ", vacuum_right - corrected.εF, " Hartree")
