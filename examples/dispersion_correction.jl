# # Dispersion correction (DFT-D3)
#
# In this example we use the DFT-D3 dispersion correction to compute how strongly the
# layers of graphite bind. Semilocal functionals such as LDA or PBE depend only on the
# density (and its gradient) at each point, so they miss the long-range dispersion
# (van der Waals) attraction between parts of a system that are not chemically bonded.
# This attraction holds together layered materials, molecular crystals and molecules
# adsorbed on surfaces.
#
# The DFT-D3 method[^GAEK2010] adds it to the DFT energy as a sum of pair terms
# ``-C_6/r^6`` and ``-C_8/r^8``, damped at short distances. The coefficients depend on the
# element and the surroundings of each atom. The D3 energy depends only on the atomic
# positions, so it does not enter the Kohn-Sham Hamiltonian and leaves the SCF unchanged.
# It contributes to energies, forces and stresses.
#
# DFTK provides D3 through [SimpleDFTD3](https://github.com/JuliaMolSim/SimpleDFTD3.jl),
# a Julia wrapper of the [simple-dftd3](https://github.com/dftd3/simple-dftd3) library.
# The [`DFTD3`](@ref) term becomes available once SimpleDFTD3 is loaded.
#
# !!! warning "Experimental feature"
#     D3 is only available for fully periodic 3D systems, and not yet for phonons or
#     second derivatives of the energy such as elastic constants.
#
# [^GAEK2010]:
#     S. Grimme, J. Antony, S. Ehrlich and H. Krieg
#     *A consistent and accurate ab initio parametrization of density functional
#     dispersion correction (DFT-D) for the 94 elements H-Pu*
#     [J. Chem. Phys. **132**, 154104 (2010)](https://doi.org/10.1063/1.3382344).
#
# [^GEG2011]:
#     S. Grimme, S. Ehrlich and L. Goerigk
#     *Effect of the damping function in dispersion corrected density functional theory*
#     [J. Comput. Chem. **32**, 1456 (2011)](https://doi.org/10.1002/jcc.21759).

using DFTK
using SimpleDFTD3
using PseudoPotentialData
using Unitful
using UnitfulAtomic
using Plots

# We consider AB-stacked graphite with two carbon layers per cell; `d` is the distance
# between the layers. Graphite is a semimetal, so we use a small smearing.

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")

function graphite_model(d)
    a = austrip(2.46u"Å")
    c = 2d
    lattice = [a  -a/2      0;
               0  a*√3/2    0;
               0  0         c]
    C = ElementPsp(:C, pseudopotentials)
    positions = [[0, 0, 1/4], [0, 0, 3/4], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
    model_DFT(lattice, fill(C, 4), positions; functionals=PBE(), extra_terms=[DFTD3()],
              temperature=0.01, smearing=Smearing.Gaussian())
end;

# `DFTD3()` uses the D3(BJ) parameters, i.e. D3 with the rational damping of Becke and
# Johnson[^GEG2011], fitted for the functional of the model (here PBE).
# The D3 energy appears as its own entry among the energies:

basis  = PlaneWaveBasis(graphite_model(austrip(3.4u"Å")); Ecut=20, kgrid=[6, 6, 2])
scfres = self_consistent_field(basis; tol=1e-6)
scfres.energies

# Since the D3 energy does not affect the SCF, the PBE energy without D3 is the total
# energy minus the D3 entry, so one SCF per distance gives both curves.
# We measure the energies per atom relative to well-separated layers (`d` = 7 Å).

function energies_per_atom(d)
    basis  = PlaneWaveBasis(graphite_model(d); Ecut=20, kgrid=[6, 6, 2])
    scfres = self_consistent_field(basis; tol=1e-6, callback=identity)
    with_d3 = scfres.energies.total / 4
    (; with_d3, without_d3=with_d3 - scfres.energies["DFTD3"] / 4)
end

distances_Å = [3.0, 3.2, 3.4, 3.6, 4.0, 5.0]
far_apart = energies_per_atom(austrip(7.0u"Å"))
energies  = energies_per_atom.(austrip.(distances_Å * u"Å"))

to_meV(E) = ustrip(auconvert(u"meV", E))
pbe    = [to_meV(e.without_d3 - far_apart.without_d3) for e in energies]
pbe_d3 = [to_meV(e.with_d3 - far_apart.with_d3) for e in energies]

p = plot(distances_Å, [pbe pbe_d3]; label=["PBE" "PBE + D3"], markershape=:circle,
         xlabel="Distance between layers (Å)", ylabel="Energy per atom (meV)")
vline!(p, [3.35]; label="Experiment", linestyle=:dash, color=:gray)

# PBE alone barely binds the layers. With D3 the energy has a clear minimum close to the
# experimental interlayer distance of 3.35 Å. Note that with the small `Ecut` and `kgrid`
# we use to generate the online documentation the calculation is far from converged,
# see [Performing a convergence study](@ref).
#
# ## Choosing the parameters
# Instead of the default, any damping from SimpleDFTD3 can be passed, for example the
# rational damping including the three-body Axilrod-Teller-Muto term, or the original zero
# damping. The real-space cutoffs (in Bohr) are set with `cutoffs`:

DFTD3(RationalDamping("pbe"; atm=true))
DFTD3(ZeroDamping("pbe"))
DFTD3(; cutoffs=(; disp2=80.0, disp3=50.0))
nothing  # hide

# The D3 contribution is included in [`compute_forces_cart`](@ref) and
# [`compute_stresses_cart`](@ref), and hence also in geometry optimizations
# (see [Geometry optimization](@ref)).
