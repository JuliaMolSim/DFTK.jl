# # Dispersion correction (DFT-D3)
#
# In this example we compute how strongly two layers of graphite bind. Since
# semilocal functionals such as LDA or PBE depend only on the density they miss
# the long-range dispersion (van der Waals) attraction between parts of a
# system that are not chemically bonded. The DFT-D3 method[^GAEK2010] adds such
# dispersions to the DFT energy as a sum of pair terms ``-C_6/r^6`` and
# ``-C_8/r^8``, damped at short distances. The coefficients depend on the
# element and the surroundings of each atom.
#
# DFTK provides D3 through [SimpleDFTD3](https://github.com/JuliaMolSim/SimpleDFTD3.jl),
# a Julia wrapper of the [simple-dftd3](https://github.com/dftd3/simple-dftd3) library.
# Once SimpleDFTD3 is loaded, D3 is added by passing a [`DFTD3`](@ref) term in the
# `extra_terms` of [`model_DFT`](@ref).
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

# We consider AB-stacked graphite with two carbon layers per cell. The distance
# `d` between the layers we keep as a variable.


function graphite_basis(d)
    a = austrip(2.46u"Å")
    c = 2d
    lattice = [a  -a/2      0;
               0  a*√3/2    0;
               0  0         c]
    C = ElementPsp(:C, PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))
    positions = [[0, 0, 1/4], [0, 0, 3/4], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
    functionals = PBE()
    model = model_DFT(lattice, fill(C, 4), positions;
                      functionals, extra_terms=[DFTD3(functionals)],
                      temperature=0.01, smearing=Smearing.Gaussian())
    PlaneWaveBasis(model; Ecut=20, kgrid=[6, 6, 2])
end;

# Here, `DFTD3(PBE())` uses D3(BJ), i.e. D3 with the rational damping of Becke
# and Johnson[^GEG2011], with the parameters fitted for PBE, the functional of
# the model.

basis  = graphite_basis(austrip(3.4u"Å"))
scfres = self_consistent_field(basis; tol=1e-5)
scfres.energies

# Since the D3 energy term does not depend on the density, the PBE energy
# is the total energy minus the D3 entry of `scfres.energies`, so this function
# computes both:

function energies_per_atom(d)
    n_atoms = 4
    scfres  = self_consistent_field(graphite_basis(d); tol=1e-5, callback=identity)
    with_d3    = scfres.energies.total / n_atoms
    without_d3 = with_d3 - scfres.energies["DFTD3"] / n_atoms
    (; with_d3, without_d3)
end;

# We measure the energies per atom relative to well-separated layers (`d` = 7 Å):

distances = [3.0, 3.2, 3.4, 3.6, 4.0, 5.0]u"Å"
far_apart = energies_per_atom(austrip(7.0u"Å"))
energies  = energies_per_atom.(austrip.(distances))

pbe    = [auconvert(u"meV", e.without_d3 - far_apart.without_d3) for e in energies]
pbe_d3 = [auconvert(u"meV", e.with_d3    - far_apart.with_d3)    for e in energies]

p = plot(distances, [pbe pbe_d3]; label=["PBE" "PBE + D3"], markershape=:circle,
         xlabel="Distance between layers", ylabel="Energy per atom")
vline!(p, [3.35u"Å"]; label="Experiment", linestyle=:dash, color=:gray)

# PBE alone barely binds the layers. With D3 the energy has a clear minimum close to the
# experimental interlayer distance of 3.35 Å. Note that with our `Ecut` and
# `kgrid` we are not yet fully converged, see [Performing a convergence study](@ref).
#
# ## Choice of D3 parameters
# Other dampings can be selected by their identifier, for example the original zero
# damping, here with the three-body Axilrod-Teller-Muto term.
# The pair terms can also be
# switched off smoothly at the real-space cutoff (width in Bohr):

DFTD3(PBE(); damping="D3(0)", three_body=true)
DFTD3(PBE(); switching_width_2body=5.0)
nothing  # hide

# Note that instead of passing a functional as a first argument, you can also
# pass a string describing the model. These are the model strings supported
# by simple-dftd3.

DFTD3("pbe"; damping="D3(BJ)")
nothing  # hide

# Damping parameters can also be given explicitly with a SimpleDFTD3 damping, and a
# SimpleDFTD3 `DispersionModel` gives full control over the evaluation, e.g. the
# real-space cutoffs (in Bohr):

DFTD3(RationalDamping(s8=0.7875, a1=0.4289, a2=4.4407))
dispersion_model = DispersionModel(basis.model.lattice, basis.model.atoms, basis.model.positions)
set_realspace_cutoff!(dispersion_model; disp2=80.0, disp3=50.0)
DFTD3(PBE(), dispersion_model)
nothing  # hide
