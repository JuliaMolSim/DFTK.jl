# # Wannierization using Wannier.jl
#
# DFTK features an interface with the program
# [Wannier.jl](https://www.wannierjl.org/),
# in order to compute maximally-localized Wannier functions (MLWFs)
# from an initial self consistent field calculation.
# All processes are handled by calling the routine `run_wannier`.
#
# !!! warning "No guarantees on Wannier.jl interface"
#     This code is at an early stage and has so far not been fully tested.
#     Bugs are likely and we welcome issues in case you find any!
#
# This example shows how to obtain the MLWFs corresponding
# to the first five bands of graphene. Since the bands 2 to 11 are entangled,
# 15 bands are first computed to obtain 5 MLWFs by a disantanglement procedure.

using DFTK
using Unitful
using UnitfulAtomic

d = 10u"Å"
a = 2.641u"Å"  # Graphene Lattice constant
lattice = [a  -a/2    0;
           0  √3*a/2  0;
           0     0    d]

C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms     = [C, C]
positions = [[0.0, 0.0, 0.0], [1//3, 2//3, 0.0]]
model  = model_PBE(lattice, atoms, positions)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[5, 5, 1])
nbandsalg = AdaptiveBands(basis.model; n_bands_converge=15)
scfres = self_consistent_field(basis; nbandsalg, tol=1e-5);

# Plot bandstructure of the system
using Plots

plot_bandstructure(scfres; kline_density=10)

# Now we use the `run_wannier` routine to generate all files needed by
# wannier90 and to perform the Wannierization procedure using Wannier.jl.
# In Wannier90's convention, all files are named with the same prefix and only differ by
# their extensions. By default all generated input and output files are stored
# in the subfolder "wannier" under the prefix "wannier" (i.e. "wannier/wannier.win",
# "wannier/wannier.wout", etc.). A different file prefix can be given with the
# keyword argument `fileprefix` as shown below.
#
# We now solve for 5 MLWF using Wannier.jl:

using Wannier  # Needed to make run_wannier available

# The Wannier.jl interface is very similar to the Wannier90 example,
# except that the function `run_wannier` is used instead of `run_wannier90`.
# To further explore the functionalities of the MLWF interface, in this example
# we use SCDM to generate a better initial guess for the MLWFs
# (by default, `run_wannier` will use random initial guess which is not good).
# We need to first unfold the `scfres` to a MP kgrid for Wannierization,
# and remove the possibly unconverged bands (bands above `scfres.n_bands_converge`)
exclude_bands = DFTK._default_exclude_bands(scfres)
basis, ψ, eigenvalues = DFTK.unfold_scfres_wannier(scfres, exclude_bands)

# Then compute the SCDM initial guess with [`compute_amn_scdm`](@ref)
# Since this is an entangled case, we need a weight factor, using `erfc` function.
# Note that the unit of `μ` and `σ` are `Ha`, as in the DFTK convention.
# Here we choose these numbers by inspecting the band structure, you can also
# test with different values to see how it affects the result.
μ, σ = 0.0, 0.01
f = DFTK.scdm_f_erfc(basis, eigenvalues, μ, σ)
# and construct 5 MLWFs
n_wann = 5
A = DFTK.compute_amn_scdm(basis, ψ, n_wann, f)
# we pass the `A` matrix to `run_wannier`, so it will skip the `compute_amn` step
wann_model, = run_wannier(
    scfres;
    fileprefix="wannier/graphene",
    n_wann,
    A,
    dis_froz_max=0.1,
);
# Note we unwrap the returned objects since in `:collinear` case, the
# `run_wannier` will return two `Wannier.Model` objects.
# As can be observed standard optional arguments for the disentanglement
# can be passed directly to `run_wannier` as keyword arguments.

# The MLWF centers and spreads can be obtained from
Wannier.omega(wann_model)

# Please refer to the [Wannier.jl documentation](https://wannierjl.org/)
# for more advanced usage of the Wannier function interface.

# (Delete temporary files.)
rm("wannier", recursive=true)
