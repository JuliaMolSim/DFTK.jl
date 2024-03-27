# # Wannierization using Wannier.jl or Wannier90
#
# DFTK features an interface with the programs
# [Wannier.jl](https://wannierjl.org) and [Wannier90](http://www.wannier.org/),
# in order to compute maximally-localized Wannier functions (MLWFs)
# from an initial self consistent field calculation.
# All processes are handled by calling the routine `wannier_model` (for Wannier.jl) or `run_wannier90` (for Wannier90).
#
# !!! warning "No guarantees on Wannier interface"
#     This code is at an early stage and has so far not been fully tested.
#     Bugs are likely and we welcome issues in case you find any!
#
# This example shows how to obtain the MLWFs corresponding
# to the first five bands of graphene. Since the bands 2 to 11 are entangled,
# 15 bands are first computed to obtain 5 MLWFs by a disantanglement procedure.

using DFTK
using Plots
using Unitful
using UnitfulAtomic

d = 10u"Å"
a = 2.641u"Å"  # Graphene Lattice constant
lattice = [a  -a/2    0;
           0  √3*a/2  0;
           0     0    d]

C = ElementPsp(:C; psp=load_psp("hgh/pbe/c-q4"))
atoms     = [C, C]
positions = [[0.0, 0.0, 0.0], [1//3, 2//3, 0.0]]
model  = model_PBE(lattice, atoms, positions)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[5, 5, 1])
nbandsalg = AdaptiveBands(basis.model; n_bands_converge=15)
scfres = self_consistent_field(basis; nbandsalg, tol=1e-5);

# Plot bandstructure of the system

bands = compute_bands(scfres; kline_density=10)
plot_bandstructure(bands)

# ## Wannierization with Wannier.jl
#
# Now we use the `wannier_model` routine to generate a Wannier.jl model
# that can be used to perform the wannierization procedure.
# For now, this model generation produces file in the Wannier90 convention,
# where all files are named with the same prefix and only differ by
# their extensions. By default all generated input and output files are stored
# in the subfolder "wannierjl" under the prefix "wannier" (i.e. "wannierjl/wannier.win",
# "wannierjl/wannier.wout", etc.). A different file prefix can be given with the
# keyword argument `fileprefix` as shown below.
#
# We now produce a simple Wannier model for 5 MLFWs.
#
# For a good MLWF, we need to provide initial projections that resemble the expected shape
# of the Wannier functions.
# Here we will use:
# - 3 bond-centered 2s hydrogenic orbitals for the expected σ bonds
# - 2 atom-centered 2pz hydrogenic orbitals for the expected π bands

using Wannier # Needed to make Wannier.Model available

# From chemical intuition, we know that the bonds with the lowest energy are:
# - the 3 σ bonds,
# - the π and π* bonds.
# We provide relevant initial projections to help Wannierization
# converge to functions with a similar shape.
s_guess(center) = DFTK.HydrogenicWannierProjection(center, 2, 0, 0, C.Z)
pz_guess(center) = DFTK.HydrogenicWannierProjection(center, 2, 1, 0, C.Z)
projections = [
    ## Note: fractional coordinates for the centers!
    ## 3 bond-centered 2s hydrogenic orbitals to imitate σ bonds
    s_guess((positions[1] + positions[2]) / 2),
    s_guess((positions[1] + positions[2] + [0, -1, 0]) / 2),
    s_guess((positions[1] + positions[2] + [-1, -1, 0]) / 2),
    ## 2 atom-centered 2pz hydrogenic orbitals
    pz_guess(positions[1]),
    pz_guess(positions[2]),
]

# Wannierize:
wannier_model = Wannier.Model(scfres;
    fileprefix="wannier/graphene",
    n_bands=scfres.n_bands_converge,
    n_wannier=5,
    projections,
    dis_froz_max=ustrip(auconvert(u"eV", scfres.εF))+1) # maximum frozen window, for example 1 eV above Fermi level

# Once we have the `wannier_model`, we can use the functions in the Wannier.jl package:
#
# Compute MLWF:
U = disentangle(wannier_model, max_iter=200);

# Inspect localization before and after Wannierization:
omega(wannier_model)
omega(wannier_model, U)

# Build a Wannier interpolation model:
kpath = irrfbz_path(model)
interp_model = Wannier.InterpModel(wannier_model; kpath=kpath)

# And so on...
# Refer to the Wannier.jl documentation for further examples.

# (Delete temporary files when done.)
rm("wannier", recursive=true)

# ### Custom initial guesses
#
# We can also provide custom initial guesses for Wannierization,
# by passing a callable function in the `projections` array.
# The function receives the basis and a list of points (fractional coordinates in reciprocal space),
# and returns the Fourier transform of the initial guess function evaluated at each point.
#
# For example, we could use Gaussians for the σ and pz guesses with the following code:
s_guess(center) = DFTK.GaussianWannierProjection(center)
function pz_guess(center)
    ## Approximate with two Gaussians offset by 0.5 Å from the center of the atom
    offset = model.inv_lattice * [0, 0, austrip(0.5u"Å")]
    center1 = center + offset
    center2 = center - offset
    ## Build the custom projector
    (basis, ps) -> DFTK.GaussianWannierProjection(center1)(basis, ps) - DFTK.GaussianWannierProjection(center2)(basis, ps)
end
## Feed to Wannier via the `projections` as before...

# This example assumes that Wannier.jl version 0.3.2 is used,
# and may need to be updated to accommodate for changes in Wannier.jl.
#
# Note: Some parameters supported by Wannier90 may have to be passed to Wannier.jl differently,
# for example the max number of iterations is passed to `disentangle` in Wannier.jl,
# but as `num_iter` to `run_wannier90`.

# ## Wannierization with Wannier90
#
# We can use the `run_wannier90` routine to generate all required files and perform the wannierization procedure:

using wannier90_jll  # Needed to make run_wannier90 available
run_wannier90(scfres;
              fileprefix="wannier/graphene",
              n_wannier=5,
              projections,
              num_print_cycles=25,
              num_iter=200,
              ##
              dis_win_max=19.0,
              dis_froz_max=ustrip(auconvert(u"eV", scfres.εF))+1, # 1 eV above Fermi level
              dis_num_iter=300,
              dis_mix_ratio=1.0,
              ##
              wannier_plot=true,
              wannier_plot_format="cube",
              wannier_plot_supercell=5,
              write_xyz=true,
              translate_home_cell=true,
             );

# As can be observed standard optional arguments for the disentanglement
# can be passed directly to `run_wannier90` as keyword arguments.

# (Delete temporary files.)
rm("wannier", recursive=true)
