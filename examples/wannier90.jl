# # Wannierization using Wannier90
#
# DFTK features an interface with the program
# [Wannier90](http://www.wannier.org/),
# in order to compute maximally-localized Wannier functions (MLWFs)
# from an initial self consistent field calculation.
# All processes are handled by calling the routine `run_wannier90`.
#
# !!! warning "No guarantees on Wannier90 interface"
#     This code is at an early stage and has so far not been fully tested.
#     Bugs are likely and we welcome issues in case you find any!
#
# This example shows how to obtain the MLWFs corresponding
# to the first eight bands of silicon. Since the bands 5 to 8 are entangled,
# 12 bands are first computed to obtain 8 MLWFs by a disantanglement procedure.

using DFTK

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
atoms = [ Si => [zeros(3), 0.25*[-1, 3, -1]] ]  # Non-symmetric silicon crystal
model = model_PBE(lattice, atoms)
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])

scfres = self_consistent_field(basis, tol=1e-12, n_bands=12);

# Now we use the `run_wannier90` routine to generate all files needed by
# wannier90 and to perform the wannierization procedure.
# In Wannier90's convention, all files are named with the same prefix and only differ by
# their extensions. By default all generated input and output files are stored
# in the subfolder "wannier90" under the prefix "wannier" (i.e. "wannier90/wannier.win",
# "wannier90/wannier.wout", etc.). A different file prefix can be given with the
# keyword argument `fileprefix` as shown below.
#
# We now solve for 8 MLWF using wannier90:

using wannier90_jll  # Needed to make run_wannier90 available
run_wannier90(scfres;
              fileprefix="wannier/Si",
              n_wannier=8,
              num_print_cycles=50,
              num_iter=500,
              dis_win_max=17.185257,
              dis_froz_max=6.8318033,
              dis_num_iter=120,
              dis_mix_ratio=1.0,
              wannier_plot=true);

# As can be observed standard optional arguments for the disentanglement
# can be passed directly to `run_wannier90` as keyword arguments.

# delete temporary files:
rm("wannier", recursive=true)
