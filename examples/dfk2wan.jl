# # DFTK - Wannier90 interface
#
# DFTK features an interface with the program
# [Wannier90](http://www.wannier.org/),
# in order to compute maximally-localised Wannier functions (MLWFs)
# from an initial scf calculation. Wannier90 needs one input file
# `.win` that contains the parameters of the calculations and the
# descrption of the system, as well as three data files `.mmn`,
# `.amn` and `.eig`. These files are respectively generated by
# the functions [`dtk2wan_win_file`](@ref) and
# [`dftk2wan_wannierization_files`](@ref). The whole procedure
# takes five steps :
#
#     - SCF calculation
#     - call to dftk2wan_win_file
#     - Wannier90 preprocessing step
#     - call to dftk2wan_wannierization_files
#     - Wannier90's computation of MLWFs
#
#
#
# ## Set of isolated bands
#
# This first example shows how to obtain the MLWFs corresponding
# to the four valence bands of silicon. We first perfom a
# SCF calculation, which is for the most part identical to the
# regular case. 
#
# !!! warning "Compatibility asks for a specific setup"
#     The `.win` file will
#     correspond to the studied system only if the number of ``k``
#     points and bands in input is the same as in output.
#     Since DFTK reduces by default
#     the number of ``k`` points by symmetry, and adds automaticaly
#     three non-converged bands, one must specify :
#     - `use_symmetry = false` in the creation of the plane wave basis
#     - `n_ep_extra = 0` in the self_consistent_field declaration.
#

using DFTK

a = 10.26
lattice = a / 2*[[-1.  0. -1.];
                 [ 0   1.  1.];
                 [ 1   1.  0.]]

Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]] ]

model = model_PBE(lattice,atoms)

kgrid = [4,4,4]
Ecut = 20
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, use_symmetry = false)

scfres = self_consistent_field(basis, tol=1e-12, n_bands = 4, n_ep_extra = 0 );

# We may now create the `.win` file. In addition of information on the system
# (via basis, scfres and kgrid) the functon asks for a name (here "Si") and
# the number of wanted MLWFs. For isolated bands, this number is the number
# of bands, i.e. four in our case.

dftk2wan_win_file("Si",basis,scfres,4, kgrid = kgrid)

# Then, we perfom a preproprocessing task via Wannier90 to generate
# the `Si.nnkp` file. This is simply done by :

PathToWannier90/wannier90.x -pp Si

#




# ## Disentanglement 









