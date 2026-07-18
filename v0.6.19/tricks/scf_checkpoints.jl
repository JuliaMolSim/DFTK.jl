# # Saving SCF results on disk and SCF checkpoints
#
# For longer DFT calculations it is pretty standard to run them on a cluster
# in advance and to perform postprocessing (band structure calculation,
# plotting of density, etc.) at a later point and potentially on a different
# machine.
#
# To support such workflows DFTK offers the two functions [`save_scfres`](@ref)
# and [`load_scfres`](@ref), which allow to save the data structure returned
# by [`self_consistent_field`](@ref) on disk or retrieve it back into memory,
# respectively. For this purpose DFTK uses the
# [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) file format and Julia package.
#
# !!! note "Availability of `load_scfres`, `save_scfres` and checkpointing"
#     As JLD2 is an optional dependency of DFTK these three functions are only
#     available once one has *both* imported DFTK and JLD2 (`using DFTK`
#     and `using JLD2`).
#
# !!! warning "DFTK data formats are not yet fully matured"
#     The data format in which DFTK saves data as well as the general interface
#     of the [`load_scfres`](@ref) and [`save_scfres`](@ref) pair of functions
#     are not yet fully matured. If you use the functions or the produced files
#     expect that you need to adapt your routines in the future even with patch
#     version bumps.
#
# To illustrate the use of the functions in practice we will compute
# the total energy of the O₂ molecule at PBE level. To get the triplet
# ground state we use a collinear spin polarisation
# (see [Collinear spin and magnetic systems](@ref) for details)
# and a bit of temperature to ease convergence:

using DFTK
using LinearAlgebra
using JLD2

d = 2.079  # oxygen-oxygen bondlength
a = 9.0    # size of the simulation box
lattice = a * I(3)
O = ElementPsp(:O; psp=load_psp("hgh/pbe/O-q6.hgh"))
atoms     = [O, O]
positions = d / 2a * [[0, 0, 1], [0, 0, -1]]
magnetic_moments = [1., 1.]

Ecut  = 10  # Far too small to be converged
model = model_PBE(lattice, atoms, positions; temperature=0.02, smearing=Smearing.Gaussian(),
                  magnetic_moments)
basis = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])

scfres = self_consistent_field(basis, tol=1e-2, ρ=guess_density(basis, magnetic_moments))
save_scfres("scfres.jld2", scfres);
#-
scfres.energies

# The `scfres.jld2` file could now be transferred to a different computer,
# Where one could fire up a REPL to inspect the results of the above
# calculation:

using DFTK
using JLD2
loaded = load_scfres("scfres.jld2")
propertynames(loaded)
#-
loaded.energies

# Since the loaded data contains exactly the same data as the `scfres` returned by the
# SCF calculation one could use it to plot a band structure, e.g.
# `plot_bandstructure(load_scfres("scfres.jld2"))` directly from the stored data.
#
# Notice that both `load_scfres` and `save_scfres` work by transferring all data
# to/from the master process, which performs the IO operations without parallelisation.
# Since this can become slow, both functions support optional arguments to speed up
# the processing. An overview:
# - `save_scfres("scfres.jld2", scfres; save_ψ=false)` avoids saving
#   the Bloch wave, which is usually faster and saves storage space.
# - `load_scfres("scfres.jld2", basis)` avoids reconstructing the basis from the file,
#   but uses the passed basis instead. This save the time of constructing the basis
#   twice and allows to specify parallelisation options (via the passed basis). Usually
#   this is useful for continuing a calculation on a supercomputer or cluster.
#
# See also the discussion on [Input and output formats](@ref) on JLD2 files.

# ## Checkpointing of SCF calculations
# A related feature, which is very useful especially for longer calculations with DFTK
# is automatic checkpointing, where the state of the SCF is periodically written to disk.
# The advantage is that in case the calculation errors or gets aborted due
# to overrunning the walltime limit one does not need to start from scratch,
# but can continue the calculation from the last checkpoint.
#
# The easiest way to enable checkpointing is to use the [`kwargs_scf_checkpoints`](@ref)
# function, which does two things. (1) It sets up checkpointing using the
# [`ScfSaveCheckpoints`](@ref) callback and (2) if a checkpoint file is detected,
# the stored density is used to continue the calculation instead of the usual
# atomic-orbital based guess. In practice this is done by modifying the keyword arguments
# passed to # [`self_consistent_field`](@ref) appropriately, e.g. by using the density
# or orbitals from the checkpoint file. For example:

checkpointargs = kwargs_scf_checkpoints(basis; ρ=guess_density(basis, magnetic_moments))
scfres = self_consistent_field(basis; tol=1e-2, checkpointargs...);

# Notice that the `ρ` argument is now passed to kwargs_scf_checkpoints instead.
# If we run in the same folder the SCF again (here using a tighter tolerance),
# the calculation just continues.

checkpointargs = kwargs_scf_checkpoints(basis; ρ=guess_density(basis, magnetic_moments))
scfres = self_consistent_field(basis; tol=1e-3, checkpointargs...);

# Since only the density is stored in a checkpoint
# (and not the Bloch waves), the first step needs a slightly elevated number
# of diagonalizations. Notice, that reconstructing the `checkpointargs` in this second
# call is important as the `checkpointargs` now contain different data,
# such that the SCF continues from the checkpoint.
# By default checkpoint is saved in the file `dftk_scf_checkpoint.jld2`, which can be changed
# using the `filename` keyword argument of [`kwargs_scf_checkpoints`](@ref). Note that the
# file is not deleted by DFTK, so it is your responsibility to clean it up. Further note
# that warnings or errors will arise if you try to use a checkpoint, which is incompatible
# with your calculation.
#
# We can also inspect the checkpoint file manually using the `load_scfres` function
# and use it manually to continue the calculation:

oldstate = load_scfres("dftk_scf_checkpoint.jld2")
scfres   = self_consistent_field(oldstate.basis, ρ=oldstate.ρ, ψ=oldstate.ψ, tol=1e-4);

# Some details on what happens under the hood in this mechanism: When using the
# `kwargs_scf_checkpoints` function, the `ScfSaveCheckpoints` callback is employed
# during the SCF, which causes the density to be stored to the JLD2 file in every iteration.
# When reading the file, the `kwargs_scf_checkpoints` transparently patches away the `ψ`
# and `ρ` keyword arguments and replaces them by the data obtained from the file.
# For more details on using callbacks with DFTK's `self_consistent_field` function
# see [Monitoring self-consistent field calculations](@ref).

# (Cleanup files generated by this notebook)
rm("dftk_scf_checkpoint.jld2")
rm("scfres.jld2")
