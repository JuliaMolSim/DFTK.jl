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
# For the moment this process is considered an experimental feature and
# has a number of caveats, see the warnings below.
#
# !!! warning "Saving `scfres` is experimental"
#     The [`load_scfres`](@ref) and [`save_scfres`](@ref) pair of functions
#     are experimental features. This means:
#
#     - The interface of these functions
#       as well as the format in which the data is stored on disk can
#       change incompatibly in the future. At this point we make no promises ...
#     - JLD2 is not yet completely matured
#       and it is recommended to only use it for short-term storage
#       and **not** to archive scientific results.
#     - If you are using the functions to transfer data between different
#       machines ensure that you use the **same version of Julia, JLD2 and DFTK**
#       for saving and loading data.
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
O = ElementPsp(:O, psp=load_psp("hgh/pbe/O-q6.hgh"))
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

# The `scfres.jld2` file could now be transfered to a different computer,
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

# ## Checkpointing of SCF calculations
# A related feature, which is very useful especially for longer calculations with DFTK
# is automatic checkpointing, where the state of the SCF is periodically written to disk.
# The advantage is that in case the calculation errors or gets aborted due
# to overrunning the walltime limit one does not need to start from scratch,
# but can continue the calculation from the last checkpoint.
#
# To enable automatic checkpointing in DFTK one needs to pass the `ScfSaveCheckpoints`
# callback to [`self_consistent_field`](@ref), for example:

callback = DFTK.ScfSaveCheckpoints()
scfres = self_consistent_field(basis; ρ=guess_density(basis, magnetic_moments),
                               tol=1e-2, callback);

# Notice that using this callback makes the SCF go silent since the passed
# callback parameter overwrites the default value (namely `DefaultScfCallback()`)
# which exactly gives the familiar printing of the SCF convergence.
# If you want to have both (printing and checkpointing) you need to chain
# both callbacks:
callback = DFTK.ScfDefaultCallback() ∘ DFTK.ScfSaveCheckpoints(keep=true)
scfres = self_consistent_field(basis; ρ=guess_density(basis, magnetic_moments),
                               tol=1e-2, callback);

# For more details on using callbacks with DFTK's `self_consistent_field` function
# see [Monitoring self-consistent field calculations](@ref).

# By default checkpoint is saved in the file `dftk_scf_checkpoint.jld2`, which is
# deleted automatically once the SCF completes successfully. If one wants to keep
# the file one needs to specify `keep=true` as has been done in the ultimate SCF
# for demonstration purposes: now we can continue the previous calculation
# from the last checkpoint as if the SCF had been aborted.
# For this one just loads the checkpoint with [`load_scfres`](@ref):

oldstate = load_scfres("dftk_scf_checkpoint.jld2")
scfres   = self_consistent_field(oldstate.basis, ρ=oldstate.ρ,
                                 ψ=oldstate.ψ, tol=1e-3);

# !!! note "Availability of `load_scfres`, `save_scfres` and `ScfSaveCheckpoints`"
#     As JLD2 is an optional dependency of DFTK these three functions are only
#     available once one has *both* imported DFTK and JLD2 (`using DFTK`
#     and `using JLD2`).

# (Cleanup files generated by this notebook)
rm("dftk_scf_checkpoint.jld2")
rm("scfres.jld2")
