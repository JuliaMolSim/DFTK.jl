# # Monitoring self-consistent field calculations
#
# The `self_consistent_field` function takes as the `callback`
# keyword argument one function to be called after each iteration.
# This function gets passed the complete internal state of the SCF
# solver and can thus be used both to monitor and debug the iterations
# as well as to quickly patch it with additional functionality.
#
# This example discusses a few aspects of the `callback` function
# taking again our favourite silicon example.

# We setup silicon in an LDA model using the ASE interface
# to build a bulk silicon lattice,
# see [Input and output formats](@ref) for more details.
using DFTK
using ASEconvert

system = pyconvert(AbstractSystem, ase.build.bulk("Si"))
model  = model_LDA(attach_psp(system; Si="hgh/pbe/si-q4.hgh"))
basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[3, 3, 3]);

# DFTK already defines a few callback functions for standard
# tasks. One example is the usual convergence table,
# which is defined in the callback [`ScfDefaultCallback`](@ref).
# Another example is [`ScfSaveCheckpoints`](@ref), which stores the state
# of an SCF at each iterations to allow resuming from a failed
# calculation at a later point.
# See [Saving SCF results on disk and SCF checkpoints](@ref) for details
# how to use checkpointing with DFTK.

# In this example we define a custom callback, which plots
# the change in density at each SCF iteration after the SCF
# has finished. This example is a bit artificial, since the norms
# of all density differences is available as `scfres.history_Δρ`
# after the SCF has finished and could be directly plotted, but
# the following nicely illustrates the use of callbacks in DFTK.

# To enable plotting we first define the empty canvas
# and an empty container for all the density differences:
using Plots
p = plot(; yaxis=:log)
density_differences = Float64[];

# The callback function itself gets passed a named tuple
# similar to the one returned by `self_consistent_field`,
# which contains the input and output density of the SCF step
# as `ρin` and `ρout`. Since the callback gets called
# both during the SCF iterations as well as after convergence
# just before `self_consistent_field` finishes we can both
# collect the data and initiate the plotting in one function.

using LinearAlgebra

function plot_callback(info)
    if info.stage == :finalize
        plot!(p, density_differences, label="|ρout - ρin|", markershape=:x)
    else
        push!(density_differences, norm(info.ρout - info.ρin))
    end
    info
end
callback = ScfDefaultCallback() ∘ plot_callback;

# Notice that for constructing the `callback` function we chained the `plot_callback`
# (which does the plotting) with the `ScfDefaultCallback`. The latter is the function
# responsible for printing the usual convergence table. Therefore if we simply did
# `callback=plot_callback` the SCF would go silent. The chaining of both callbacks
# (`plot_callback` for plotting and `ScfDefaultCallback()` for the convergence table)
# makes sure both features are enabled. We run the SCF with the chained callback …
scfres = self_consistent_field(basis; tol=1e-5, callback);

# … and show the plot
p

# The `info` object passed to the callback contains not just the densities
# but also the complete Bloch wave (in `ψ`), the `occupation`, band `eigenvalues`
# and so on.
# See [`src/scf/self_consistent_field.jl`](https://dftk.org/blob/master/src/scf/self_consistent_field.jl#L101)
# for all currently available keys.
#
# !!! tip "Debugging with callbacks"
#     Very handy for debugging SCF algorithms is to employ callbacks
#     with an `@infiltrate` from [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl)
#     to interactively monitor what is happening each SCF step.
