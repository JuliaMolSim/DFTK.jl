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
# to build the silicon lattice,
# see [Creating slabs with ASE](@ref) for more details.
using DFTK
using PyCall

silicon = pyimport("ase.build").bulk("Si")
atoms = load_atoms(silicon)
atoms = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="lda")) => position
         for (el, position) in atoms]
lattice = load_lattice(silicon);

model = model_LDA(lattice, atoms)
kgrid = [3, 3, 3]  # k-point grid
Ecut = 5           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid);

# DFTK already defines a few callback functions for standard
# tasks. One example is the usual convergence table,
# which is defined in the callback `ScfDefaultCallback`.
# Another example is `ScfPlotTrace`, which records the total
# energy at each iteration and uses it to plot the convergence
# of the SCF graphically once it is converged.
# For details and other callbacks
# see [`src/scf/scf_callbacks.jl`](https://dftk.org/blob/master/src/scf/scf_callbacks.jl).
#
# !!! note "Callbacks are not exported"
#     Callbacks are not exported from the DFTK namespace as of now,
#     so you will need to use them, e.g., as `DFTK.ScfDefaultCallback`
#     and `DFTK.ScfPlotTrace`.


# In this example we define a custom callback, which plots
# the change in density at each SCF iteration after the SCF
# has finished. For this we first define the empty plot canvas
# and an empty container for all the density differences:
using Plots
p = plot(yaxis=:log)
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
callback = DFTK.ScfDefaultCallback() ∘ plot_callback;

# Notice that for constructing the `callback` function we chained the `plot_callback`
# (which does the plotting) with the `ScfDefaultCallback`, such that when using
# the `plot_callback` function with `self_consistent_field` we still get the usual
# convergence table printed. We run the SCF with this callback ...
scfres = self_consistent_field(basis, tol=1e-8, callback=callback);

# ... and show the plot
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
