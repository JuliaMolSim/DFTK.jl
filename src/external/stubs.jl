# Stubs for conditionally defined functions

"""
Wannerize the obtained bands using wannier90. By default all converged
bands from the `scfres` are employed (change with `n_bands` kwargs)
and `n_wannier = n_bands` wannier functions are computed.
Random Gaussians are used as guesses by default, can be changed
using the `projections` kwarg. All keyword arguments supported by
Wannier90 for the disentanglement may be added as keyword arguments.
The function returns the `fileprefix`.

!!! warning "Experimental feature"
    Currently this is an experimental feature, which has not yet been tested
    to full depth. The interface is considered unstable and may change
    incompatibly in the future. Use at your own risk and please report bugs
    in case you encounter any.
"""
function run_wannier90 end

"""
    DFTD3(method; damping="D3(BJ)", three_body=false,
          switching_width_2body=0.0, switching_width_3body=0.0)
    DFTD3(method, dispersion_model; damping="D3(BJ)", three_body=false)
    DFTD3(damping::AbstractDamping; switching_width_2body=0.0, switching_width_3body=0.0)
    DFTD3(damping::AbstractDamping, dispersion_model)

DFT-D3 dispersion correction of [Grimme et al. (2010)](https://doi.org/10.1063/1.3382344),
to be passed in the `extra_terms` of [`model_DFT`](@ref) (or the `terms` of a
[`Model`](@ref)). Requires `using SimpleDFTD3`.

`method` is a method name of [SimpleDFTD3](https://github.com/JuliaMolSim/SimpleDFTD3.jl)
(e.g. `"pbe"`) or an [`Xc`](@ref) term (e.g. `PBE()`), for which the parameters stored
in SimpleDFTD3 are used. If you customise the XC functional (i.e. change its parameters,
scale it, change the EXX fraction, etc.) custom parameters should be used instead. `damping`
selects the dispersion damping function by its SimpleDFTD3 identifier, e.g. `"D3(BJ)"`
(rational Becke-Johnson damping) or `"D3(0)"` (zero damping), and `three_body` includes
the three-body Axilrod-Teller-Muto term.

With hard real-space cutoffs (the default) the energy jumps slightly when atoms cross a
cutoff. `switching_width_2body` and `switching_width_3body` (in Bohr) smoothly switch off
the two-body and three-body terms at their cutoffs instead.

For full control a SimpleDFTD3 `DispersionModel` can be passed, e.g. one built with
`DispersionModel(lattice, atoms, positions)` and modified using `set_realspace_cutoff!`
or `set_ewald_summation!`. It is used as is, so it must describe the structure of the model.

# Examples
```julia-repl
julia> model_DFT(system; functionals=PBE(), pseudopotentials, extra_terms=[DFTD3(PBE())])
```
Builds a PBE-D3(BJ) model.
"""
function DFTD3 end
