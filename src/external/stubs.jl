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
    DFTD3(damping=nothing; cutoffs=(;))

DFT-D3 dispersion correction of [Grimme et al. (2010)](https://doi.org/10.1063/1.3382344),
to be passed in the `extra_terms` of [`model_DFT`](@ref) (or the `terms` of a
[`Model`](@ref)). Requires `using SimpleDFTD3`.

`damping` can be any damping from
[SimpleDFTD3](https://github.com/JuliaMolSim/SimpleDFTD3.jl), e.g.
`RationalDamping("pbe"; atm=true)`. By default the D3(BJ) parameters (D3 with rational
Becke-Johnson damping) fitted for the model's exchange-correlation functional are used.
For a modified functional (scaled, or with custom exact exchange) pass a damping instead.
`cutoffs` is passed to `SimpleDFTD3.set_realspace_cutoff!` (in Bohr). With the default hard
cutoffs the energy jumps slightly when atoms cross a cutoff.

The D3 energy depends only on the atomic positions, so it does not enter the Kohn-Sham
Hamiltonian and leaves the SCF unchanged. It contributes to energies, forces and stresses.
See [Dispersion correction (DFT-D3)](@ref) for an example.

# Examples
```julia-repl
julia> model_DFT(system; functionals=PBE(), pseudopotentials, extra_terms=[DFTD3()])
```
builds a PBE-D3(BJ) model.

!!! warning "Experimental feature"
    Only fully periodic 3D systems are supported. Phonons and second derivatives of the
    energy (e.g. elastic constants) are not available. Results computed with an explicit
    `damping` cannot be reloaded with [`load_scfres`](@ref), since SimpleDFTD3 damping
    objects refer to memory of the C library.
"""
function DFTD3 end
