# Stubs for conditionally defined functions

"""
Wannerize the obtained bands using wannier90. By default all converged
bands from the `scfres` are employed (change with `n_bands` kwargs)
and `n_wannier = n_bands` wannier functions are computed using
random Gaussians as guesses. All keyword arguments supported by
Wannier90 for the disentanglement may be added as keyword arguments.
The function returns the `fileprefix`.

!!! warning "Experimental feature"
    Currently this is an experimental feature, which has not yet been tested
    to full depth. The interface is considered unstable and may change
    incompatibly in the future. Use at your own risk and please report bugs
    in case you encounter any.
"""
function run_wannier90 end
