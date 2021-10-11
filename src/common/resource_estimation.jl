"""
Returns a very (very) rough estimate of the time per SCF step (in seconds),
assuming FFTs are the limiting operation.
"""
function estimate_time_of_scf_step(basis::PlaneWaveBasis)
    # TODO pretty print this instead of just seconds.
    # There's nothing in julia stdlib for this apparently...
    # Super rough figure from various tests on cluster, laptops etc.
    # on a 128^3 FFT grid.
    time_per_FFT_per_grid_point = 30 #= ms =# / 1000 / 128^3

    (time_per_FFT_per_grid_point
     * prod(basis.fft_size)
     * length(basis.kpoints)
     * div(basis.model.n_electrons, filled_occupation(basis.model), RoundUp)
     * 8  # mean number of FFT steps per state per k-point per iteration
     )
end
