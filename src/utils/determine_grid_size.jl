@doc raw"""
    determine_grid_size(lattice, Ecut; supersampling=2)

Determine the minimal grid size for the fourier grid ``C_ρ`` subject to the
kinetic energy cutoff `Ecut` for the wave function and a density  `supersampling` factor.
Optimise the grid afterwards for the FFT procedure by ensuring factorisation into
small primes.
The function will determine the smallest cube ``C_ρ`` containing the basis ``B_ρ``,
i.e. the wave vectors ``|G|^2/2 \leq E_\text{cut} ⋅ \text{supersampling}^2``.
For an exact representation of the density resulting from wave functions
represented in the basis ``B_k = \{G : |G + k|^2/2 \leq E_\text{cut}\}``,
`supersampling` should be at least `2`.
"""
function determine_grid_size(lattice::AbstractMatrix, Ecut; supersampling=2, tol=1e-8)
    # See the documentation about the grids for details on the construction of C_ρ
    cutoff_Gsq = 2 * supersampling^2 * Ecut
    fft_size = [norm(lattice[:, i]) / 2π * sqrt(cutoff_Gsq) for i in 1:3]

    # Convert fft_size into integers by rounding up unless the value is only
    # `tol` larger than an actual integer.
    fft_size = ceil.(Int, fft_size .- tol)

    # Optimise FFT grid size: Make sure the number factorises in small primes only
    return Vec3([nextprod([2, 3, 5], 2 * gs + 1) for gs in fft_size])
end
