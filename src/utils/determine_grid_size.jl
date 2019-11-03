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
function determine_grid_size(lattice::AbstractMatrix, Ecut; supersampling=2, tol=1e-8, ensure_smallprimes=true)
    # See the documentation about the grids for details on the construction of C_ρ
    cutoff_Gsq = 2 * supersampling^2 * Ecut
    Gmax= [norm(lattice[:, i]) / 2π * sqrt(cutoff_Gsq) for i in 1:3]
    # The above is a simplification valid for large Gmax. Actually we
    # just need to be able to represent the G-G' for G in B_ρ, so if
    # |G| is always < 1/2 in one direction, Gmax = 0 is fine. This is
    # important to represent 1D systems with just one point in the
    # relevant direction
    Gmax[Gmax .< 1/2] .= 0
    # Round up
    Gmax = ceil.(Int, Gmax .- tol)

    # Optimise FFT grid size: Make sure the number factorises in small primes only
    if ensure_smallprimes
        Vec3([nextprod([2, 3, 5], 2gs + 1) for gs in Gmax])
    else
        Vec3([2gs+1 for gs in Gmax])
    end
end
function determine_grid_size(model::Model, Ecut; kwargs...)
    determine_grid_size(model.lattice, Ecut; kwargs...)
end
