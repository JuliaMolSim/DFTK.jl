@doc raw"""
    determine_grid_size(lattice, Ecut; kpoints=[[0,0,0]], supersampling=2)

Determine the minimal grid size for the density fourier grid ``B_ρ`` subject to the
kinetic energy cutoff `Ecut` for the wave function and a density  `supersampling` factor.

The function will determine the union of wave vectors ``G`` required to satisfy
``|G + k|^2/2 \leq Ecut * \text{supersampling}^2`` for all ``k``-Points. The
returned grid dimensions are the smallest cartesian box to incorporate these ``G``.

For an exact representation of the density resulting from wave functions
represented in the basis ``B_ρ = \{G : |G + k|^2/2 \leq Ecut\}``, `supersampling`
should be at least `2`.
"""
function determine_grid_size(lattice::AbstractMatrix, Ecut; kpoints=[[0, 0, 0]],
                             supersampling=2)
    # Lattice and reciprocal lattice
    lattice = SMatrix{3, 3}(lattice)
    recip_lattice = 2π * inv(Matrix(lattice'))

    cutoff_qsq = 2 * supersampling^2 * Ecut
    # For a particular k-Point, the coordinates [m n o] of the
    # complementary reciprocal lattice vectors B satisfy
    #     |B * [m n o] + k|^2 ≤ cutoff_qsq
    # Now
    #     |B * [m n o] + k| ≥ abs(|B * [m n o]| - |k|) = |B * [m n o]| - |k|
    # provided that |k| ≤ |B|, which is typically the case. Therefore
    #     |[m n o]| / |B^{-1}| ≤ |B * [m n o]| ≤ sqrt(cutoff_qsq) + |k|
    # (where |B^{-1}| is the operator norm of the inverse of B), such that
    #     |[m n o]| ≤ (sqrt(cutoff_qsq) + |k|) * |B^{-1}|
    # In the extremal case, m = o = 0, such that
    #    n_max_trial = (sqrt(cutoff_qsq) + |k|) * |B^{-1}|
    #                = (sqrt(cutoff_qsq) + |k|) * |A| / 2π

    # Estimate trial upper bound n_max
    max_k = maximum(norm.(kpoints))
    @assert max_k ≤ opnorm(recip_lattice)
    trial_n_max = ceil(Int, (max_k + sqrt(cutoff_qsq)) * opnorm(lattice) / 2π)

    # Determine actual n_max (trial_n_max is extended by 1 for safety)
    trial_n_range = -trial_n_max-1:trial_n_max+1
    n_max = 0
    for coord in CartesianIndices((trial_n_range, trial_n_range, trial_n_range))
        energy(q) = sum(abs2, recip_lattice * q) / 2
        if any(energy([coord.I...] + k) ≤ supersampling^2 * Ecut for k in kpoints)
            @assert all(abs.([coord.I...]) .<= trial_n_max)
            n_max = max(n_max, maximum(abs.([coord.I...])))
        end
    end
    return 2 * n_max + 1
end
