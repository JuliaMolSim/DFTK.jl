function compute_dipole_moment(basis::PlaneWaveBasis{T}, ρ; center=Vec3{T}(ones(3)/2)) where T
    dVol  = basis.model.unit_cell_volume / prod(basis.fft_size)
    ρsum  = (vec(sum(ρ, dims=(2, 3))),
             vec(sum(ρ, dims=(1, 3))),
             vec(sum(ρ, dims=(1, 2))))

    dipmom_frac = Vec3(sum(ρsum[α] .* (r_values(basis, α) .- center[α]) .* dVol) for α = 1:3)
    return -dipmom_frac .* dVol
    -1 * basis.model.lattice * dipmom_frac  # -1 is for charge of electrons
end
compute_dipole_moment(scfres; kwargs...) = compute_dipole_moment(scfres.basis, scfres.ρ; kwargs...)
