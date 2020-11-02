function compute_dipole_moment(ρ::RealFourierArray{T}; center=Vec3{T}(ones(3)/2)) where T
    basis = ρ.basis
    dVol  = basis.model.unit_cell_volume / prod(basis.fft_size)
    ρsum  = (vec(sum(ρ.real, dims=(2, 3))),
             vec(sum(ρ.real, dims=(1, 3))),
             vec(sum(ρ.real, dims=(1, 2))))

    dipmom_frac = Vec3(sum(ρsum[α] .* (r_values(basis, α) .- center[α]) .* dVol) for α = 1:3)
    -1 * basis.model.lattice * dipmom_frac  # -1 is for charge of electrons
end
compute_dipole_moment(scfres; kwargs...) = compute_dipole_moment(scfres.ρ; kwargs...)
