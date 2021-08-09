function nuclear_dipole_moment(model)  # In cartesian coordinates per unit volume
    Ztot = 0.0
    dipmom = zeros(3)
    for (el, pos) in model.atoms
        for p in pos
            Ztot += charge_ionic(el)
            dipmom .+= charge_ionic(el) * p
        end
    end
    model.lattice * dipmom / model.unit_cell_volume
end

function center_of_charge(atoms)  # in fractional coordinates
    Ztot = 0.0
    dipmom = zeros(3)
    for (el, pos) in atoms
        for p in pos
            Ztot += charge_ionic(el)
            dipmom .+= charge_ionic(el) * p
        end
    end
    dipmom / Ztot
end

# cartesian dipole moment per unit volume:
function compute_dipole_moment(basis::PlaneWaveBasis{T}, ρ; center=Vec3{T}(ones(3)/2)) where T
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
    dVol /= basis.model.unit_cell_volume

    @assert size(ρ, 4) == 1
    ρ = dropdims(ρ, dims=4)

    wrapcoord(x) = mod(x + 1.5, 1.0) - 0.5

    ρsum  = (vec(sum(ρ, dims=(2, 3))),
             vec(sum(ρ, dims=(1, 3))),
             vec(sum(ρ, dims=(1, 2))))

    # For electronic dipole moment center needs to be center of change (convention of abinit)

    # TODO Need to wrap r_values(basis, α) .- center[α]) to ensure this is in [-0.5, 0.5)
    dipmom_frac = Vec3(sum(ρsum[α] .* wrapcoord.(r_values(basis, α) .- center[α]) .* dVol) for α = 1:3)
    -1 * basis.model.lattice * dipmom_frac  # -1 is for charge of electrons
end
compute_dipole_moment(scfres; kwargs...) = compute_dipole_moment(scfres.basis, scfres.ρ; kwargs...)


#
#
#   \int_\Omega x ρ(r)   D r
# = \sum_G ρ(G) / sqrt(\Omega)
#   \int_\Omega x e^{i (Gx.x + Gy.y + Gz.z)} Dx Dy Dz
# = \sum_G ρ(G) / sqrt(\Omega)
#   \int_{-a/2}^{a/2} x e^{i Gx.x} Dx
# = \sum_G ρ(G) / sqrt(\Omega)
#   \int_{-a/2}^{a/2} x e^{i Gx.x} Dx
# = \sum_{Gf} ρ(AGf) / sqrt(\Omega)
#   A \int_{-1/2}^{1/2} xf e^{2πi Gfx.xf} Dxf
# = \sum_{Gf} ρ(AGf) / sqrt(\Omega)
#   A \int_{-1/2}^{1/2} xf i/2 sin(2π Gfx.xf) Dxf
#
#
#
# Modelled after what GPAW does for the dipole moment ... not properly normalised
# missing the right factors ...
function compute_dipole_moment_fft(basis, ρ)
    fft_size = basis.fft_size    # ????
    ρF = imag.(r_to_G(basis, ρ)) #* basis.model.unit_cell_volume
    stuff = [
        sum(ρF[2:end, 1, 1] ./ (2:fft_size[1])),
        sum(ρF[1, 2:end, 1] ./ (2:fft_size[2])),
        sum(ρF[1, 1, 2:end] ./ (2:fft_size[3])),
    ]

    -1 * basis.model.lattice * stuff .* (basis.dvol / π)
end
