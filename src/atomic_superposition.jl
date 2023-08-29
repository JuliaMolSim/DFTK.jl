function group_local_quantities(basis::PlaneWaveBasis, name::Symbol)
    model = basis.model
    # Filter out atom groups whose potential doesn't have the quantity of interest
    atom_groups = [group for group in model.atom_groups
                   if hasquantity(model.atoms[first(group)], name)]
    # Collect the quantity of interest for each atom group
    quantities = [getproperty(model.atoms[first(group)].potential, name)
                  for group in atom_groups]
    # Organize the positions by atom group
    positions = [model.positions[group] for group in atom_groups]
    return (;quantities, positions)
end

function build_atomic_superposition(
    basis::PlaneWaveBasis{T},
    quantities::AbstractVector{<:AbstractQuantity},
    positions::AbstractVector{<:AbstractVector{<:Vec3}};
    weights=ones.(T, length.(positions))
) where {T}
    @assert length(quantities) == length(positions) == length(weights)
    @assert all(length.(positions) .== length.(weights))

    # Compute structure factors (the position-dependent part of the atomic quantities):
    # sf[a][j][q]
    structure_factors = compute_structure_factors(basis, positions)

    # Compute form factors (the position-dependent part of the atomic quantities):
    # ff[a][q]
    form_factors_radial = compute_form_factors_radial(basis, quantities)

    # Compute the superposition in Fourier space: F[G]
    F = sum(zip(form_factors_radial, structure_factors, weights)) do (ffr_a, sf_a, w_a)
        sum(zip(sf_a, w_a)) do (sf_aj, w_aj)
            w_aj .* sf_aj .* ffr_a
        end
    end
    F ./= sqrt(basis.model.unit_cell_volume)

    # Enforce that the inverse Fourier transform of F(G) is real, then perform the iFT
    # and return F[r]
    enforce_real!(basis, F)
    return irfft(basis, F)
end

function compute_scalar_field_forces(
    basis::PlaneWaveBasis{T},
    quantities::AbstractVector{<:AbstractQuantity},
    positions::Vector{Vector{Vec3{T}}},
    field::AbstractArray{Complex{T}}
) where {T}
    model = basis.model

    # Compute the gradients of the structure factors w.r.t. the positions
    # sf[a][j][q][α]
    structure_factor_gradients = compute_structure_factor_gradients(basis, positions)

    # Compute form factors (the position-dependent part of the atomic quantities):
    # ff[a][q]
    form_factors_radial = compute_form_factors_radial(basis, quantities)

    # Atomic species
    group_forces = map(zip(structure_factor_gradients, form_factors_radial)) do (∇sf_a, ffr_a)
        # Position
        map(∇sf_a) do ∇sf_aj
            # G-vector: -∑_{G} [ Re[F(G)'] * ff_{G} * ∇sf_{G} ] / √Ω
            f_aj = sum(zip(∇sf_aj, ffr_a, field)) do (∇sf_ajq, ffr_aq, field_q)
                real(conj(field_q) .* ffr_aq .* ∇sf_ajq)
            end
            f_aj = -f_aj ./ sqrt(model.unit_cell_volume)
        end
    end

    # Fill in the full forces vector
    forces = fill(zero(Vec3{T}), length(model.positions))
    for (atom_group, atom_group_forces) in zip(model.atom_groups, group_forces)
        forces[atom_group] .= atom_group_forces
    end

    return forces
end
