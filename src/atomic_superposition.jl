function prepare_local_quantities(basis::PlaneWaveBasis, name::Symbol)
    model = basis.model
    # Filter out atom groups whose potential doesn't have the quantity of interest
    atom_groups = [group for group in model.atom_groups
                   if hasquantity(model.atoms[first(group)], name)]
    # Create callable objects which evaluate the quantity in Fourier space for each atom
    # group
    evaluators = map(atom_groups) do group
        el = model.atoms[first(group)]
        qty_real = getproperty(el.potential, name)
        qty_fourier = rft(qty_real, basis.atom_qgrid;
                          quadrature_method=basis.atom_rft_quadrature_method)
        evaluate(qty_fourier, basis.atom_q_interpolation_method)
    end

    # Organize the positions by atom group
    positions = [model.positions[group] for group in atom_groups]
    return (evaluators, positions)
end

function build_atomic_superposition(
    basis::PlaneWaveBasis{T},
    quantities,
    positions::Vector{Vector{Vec3{T}}};
    weights=ones.(T, length.(positions))
) where {T}
    @assert length(quantities) == length(positions) == length(weights)
    @assert all(length.(positions) .== length.(weights))

    # Compute structure factors (the position-dependent part of the atomic quantities):
    # sf[a][j][q]
    structure_factors = compute_structure_factors(basis, positions)

    # Compute form factors (the position-dependent part of the atomic quantities):
    # ff[a][q]
    qs_cart = norm.(G_vectors_cart(basis))
    form_factors_radials = map(qty -> qty.(qs_cart), quantities)

    # Compute the superposition in Fourier space: F[G]
    F = sum(zip(form_factors_radials, structure_factors, weights)) do (ffr_a, sf_a, w_a)
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
    quantities,
    positions::Vector{Vector{Vec3{T}}},
    field::AbstractArray{Complex{T}}
) where {T}
    model = basis.model

    # Compute the gradients of the structure factors w.r.t. the positions
    # sf[a][j][q][α]
    ∇sf = compute_structure_factor_gradients(basis, positions)

    # Compute form factors (the position-dependent part of the atomic quantities):
    # ff[a][q]
    qs_cart = norm.(G_vectors_cart(basis))
    ffr = map(qty -> qty.(qs_cart), quantities)

    # Atomic species
    group_forces = map(zip(∇sf, ffr)) do (∇sf_a, ffr_a)
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
