function compute_structure_factors!(
    structure_factor_work::AbstractVector{Complex{T}},
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    position::Vec3{T}
) where {T}
    qs_frac = Gplusk_vectors(basis, kpt)
    structure_factor_work .= -dot.(qs_frac, Ref(position))
    structure_factor_work .= cis2pi.(structure_factor_work)
    return structure_factor_work
end

function build_projection_vectors!(
    P::AbstractArray{Complex{T}},
    structure_factor_work::AbstractVector{Complex{T}},
    radial_work::AbstractArray{Complex{T}},
    angular_work::AbstractVector{Complex{T}},
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    projectors::AbstractVector{TP},
    position::Vec3{T}
) where {T,TP<:AbstractQuantity{FourierSpace}}
    qs_cart = Gplusk_vectors_cart(basis, kpt)

    # Precompute structure factors so they aren't re-computed for each l, m, n
    compute_structure_factors!(structure_factor_work, basis, kpt, position)

    for projector in projectors
        l = angular_momentum(projector)
        @views for m in (-l):(+l)
            angular_work .= (-im)^l .* ylm_real.(l, m, qs_cart)
            P[:, i_proj] .= structure_factor_work
            P[:, i_proj] .*= angular_work  # Angular part of the form factors
            P[:, i_proj] .*= radial_work[:, n, l+1]  # Radial part of the form factors
            P[:, i_proj] ./= sqrt(basis.model.unit_cell_volume)  # Normalization
            i_proj += 1
        end
    end
    return P
end

function compute_form_factor_radials!(
    radial_work::AbstractArray{Complex{T},3},
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    projectors::AbstractVector{PT}
) where {T,PT<:AbstractQuantity{FourierSpace,Numerical}}
    qs_cart = norm.(Gplusk_vectors_cart(basis, kpt))
    @views for n in eachindex(projectors)
        itp = interpolate(projectors[n]; kwargs...)
        radial_work[:,n] .= itp.(qs_cart)
    end
    return radial_work
end

function compute_form_factor_radials!(
    radial_work::AbstractArray{Complex{T},3},
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    projectors::AbstractVector{PT}
) where {T,PT<:AbstractQuantity{FourierSpace,Analytical}}
    qs_cart = norm.(Gplusk_vectors_cart(basis, kpt))
    @views for n in eachindex(projectors)
        radial_work[:,n] .= projectors[n].(qs_cart)
    end
    return radial_work
end

function build_projection_vectors!(
    P::AbstractArray{Complex{T}},
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    species_projectors,
    species_positions::Vector{Vector{Vec3{T}}}
) where {T}
    # Allocate working arrays for the angular and radial parts of the form factors
    max_l = maximum(lastindex, species_projectors)
    max_n_radials = maximum(species_projectors) do projectors
        maximum(length, projectors)
    end
    radial_work = zeros_like(P, size(P, 1), max_n_radials, max_l + 1)
    angular_work = zeros_like(P, size(P, 1))
    structure_factor_work = zeros_like(P, size(P, 1))

    i_proj = 1
    for (projectors, positions) in zip(species_projectors, species_positions)
        n_specie_projs = sum(projector -> 2 * angular_momentum(projector) + 1, projectors)
        # Precompute the radial parts of the form factors for all l, n
        compute_form_factor_radials!(radial_work, basis, kpt, projectors)
        for position in positions
            build_projection_vectors!(
                @view(P[:, i_proj:(i_proj+n_specie_projs-1)]),
                structure_factor_work,
                radial_work,
                angular_work,
                basis,
                kpt,
                projectors,
                position,
            )
            i_proj += n_specie_projs
        end
    end
    return P
end

function build_projection_vectors(
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    species_projectors,
    species_positions::Vector{Vector{Vec3{T}}}
) where {T}
    n_q = length(Gplusk_vectors(basis, kpt))
    n_proj = sum(zip(species_projectors, species_positions)) do (projectors, positions)
        n_angulars = sum(l -> length(projectors[l]) * (2l + 1), eachindex(projectors))
        n_angulars * length(positions)
    end
    P = zeros_like(Gplusk_vectors(basis, kpt), Complex{T}, n_q, n_proj)
    return build_projection_vectors!(P, basis, kpt, species_projectors, species_positions)
end
