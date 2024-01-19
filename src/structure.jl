"""
Compute the inverse of the lattice. Takes special care of 1D or 2D cases.
"""
function compute_inverse_lattice(lattice::AbstractMatrix{T}) where {T}
    # Note: pinv pretty much does the same, but the implied SVD causes trouble
    #       with interval arithmetic and dual numbers, so we go for this version.
    n_dim = count(!iszero, eachcol(lattice))
    @assert 1 ≤ n_dim ≤ 3
    if n_dim == 3
        inv(lattice)
    else
        Mat3{T}([
            inv(lattice[1:n_dim, 1:n_dim])   zeros(T, n_dim, 3 - n_dim);
            zeros(T, 3 - n_dim, 3)
        ])
    end
end

"""
Compute the reciprocal lattice.
We use the convention that the reciprocal lattice is the set of G vectors such
that G ⋅ R ∈ 2π ℤ for all R in the lattice.
"""
function compute_recip_lattice(lattice::AbstractMatrix{T}) where {T}
    2T(π) * compute_inverse_lattice(lattice')
end

"""
Compute unit cell volume volume. In case of 1D or 2D case, the volume is the length/surface.
"""
function compute_unit_cell_volume(lattice)
    n_dim = count(!iszero, eachcol(lattice))
    abs(det(lattice[1:n_dim, 1:n_dim]))
end

"""Compute the diameter of the unit cell"""
function diameter(lattice::AbstractMatrix)
    # brute force search
    diam = zero(eltype(lattice))
    for vec in Vec3.(Iterators.product(-1:1, -1:1, -1:1))
        diam = max(diam, norm(lattice * vec))
    end
    diam
end

"""
Estimate integer bounds for dense space loops from a given inequality ||Mx|| ≤ δ.
For 1D and 2D systems the limit will be zero in the auxiliary dimensions.
"""
function estimate_integer_lattice_bounds(M::AbstractMatrix{T}, δ, shift=zeros(3);
                                         tol=sqrt(eps(T))) where {T}
    # As a general statement, with M a lattice matrix, then if ||Mx|| <= δ,
    # then xi = <ei, M^-1 Mx> = <M^-T ei, Mx> <= ||M^-T ei|| δ.
    inv_lattice_t = compute_inverse_lattice(M')
    xlims = [norm(inv_lattice_t[:, i]) * δ + shift[i] for i = 1:3]

    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one x vector for 1D or 2D systems)
    xlims = [xlim == 0 ? 0 : ceil(Int, xlim .- tol) for xlim in xlims]
    xlims
end
