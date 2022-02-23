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
Returns the sum formula of the atoms list as a string.
"""
function chemical_formula(atoms)
    element_count = Dict{Symbol, Int}()
    for (element, positions) in atoms
        symbol = atomic_symbol(element)
        if symbol in keys(element_count)
            element_count[symbol] += length(positions)
        else
            element_count[symbol]  = length(positions)
        end
    end
    formula = join(string(elem) * string(element_count[elem])
                      for elem in sort(collect(keys(element_count))))
    for i in 0:9
        formula = replace(formula, ('0' + i) => ('₀' + i))
    end
    formula
end
chemical_formula(model::Model) = chemical_formula(model.atoms)
