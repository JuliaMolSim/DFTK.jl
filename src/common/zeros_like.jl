# Create an array of same type as X filled with zeros, minimizing the number
# of allocations.
function zeros_like(X::AT, n, m) where AT <: AbstractArray
    Z = similar(X, n, m)
    Z .= 0
    Z
end
