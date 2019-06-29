"""
Class for holding the values of a local potential,
like the local part of a pseudopotential
"""
struct PotLocal{T<:AbstractArray}
    values_real::T
end
