"""
Returns the unique norms of input vector Gs and a mapping such that
norm(Gs[i]) = unique_ps[iG2ifnorm[i]]. Runs on CPU and GPU.
"""
function unique_norms_and_mapping(Gs::AbstractArray{<:Vec3})
    # Sort the norms and remember where each original element were
    ps = vec(map(norm, Gs))  # vec() insures 1D array for sorting
    perm = sortperm(ps)
    sorted_ps = ps[perm]

    # Mark the first occurrence of each distinct value in the sorted list
    diffs = diff(sorted_ps)
    isnew = similar(sorted_ps, Bool, length(sorted_ps))
    isnew[1:1] .= true
    isnew[2:end] .= diffs .!= zero(eltype(diffs))

    # Use cumulative sum to assign a unique norm id to each distinct value
    norm_id = cumsum(isnew)
    iG2ifnorm = similar(norm_id)
    iG2ifnorm[perm] = norm_id

    # Keep only the unique norms
    unique_ps = sorted_ps[isnew]

    (; unique_ps, iG2ifnorm)
end