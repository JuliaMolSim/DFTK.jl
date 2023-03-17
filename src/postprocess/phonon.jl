@timing function compute_dynmat(scfres::NamedTuple; kwargs...)
    dynmats_per_term = [compute_dynmat(term, scfres; kwargs...)
                       for term in scfres.basis.terms]
    sum(filter(!isnothing, dynmats_per_term))
end

function compute_dynmat_cart(scfres::NamedTuple; kwargs...)
    dynmats_reduced = compute_dynmat(scfres; kwargs...)
    dynmat_to_cart(scfres.basis, dynmats_reduced)
end

@timing function compute_δHψ(scfres::NamedTuple; kwargs...)
    δHψ_per_term = [compute_δHψ(term, scfres; kwargs...)
                    for term in scfres.basis.terms]
    sum(filter(!isnothing, δHψ_per_term))
end
