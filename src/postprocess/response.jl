function compute_hartree_kernel(basis::PlaneWaveBasis{T}) where {T}
    vc_G = 4T(π) ./ [sum(abs2, basis.model.recip_lattice * G)
                     for G in G_vectors(basis)]
    vc_G[1] = 0
    G_to_r_matrix(basis) * Diagonal(vec(vc_G)) * r_to_G_matrix(basis)
end
function apply_hartree_kernel(basis::PlaneWaveBasis{T}, dρ) where {T}
    vc_G = 4T(π) ./ [sum(abs2, basis.model.recip_lattice * G)
                     for G in G_vectors(basis)]
    vc_G[1] = 0
    real(G_to_r(basis, vc_G .* r_to_G(basis, complex(dρ))))
end

function compute_xc_kernel(basis::PlaneWaveBasis{T}, ρ) where {T}
    xc_terms = [t for t in basis.terms if (isa(t,  XcTerm) || isa(t, TermPowerNonlinearity))]
    @assert length(xc_terms) == 1
    xc = xc_terms[1]
    if isa(xc, XcTerm) && any(xc.family == Libxc.family_gga for xc in xc.functionals)
        error("Only LDA supported for response for the moment")
    end
    # In LDA, the potential depends on each component individually, so
    # we just compute each with finite differences
    pot0 = ene_ops(xc, nothing, nothing; ρ=from_real(basis, ρ)).ops[1].potential
    ε = 1e-8
    ρ_pert = from_real(basis, ρ .+ ε .* ones(T, size(ρ)))
    pot1 = ene_ops(xc, nothing, nothing; ρ=ρ_pert).ops[1].potential
    Diagonal((vec(pot1) .- vec(pot0)) ./ ε)
end

function apply_xc_kernel(basis::PlaneWaveBasis, ρ, dρ)
    reshape(compute_xc_kernel(basis, ρ) * vec(dρ), size(dρ))
end
