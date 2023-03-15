"""
Exact exchange term: the Hartree-Exact exchange energy of the orbitals


-1/2 ∑ ∫∫ ϕ_i^*(r)ϕ_j^*(r')ϕ_i(r')ϕ_j(r) / |r - r'| dr dr'

"""
struct ExactExchange
    scaling_factor::Real  # to scale by an arbitrary factor (useful for hybrid models)
end
ExactExchange(; scaling_factor=1) = ExactExchange(scaling_factor)
(exchange::ExactExchange)(basis) = TermExactExchange(basis, exchange.scaling_factor)
function Base.show(io::IO, exchange::ExactExchange)
    fac = isone(exchange.scaling_factor) ? "" : "scaling_factor=$(exchange.scaling_factor)"
    print(io, "ExactExchange($fac)")
end
struct TermExactExchange <: Term
    scaling_factor::Real  # scaling factor, absorbed into poisson_green_coeffs
    poisson_green_coeffs::AbstractArray
    poisson_green_coeffs_kpt::AbstractArray
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor) where T
    model = basis.model

    poisson_green_coeffs = T(scaling_factor) .* 4T(π) ./ [sum(abs2, G) 
        for G in G_vectors_cart(basis)]
    poisson_green_coeffs_kpt = T(scaling_factor) .* 4T(π) ./ [sum(abs2, G) 
        for G in Gplusk_vectors_cart(basis, basis.kpoints[1])]
    
    # Compensating charge background => Zero DC
    poisson_green_coeffs[1] = 0
    poisson_green_coeffs_kpt[1] = 0

    TermExactExchange(T(scaling_factor), poisson_green_coeffs, poisson_green_coeffs_kpt)
end

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange, 
                                                  basis::PlaneWaveBasis{T}, ψ, occ; ρ, 
                                                  kwargs...) where {T}
    ops = [NoopOperator(basis, kpoint) for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(0), ops=ops)

    ψ, occ = select_occupied_orbitals(basis, ψ, occ; threshold=0.1)
    @assert length(ψ) == 1  # TODO: make it work for more kpoints
    @assert basis.model.temperature == 0  # ground state
    E = T(0)
    for (k,kpoint) in enumerate(basis.kpoints)
        for (i,psi_i) in enumerate(eachcol(ψ[k]))
            for (j,psi_j) in enumerate(eachcol(ψ[k]))

                psi_i_real = G_to_r(basis, kpoint, psi_i)
                psi_j_real = G_to_r(basis, kpoint, psi_j)
                
                # ρ_ij(r) = ψ_i(r)^* ψ_j(r)
                rho_ij_real = conj(psi_j_real) .* psi_i_real
                rho_ij_real_conj = conj(psi_i_real) .* psi_j_real

                # Poisson solve - ∫ρ_ij(r') / |r-r'|dr'
                rho_ij_four = r_to_G(basis, kpoint, rho_ij_real)
                v_ij_four = rho_ij_four .* term.poisson_green_coeffs_kpt

                v_ij_real = G_to_r(basis, kpoint, v_ij_four)
                E += real(dot(v_ij_real, rho_ij_real_conj))

            end
        end
    end
    E_scaled = -.5 * E
    ops = [ExchangeOperator(ψ[ik], term.poisson_green_coeffs_kpt, basis, kpt) 
        for (ik,kpt) in enumerate(basis.kpoints)]
    (E=E_scaled, ops=ops)
end