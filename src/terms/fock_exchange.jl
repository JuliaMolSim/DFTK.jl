"""
Exact exchange term: the Hartree-Fock exchange energy of the orbitals


-1/2 ∑ ∫∫ ϕ_i^*(r)ϕ_j^*(r')ϕ_i(r')ϕ_j(r) / |r - r'| dr dr'

"""
struct FockExchange
end
(fock_exchange::FockExchange)(basis)   = TermFockExchange(basis)

struct TermFockExchange <: Term
end
function TermFockExchange(basis::PlaneWaveBasis{T}) where T
    model = basis.model

    TermFockExchange()
end

@timing "ene_ops: FockExchange" function ene_ops(term::TermFockExchange, basis::PlaneWaveBasis{T},
                                            ψ, occ; ρ, kwargs...) where {T}
    ops = [NoopOperator(basis, kpoint)
    for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(0), ops=ops)

    ψ, occ = select_occupied_orbitals(basis, ψ, occ; threshold=0.1)

    # @assert length(ψ) == 1 # TODO: make it work for more kpoints

    poisson_green_coeffs = 4T(π) ./ [sum(abs2, G) for G in G_vectors_cart(basis)]
    poisson_green_coeffs_kpt = 4T(π) ./ [sum(abs2, G) for G in G_vectors_cart(basis, basis.kpoints[1])]

    E = T(0)
    for psi_i in eachcol(ψ[1])
        for psi_j in eachcol(ψ[1])
            rho_ij_real = conj(G_to_r(basis, basis.kpoints[1], psi_i)) .* G_to_r(basis, basis.kpoints[1], psi_j)

            rho_ij_real_conj = conj(rho_ij_real)
            rho_ij_four = r_to_G(basis, rho_ij_real)
            rho_ij_four_conj = r_to_G(basis, rho_ij_real_conj)

            vij_fourier = rho_ij_four_conj .* poisson_green_coeffs
            E += real(dot(rho_ij_four, vij_fourier))
        end
    end

    (E=E, ops=ops)
end