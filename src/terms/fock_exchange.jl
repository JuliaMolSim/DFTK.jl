"""
Exact exchange term: the Hartree-Fock exchange energy of the orbitals


-1/2 ∑ ∫∫ ϕ_i^*(r)ϕ_j^*(r')ϕ_i(r')ϕ_j(r) / |r - r'| dr dr'

"""
struct FockExchange
    scaling_factor::Real  # to scale by an arbitrary factor (useful for hybrid models)
end
FockExchange(; scaling_factor=1) = FockExchange(scaling_factor)
(exchange::FockExchange)(basis)   = TermFockExchange(basis, exchange.scaling_factor)
function Base.show(io::IO, exchange::FockExchange)
    fac = isone(exchange.scaling_factor) ? "" : ", scaling_factor=$scaling_factor"
    print(io, "FockExchange($fac)")
end
struct TermFockExchange <: Term
    scaling_factor::Real  # scaling factor, absorbed into poisson_green_coeffs
end
function TermFockExchange(basis::PlaneWaveBasis{T}, scaling_factor) where T
    model = basis.model

    TermFockExchange(T(scaling_factor))
end

@timing "ene_ops: FockExchange" function ene_ops(term::TermFockExchange, basis::PlaneWaveBasis{T},
                                            ψ, occ; ρ, kwargs...) where {T}
    ops = [NoopOperator(basis, kpoint)
    for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(0), ops=ops)

    ψ, occ = select_occupied_orbitals(basis, ψ, occ; threshold=0.1)

    # @assert length(ψ) == 1 # TODO: make it work for more kpoints

    poisson_green_coeffs = term.scaling_factor * 4T(π) ./ [sum(abs2, G) for G in G_vectors_cart(basis)]
    poisson_green_coeffs_kpt = term.scaling_factor * 4T(π) ./ [sum(abs2, G) for G in G_vectors_cart(basis, basis.kpoints[1])]

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

    ops = [ExchangeOperator(ψ[ik], poisson_green_coeffs, poisson_green_coeffs_kpt, basis, kpt) for (ik,kpt) in enumerate(basis.kpoints)]
    (E=E, ops=ops)
end