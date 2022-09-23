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
        for G in G_vectors_cart(basis, basis.kpoints[1])]
    
    # Compensating charge background => Zero DC
    poisson_green_coeffs[1] = 0
    poisson_green_coeffs_kpt[1] = 0

    TermExactExchange(T(scaling_factor), poisson_green_coeffs, poisson_green_coeffs_kpt)
end

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange, 
                                                  basis::PlaneWaveBasis{T}, ψ, occ; ρ, 
                                                  kwargs...) where {T}
    ops = [NoopOperator(basis, kpoint)
    for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(0), ops=ops)

    ψ, occ = select_occupied_orbitals(basis, ψ, occ; threshold=0.1)
    @assert length(ψ) == 1 # TODO: make it work for more kpoints

    E = T(0)
    for (k,kpoint) in enumerate(basis.kpoints)
        for psi_i in eachcol(ψ[k])
            for psi_j in eachcol(ψ[k])

                rho_ij_real = conj(G_to_r(basis, kpoint, psi_i)) .* G_to_r(basis, kpoint, psi_j)
                rho_ij_four = r_to_G(basis, rho_ij_real)

                # Re-ordering of rho_ij_four coeffs to match opposing sign G vectors
                # f(r) = Σc_G e^{iG⋅r} implies f^*(r) = Σc_G^* e^{i(-G)⋅r}
                ind1 = cat(1,basis.fft_size[1]:-1:2;dims=1)
                ind2 = cat(1,basis.fft_size[2]:-1:2;dims=1)
                ind3 = cat(1,basis.fft_size[3]:-1:2;dims=1)
                

                # equivalent to r_to_G(basis, kpoint, rho_ij_real_conj)
                rho_ij_four_conj = conj(rho_ij_four[ind1,ind2,ind3])

                # Solving poisson equation to obtain potential corresponding to 
                # ϕ_i(r')ϕ_j(r) / |r - r'|
                vij_fourier = rho_ij_four_conj .* term.poisson_green_coeffs

                E += real(dot(rho_ij_four, vij_fourier))
            end
        end
    end

    ops = [ExchangeOperator(ψ[ik], term.poisson_green_coeffs_kpt, 
    basis, kpt) for (ik,kpt) in enumerate(basis.kpoints)]
    (E=E, ops=ops)
end

function pp(x,y)
    println()
end