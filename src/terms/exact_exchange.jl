"""
Exact exchange term: the Hartree-Exact exchange energy of the orbitals

-1/2 ∑ ∫∫ ϕ_i^*(r)ϕ_j^*(r')ϕ_i(r')ϕ_j(r) / |r - r'| dr dr'

"""
struct ExactExchange
    scaling_factor::Real  # to scale the term (e.g. for hybrid models)
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
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor) where T
    scale = T(scaling_factor)

    # q = T[0.5, 0.5, 0.5]
    # Gqs = map(G_vectors(basis)) do G
    #     recip_vector_red_to_cart(basis.model, G + q)
    # end
    # poisson_green_coeffs     = 4T(π) * scale ./ norm2.(Gqs)

    poisson_green_coeffs    = 4T(π) * scale ./ norm2.(G_vectors_cart(basis))
    poisson_green_coeffs[1] = 0  # Compensating charge background => Zero DC

    @assert iszero(G_vectors(basis, basis.kpoints[1])[1])

    TermExactExchange(scale, poisson_green_coeffs)
end

# Note: Implementing exact exchange in a scalable and numerically stable way, such that it
# rapidly converges with k-points is tricky. This implementation here is far too simple and
# slow to be useful.
#
# For further information (in particular on regularising the Coulomb), consider the following
#      https://www.vasp.at/wiki/index.php/Coulomb_singularity
#      https://journals.aps.org/prb/pdf/10.1103/PhysRevB.34.4405   (QE default)
#      https://journals.aps.org/prb/pdf/10.1103/PhysRevB.73.205119
#      https://docs.abinit.org/topics/documents/hybrids-2017.pdf (Abinit apparently
#           uses a short-ranged Coulomb)

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange,
                                                  basis::PlaneWaveBasis{T}, ψ, occupation;
                                                  kwargs...) where {T}
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(0), ops=NoopOperator.(basis, basis.kpoints))
    end

    @assert iszero(basis.model.temperature)  # ground state
    ψ, occupation = select_occupied_orbitals(basis, ψ, occupation; threshold=0.1)
    E = zero(T)

    @assert length(ψ) == 1  # TODO: make it work for more kpoints
    ik   = 1
    kpt  = basis.kpoints[ik]
    occk = occupation[ik]
    ψk   = ψ[ik]

    for (n, ψn) in enumerate(eachcol(ψk))
        for (m, ψm) in enumerate(eachcol(ψk))
            ψn_real = ifft(basis, kpt, ψn)
            ψm_real = ifft(basis, kpt, ψm)

            ρnm_real = conj(ψn_real) .* ψm_real
            ρmn_real = conj(ψm_real) .* ψn_real

            Vx_nm_four = fft(basis, ρnm_real) .* term.poisson_green_coeffs
            Vx_nm_real = ifft(basis, Vx_nm_four)

            occ_mn = occk[n] * occk[m]
            E -= real(dot(Vx_nm_real, ρmn_real)) * basis.dvol * occ_mn / 2
        end
    end

    ops = [ExchangeOperator(basis, kpt, term.poisson_green_coeffs, occk, ψk)]
    (E=E, ops=ops)
end
