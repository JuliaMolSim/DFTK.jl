@doc raw"""
Exact exchange term: the Hartree-Exact exchange energy of the orbitals
```math
-1/2 ∑_{nm} f_n f_m ∫∫ \frac{ψ_n^*(r)ψ_m^*(r')ψ_n(r')ψ_m(r)}{|r - r'|} dr dr'
```
"""
struct ExactExchange
    scaling_factor::Real  # to scale the term
    coulomb_model::CoulombModel
end
ExactExchange(; scaling_factor=1, coulomb_model=ProbeCharge()) = ExactExchange(scaling_factor, coulomb_model)
(exchange::ExactExchange)(basis) = TermExactExchange(basis, exchange.scaling_factor, exchange.coulomb_model)
function Base.show(io::IO, exchange::ExactExchange)
    fac = isone(exchange.scaling_factor) ? "" : "scaling_factor=$(exchange.scaling_factor), "
    print(io, "ExactExchange($faccoulomb_model=$(exchange.coulomb_model))")
end
struct TermExactExchange <: Term
    scaling_factor::Real  # scaling factor, absorbed into poisson_green_coeffs
    poisson_green_coeffs::AbstractArray
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor, coulomb_model::CoulombModel) where T
    poisson_green_coeffs = compute_coulomb_kernel(basis; scaling_factor=scaling_factor, coulomb_model=coulomb_model) # TODO: we need this for every q-point
    TermExactExchange(T(scaling_factor), poisson_green_coeffs)
end

# Note: Implementing exact exchange in a scalable and numerically stable way, such that it
# rapidly converges with k-points is tricky. This implementation here is far too simple and
# slow to be useful.
#
# ABINIT/src/66_wfs/m_getghc.F90
# ABINIT/src/66_wfs/m_fock_getghc.F90
# ABINIT/src/66_wfs/m_prep_kgb.F90
# ABINIT/src/66_wfs/m_bandfft_kpt.F90
#
# For further information (in particular on regularising the Coulomb), consider the following
#      https://www.vasp.at/wiki/index.php/Coulomb_singularity
#      https://journals.aps.org/prb/pdf/10.1103/PhysRevB.34.4405   (QE default)
#      https://journals.aps.org/prb/pdf/10.1103/PhysRevB.73.205119
#      https://journals.aps.org/prb/pdf/10.1103/PhysRevB.77.193110
#      https://docs.abinit.org/topics/documents/hybrids-2017.pdf (Abinit apparently
#           uses a short-ranged Coulomb)

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange,
                                                  basis::PlaneWaveBasis{T}, ψ, occupation;
                                                  kwargs...) where {T}
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(0), ops=NoopOperator.(basis, basis.kpoints))
    end

    # TODO Occupation threshold
    ψ, occupation = select_occupied_orbitals(basis, ψ, occupation; threshold=1e-8)

    E = zero(T)

    @assert length(ψ) == 1  # TODO: make it work for more kpoints
    ik   = 1
    kpt  = basis.kpoints[ik]
    occk = occupation[ik]
    ψk   = ψ[ik]

    for (n, ψn) in enumerate(eachcol(ψk))
        ψn_real = ifft(basis, kpt, ψn)
        for (m, ψm) in enumerate(eachcol(ψk)) # TODO: restrict loop m<=n
            ψm_real  = ifft(basis, kpt, ψm) # TODO: avoid FFT inside loop (e.g. prepare chunks of ψm_real in advance)
            ρnm_real = conj(ψn_real) .* ψm_real
            ρnm_fourier = fft(basis, kpt, ρnm_real) # actually we need a q-point here

            fac_mn = occk[n] * occk[m] / T(2)
            E -= 0.5 * fac_mn * real(dot(ρnm_fourier .* term.poisson_green_coeffs, ρnm_fourier)) 
        end
    end

    ops = [ExchangeOperator(basis, kpt, term.poisson_green_coeffs, occk, ψk)]
    (; E, ops)
end


