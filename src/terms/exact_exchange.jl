@doc raw"""
Exact exchange term: the Hartree-Exact exchange energy of the orbitals
```math
-1/2 ∑_{nm} f_n f_m ∫∫ \frac{ψ_n^*(r)ψ_m^*(r')ψ_n(r')ψ_m(r)}{|r - r'|} dr dr'
```
"""

abstract type EXXstrategy end

struct ExactExchange{S <: EXXstrategy}
    scaling_factor::Real
    coulomb_kernel_model::CoulombKernelModel
    exx_strategy::S
end

ExactExchange(; scaling_factor=1, 
                coulomb_kernel_model=ProbeCharge(), 
                exx_strategy=CanonicalEXX()) = 
    ExactExchange(scaling_factor, coulomb_kernel_model, exx_strategy)

(exchange::ExactExchange)(basis) = TermExactExchange(basis, 
                                                     exchange.scaling_factor, 
                                                     exchange.coulomb_kernel_model,
                                                     exchange.exx_strategy)

function Base.show(io::IO, exchange::ExactExchange)
    fac = isone(exchange.scaling_factor) ? "" : "scaling_factor=$(exchange.scaling_factor), "
    print(io, "ExactExchange(coulomb_kernel_model=$(exchange.coulomb_kernel_model), exx_strategy=$(exchange.exx_strategy))")
end
struct TermExactExchange <: Term
    scaling_factor::Real  # scaling factor
    coulomb_kernel::AbstractArray
    exx_strategy::EXXstrategy 
end
function TermExactExchange(basis::PlaneWaveBasis{T}, 
                           scaling_factor, 
                           coulomb_kernel_model::CoulombKernelModel, 
                           exx_strategy::EXXstrategy) where T
    coulomb_kernel = compute_coulomb_kernel(basis; coulomb_kernel_model) # TODO: we need this for every q-point
    TermExactExchange(T(scaling_factor), coulomb_kernel, exx_strategy)
end

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange,
                                                  basis::PlaneWaveBasis{T}, ψ, occupation;
                                                  kwargs...) where {T}
    # check for the inconsistency first (ψ given but no occupation)
    if !isnothing(ψ) && isnothing(occupation)
        @warn "ψ provided but occupation is missing; cannot set up exact exchange."
    end
    # If either is missing, we cannot proceed
    if isnothing(ψ) || isnothing(occupation)
        return (; E=zero(T), ops=NoopOperator.(basis, basis.kpoints))
    end

    @assert length(ψ) == 1  # TODO: make it work for more kpoints

    # TODO Occupation threshold
    ψ, occupation = select_occupied_orbitals(basis, ψ, occupation; threshold=1e-8)

    compute_exx_ene_ops(term.exx_strategy, term.coulomb_kernel, basis, ψ, occupation)
end


"""
    CanonicalEXX

Canonical Fock exchange implementation.
"""
struct CanonicalEXX <: EXXstrategy end
function compute_exx_ene_ops(::CanonicalEXX,
                             coulomb_kernel::AbstractArray,
                             basis::PlaneWaveBasis{T}, ψ, occupation) where {T}
    ik   = 1
    kpt  = basis.kpoints[ik]
    occk = occupation[ik]
    ψk   = ψ[ik]
   
    nocc = size(ψk, 2) 
    ψk_real = similar(ψk, complex(T), basis.fft_size..., nocc)
    for i = 1:nocc
        ifft!(view(ψk_real,:,:,:,i), basis, kpt, ψk[:,i])
    end

    E = zero(T)
    for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
        for (m, ψmk_real) in enumerate(eachslice(ψk_real, dims=4))
            if m > n continue end
            ρmn_real = conj(ψmk_real) .* ψnk_real
            ρmn_fourier = fft(basis, kpt, ρmn_real) # actually we need a q-point here
            fac_mn = occk[n] * occk[m] / T(2)
            fac_mn *= (m != n ? 2 : 1) # factor 2 because we skipped m>n
            E -= 0.5 * fac_mn * real(dot(ρmn_fourier .* coulomb_kernel, ρmn_fourier)) 
        end
    end
    ops = [ExchangeOperator(basis, kpt, coulomb_kernel, occk, ψk, ψk_real)]
    (; E, ops)
end


"""
    ACEXX

Adaptively Compressed Exchange (ACE) implementation of the Fock exchange.

# Reference
JCTC 2016, 12, 5, 2242–2249, doi.org/10.1021/acs.jctc.6b00092
"""
struct ACEXX <: EXXstrategy end
function compute_exx_ene_ops(::ACEXX,
                             coulomb_kernel::AbstractArray,
                             basis::PlaneWaveBasis{T}, ψ, occupation) where {T}
    ik   = 1
    kpt  = basis.kpoints[ik]
    occk = occupation[ik]
    ψk   = ψ[ik]
   
    nocc = size(ψk, 2) 
    ψk_real = similar(ψk, complex(T), basis.fft_size..., nocc)
    for i = 1:nocc
        ifft!(view(ψk_real,:,:,:,i), basis, kpt, ψk[:,i])
    end

    E = zero(T)
    Wk = similar(ψk)
    Vn_real = zeros(complex(T), basis.fft_size...) 
    for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
        Wnk_real = zeros(complex(T), basis.fft_size...)
        for (m, ψmk_real) in enumerate(eachslice(ψk_real, dims=4))
            ρmn_real = conj(ψmk_real) .* ψnk_real
            ρmn_fourier = fft(basis, kpt, ρmn_real) # actually we need a q-point here

            Vmn_fourier = ρmn_fourier .* coulomb_kernel
            Vmn_real = ifft(basis, kpt, Vmn_fourier)
            Wnk_real .+= occk[m]/T(2) .* ψmk_real .* Vmn_real
            
            fac_mn = occk[n] * occk[m] / T(2)
            E -= 0.5 * fac_mn * real(dot(ρmn_fourier .* coulomb_kernel, ρmn_fourier)) 
        end
        Wk[:, n] .= fft(basis, kpt, Wnk_real)
    end
    M = ψk' * Wk
    M = Hermitian(0.5*(M+M'))
    B = inv(M)
    ops = [ACExchangeOperator(basis, kpt, Wk, B)]
    (; E, ops)
end
