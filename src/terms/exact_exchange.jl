@doc raw"""
Exact exchange term: the Hartree-Exact exchange energy of the orbitals
```math
-1/2 ∑_{nm} f_n f_m ∫∫ \frac{ψ_n^*(r)ψ_m^*(r')ψ_n(r')ψ_m(r)}{|r - r'|} dr dr'
```
"""
struct ExactExchange
    scaling_factor::Real  # to scale the term
    coulomb_kernel_model::CoulombKernelModel
    approximation::Symbol # :none or :ace
end

ExactExchange(; scaling_factor=1, coulomb_kernel_model=ProbeCharge(), approximation=:none) = 
    ExactExchange(scaling_factor, coulomb_kernel_model, approximation)

(exchange::ExactExchange)(basis) = TermExactExchange(basis, 
                                                     exchange.scaling_factor, 
                                                     exchange.coulomb_kernel_model,
                                                     exchange.approximation)

function Base.show(io::IO, exchange::ExactExchange)
    fac = isone(exchange.scaling_factor) ? "" : "scaling_factor=$(exchange.scaling_factor), "
    print(io, "ExactExchange($coulomb_kernel_model=$(exchange.coulomb_kernel_model))")
end
struct TermExactExchange <: Term
    scaling_factor::Real  # scaling factor
    coulomb_kernel::AbstractArray
    approximation::Symbol 
end
function TermExactExchange(basis::PlaneWaveBasis{T}, 
                           scaling_factor, 
                           coulomb_kernel_model::CoulombKernelModel, 
                           approximation::Symbol) where T
    coulomb_kernel = compute_coulomb_kernel(basis; coulomb_kernel_model=coulomb_kernel_model) # TODO: we need this for every q-point
    TermExactExchange(T(scaling_factor), coulomb_kernel, approximation)
end

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange,
                                                  basis::PlaneWaveBasis{T}, ψ, occupation;
                                                  kwargs...) where {T}
    # check for the inconsistency first (ψ given but no occupation)
    if !isnothing(ψ) && isnothing(occupation)
        @warn "ψ provided but occupation is missing; cannot set up calculate exact exchange."
    end
    # If either is missing, we cannot proceed
    if isnothing(ψ) || isnothing(occupation)
        return (; E=zero(T), ops=NoopOperator.(basis, basis.kpoints))
    end

    # TODO Occupation threshold
    ψ, occupation = select_occupied_orbitals(basis, ψ, occupation; threshold=1e-8)

    E = zero(T)

    @assert length(ψ) == 1  # TODO: make it work for more kpoints
    ik   = 1
    kpt  = basis.kpoints[ik]
    occk = occupation[ik]
    ψk   = ψ[ik]
   
    nocc = size(ψk, 2) 
    ψk_real = similar(ψk, complex(T), basis.fft_size..., nocc)
    for i = 1:nocc
        ifft!(view(ψk_real,:,:,:,i), basis, kpt, ψk[:,i])
    end

    #### ACE ####
    if term.approximation == :ace
        Wk = similar(ψk)
        Vn_real = zeros(complex(T), basis.fft_size...) 
    end
    #############

    for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
        
        #### ACE ####
        if term.approximation == :ace
            Wnk_real = zeros(complex(T), basis.fft_size...)
        end
        #############
        
        for (m, ψmk_real) in enumerate(eachslice(ψk_real, dims=4))
            #if m > n continue end
            ρmn_real = conj(ψmk_real) .* ψnk_real
            ρmn_fourier = fft(basis, kpt, ρmn_real) # actually we need a q-point here

            #### ACE ####
            if term.approximation == :ace
                Vmn_fourier = ρmn_fourier .* term.coulomb_kernel
                Vmn_real = ifft(basis, kpt, Vmn_fourier)
                Wnk_real .+= 0.5*occk[m]/T(2) .* ψmk_real .* Vmn_real
            end
            #############

            fac_mn = occk[n] * occk[m] / T(2)
            #fac_mn *= (m != n ? 2 : 1) # factor 2 because we skipped m>n
            E -= 0.5 * fac_mn * real(dot(ρmn_fourier .* term.coulomb_kernel, ρmn_fourier)) 
        end

        #### ACE ####
        if term.approximation == :ace
            Wk[:, n] .= fft(basis, kpt, Wnk_real)
        end
        #############
    end
     
    #### ACE ####
    if term.approximation == :ace
        M = ψk' * Wk
        M = Hermitian(0.5*(M+M'))
        B = inv(M)
        ops = [ACExchangeOperator(basis, kpt, Wk, B)]
    #############
    else
        ops = [ExchangeOperator(basis, kpt, term.coulomb_kernel, occk, ψk, ψk_real)]
    end

    (; E, ops)
end


