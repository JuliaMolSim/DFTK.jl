@doc raw"""
Exact exchange term: the Hartree-Exact exchange energy of the orbitals
```math
-1/2 ∑_{nm} f_n f_m ∫∫ \frac{ψ_n^*(r)ψ_m^*(r')ψ_n(r')ψ_m(r)}{|r - r'|} dr dr'
```
"""

abstract type ExxAlgorithm end

@kwdef struct ExactExchange
    scaling_factor::Real = 1.0
    coulomb_kernel_model::CoulombKernelModel = ProbeCharge()
    exx_algorithm::ExxAlgorithm = AceExx()
end
function (exchange::ExactExchange)(basis)
    TermExactExchange(basis, exchange.scaling_factor,
    exchange.coulomb_kernel_model, exchange.exx_algorithm)
end

struct TermExactExchange <: Term
    scaling_factor::Real  # scaling factor, absorbed into coulomb_kernel
    coulomb_kernel::AbstractArray
    exx_algorithm::ExxAlgorithm
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor, coulomb_kernel_model, exx_algorithm) where {T}
    # TODO: we need this for every q-point
    fac::T = scaling_factor
    coulomb_kernel = fac .* compute_coulomb_kernel(basis; coulomb_kernel_model)
    TermExactExchange(fac, coulomb_kernel, exx_algorithm)
end

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange, basis::PlaneWaveBasis{T},
                                                  ψ, occupation; occupation_threshold=zero(T),
                                                  kwargs...) where {T}
    if isnothing(ψ) || isnothing(occupation)
        @warn "Exact exchange requires orbitals and occupation, return NoopOperator." 
        return (; E=zero(T), ops=NoopOperator.(basis, basis.kpoints))
    end

    mask_occ = occupied_empty_masks(occupation, occupation_threshold).mask_occ
    ψ_occ    = @views [ψ[ik][:, maskk]        for (ik, maskk) in enumerate(mask_occ)]
    occ_occ  = @views [occupation[ik][maskk]  for (ik, maskk) in enumerate(mask_occ)]

    @assert length(basis.kpoints) == basis.model.n_spin_components # no k-points, only spin

    E = zero(T)
    ops = []
    for (ik, kpt) in enumerate(basis.kpoints)
        occk = occ_occ[ik]
        ψk   = ψ_occ[ik]

        n_occ = length(occk)
        ψk_real = similar(ψk, complex(T), basis.fft_size..., n_occ)
        @views for n = 1:n_occ
            ifft!(ψk_real[:, :, :, n], basis, kpt, ψk[:, n])
        end

        # TODO: Actually for ACE it probably makes sense to pass *all* orbitals for sketching
        #       and not just the ones with occupation above the threshold (especially for metals),
        #       see the note in the AceExx routines below.
        (; Ek, op) = exx_operator(term.exx_algorithm, basis, kpt,
                                  term.coulomb_kernel, ψk, ψk_real, occk)
        push!(ops, op)
        E += Ek  # TODO: Need kweight here later for energy; see also non-local term for ideas
    end
    (; E, ops)
end

# TODO: Should probably define an energy-only function, which directly calls into
#       exx_energy_only for both ACE and Vanilla version.


"""
Plain vanilla Fock exchange implementation without any tricks.
"""
struct VanillaExx <: ExxAlgorithm end
function exx_operator(::VanillaExx, basis::PlaneWaveBasis{T}, kpt, coulomb_kernel,
                      ψk, ψk_real, occk) where {T}
    Ek = exx_energy_only(basis, kpt, coulomb_kernel, ψk_real, occk)
    op = ExchangeOperator(basis, kpt, coulomb_kernel, occk, ψk, ψk_real)
    (; Ek, op)
end


"""
Adaptively Compressed Exchange (ACE) implementation of the Fock exchange.
Note, that this sketches the exchange operator using the occupied orbitals. ACE is therefore
inaccurate when applying the compressed exchange operator to virtual orbitals.

# Reference
JCTC 2016, 12, 5, 2242-2249, doi.org/10.1021/acs.jctc.6b00092
"""
struct AceExx <: ExxAlgorithm end 
function exx_operator(::AceExx, basis::PlaneWaveBasis{T}, kpt, coulomb_kernel::AbstractArray,
                      ψk, ψk_real, occk) where {T}
    # TODO: Higher accuracy (especially for systems with large mixing between
    #       occupied and unoccupied orbitals during the SCF) can probably be
    #       achieved by sketching also with the orbitals with occupation below
    #       the occupation_threshold, i.e. by making ψk and ψk_real hold all SCF orbitals
    #       (including non-converged). Probably it makes sense in general to enable
    #       sketching with arbitrary orbitals (and not just the occupied ones).

    # Perform sketch of the exchange operator with the orbitals ψk
    Wk = similar(ψk)
    Wnk_real_tmp = similar(ψk_real[:, :, :, 1])
    Vx = ExchangeOperator(basis, kpt, coulomb_kernel, occk, ψk, ψk_real)
    for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
        # Compute Wnk = Vx * ψnk in real space
        Wnk_real_tmp .= 0
        apply!((; real=Wnk_real_tmp), Vx, (; real=ψnk_real))
        Wk[:, n] .= fft(basis, kpt, Wnk_real_tmp)
    end

    # In ACE Wnk = Vx * ψnk  for all occupied ψnk.
    # Therefore [Mk]_{nm} is just < ψ_{nk}, V_x ψmk>, which means that the
    # energy contribution from this k-point can be computed as
    #     ∑_n occ_{nk} [Mk]_{nn}
    Mk = Hermitian(ψk' * Wk)
    Ek = 1/T(2) * real(tr(Diagonal(Mk) * Diagonal(occk)))
    Bk = InverseNegatedMap(cholesky(-Mk))
    op = NonlocalOperator(basis, kpt, Wk, Bk)
    (; Ek, op)
end


struct InverseNegatedMap{T}
    B::T
end
Base.:*(op::InverseNegatedMap, x) = -(op.B \ x)


function exx_energy_only(basis::PlaneWaveBasis{T}, kpt, coulomb_kernel, ψk_real, occk) where {T}
    # Naive algorithm for computing the exact exchange energy only.

    Ek = zero(T)
    for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
        for (m, ψmk_real) in enumerate(eachslice(ψk_real, dims=4))
            if m > n continue end
            ρmn_real = conj(ψmk_real) .* ψnk_real
            ρmn_fourier = fft(basis, kpt, ρmn_real) # actually we need a q-point here

            # Exact exchange is quadratic in occupations but linear in spin,
            # hence we need to undo the fact that in DFTK for non-spin-polarized calcuations
            # orbitals are considered as spin orbitals and thus occupations run from 0 to 2
            # We do this by dividing by the filled_occupation.
            fac_mn = occk[n] * occk[m] / filled_occupation(basis.model)

            fac_mn *= (m != n ? 2 : 1) # factor 2 because we skipped m>n
            Ek -= 1/T(2) * fac_mn * real(dot(ρmn_fourier .* coulomb_kernel, ρmn_fourier))
        end
    end
    Ek
end
