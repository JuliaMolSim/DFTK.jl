"""Abstract types for different algorithms to evaluate exact exchange"""
abstract type ExxAlgorithm end

@doc raw"""
Term for (possibly screened) Hartree-Fock exact exchange energy of the form
```math
-1/2 ∑_{nm} f_n f_m ∫∫ ψ_n^*(r)ψ_m^*(r') kernel(r, r')  ψ_n(r')ψ_m(r) dr dr'
```
where the `kernel` keyword argument is an [`InteractionKernel`](@ref) , typically
- the untruncated, unscreened [`Coulomb`](@ref) kernel `G(r, r') = 1/|r - r'|` for
  Hartree-Fock exact exchange, by default some form of regularisation is applied,
  see e.g. [`ProbeCharge`](@ref).
- [`SphericallyTruncatedCoulomb`](@ref) for a Coulomb kernel with truncated range, that
  converges faster with the ``k``-point grid.
- [`ShortRangeCoulomb`](@ref) the `erf`-truncated short-range Coulomb kernel
- [`LongRangeCoulomb`](@ref) the `erf`-truncated long-range Coulomb kernel
"""
@kwdef struct ExactExchange
    scaling_factor::Real = 1.0
    kernel = Coulomb()
end
(ex::ExactExchange)(basis) = TermExactExchange(basis, ex.scaling_factor, ex.kernel)

struct TermExactExchange <: Term
    scaling_factor::Real  # scaling factor, absorbed into interaction_kernel
    interaction_kernel::AbstractArray  # kernel values in Fourier space
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor, kernel) where {T}
    # TODO: we need this for every q-point
    fac::T = scaling_factor
    interaction_kernel = fac .* compute_kernel_fourier(kernel, basis)
    TermExactExchange(fac, interaction_kernel)
end

@timing "ene_ops: ExactExchange" function ene_ops(term::TermExactExchange, basis::PlaneWaveBasis{T},
                                                  ψ, occupation;
                                                  occupation_threshold=zero(T),
                                                  exxalg::ExxAlgorithm=VanillaExx(),
                                                  kwargs...) where {T}
    if isnothing(ψ) || isnothing(occupation)
        @warn "Exact exchange requires orbitals and occupation, return NoopOperator." 
        return (; E=zero(T), ops=NoopOperator.(basis, basis.kpoints))
    end

    mask_occ = occupied_empty_masks(occupation, occupation_threshold).mask_occ
    @assert length(basis.kpoints) == basis.model.n_spin_components  # no k-points, only spin

    E = zero(T)
    ops = []
    for (ik, kpt) in enumerate(basis.kpoints)
        (; Ek, opk) = exx_operator(exxalg, basis, kpt, term.interaction_kernel,
                                   ψ[ik], occupation[ik], mask_occ[ik])
        push!(ops, opk)
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
function exx_operator(::VanillaExx, basis::PlaneWaveBasis{T}, kpt, interaction_kernel,
                      ψk, occk, maskk_occ) where {T}
    # Perform FFT on occupied orbitals only
    ψk_occ = @view ψk[:, maskk_occ]
    ψk_occ_real = similar(ψk, basis.fft_size..., length(maskk_occ))
    for (ψnk_real, ψnk) in zip(eachslice(ψk_occ_real; dims=4), eachcol(ψk_occ))
        ifft!(ψnk_real, basis, kpt, ψnk)
    end

    # Compute energy and build exchange operator
    occk_occ = @view occk[maskk_occ]
    Ek  = exx_energy_only(basis,  kpt, interaction_kernel, ψk_occ_real, occk_occ)
    opk = ExchangeOperator(basis, kpt, interaction_kernel, occk_occ, ψk_occ, ψk_occ_real)
    (; Ek, opk)
end

"""
Adaptively Compressed Exchange (ACE) implementation of the Fock exchange.
Note, that this sketches the exchange operator using the currently available orbitals.
By default (`sketch_with_extra_orbitals=true`) both the occupied and the extra (unconverged)
orbitals are used for sketching. In contrast, with `sketch_with_extra_orbitals=false`
only the occupied orbitals are used for sketching. `sketch_with_extra_orbitals=false`
is the setting of most DFT codes (e.g. QuantumEspresso), since it is slightly less
expensive per SCF step and has a lower memory footprint. However,
`sketch_with_extra_orbitals=true` leads to a more stable SCF, which often converges
in less SCF iterations.

# Reference
JCTC 2016, 12, 5, 2242-2249, doi.org/10.1021/acs.jctc.6b00092
"""
@kwdef struct AceExx <: ExxAlgorithm
    # Sketch using both converged and extra (non-converged and unoccupied) orbitals
    # or using only the occupied orbitals (false).
    sketch_with_extra_orbitals::Bool = true
end 
function exx_operator(ace::AceExx, basis::PlaneWaveBasis{T}, kpt, interaction_kernel,
                      ψk, occk, maskk_occ) where {T}
    # Perform FFT on orbitals
    fftmask = ace.sketch_with_extra_orbitals ? (1:size(ψk, 2)) : maskk_occ
    ψk_real = similar(ψk, basis.fft_size..., length(fftmask))
    for (ψnk_real, n) in zip(eachslice(ψk_real; dims=4), fftmask)
        ifft!(ψnk_real, basis, kpt, @view ψk[:, n])
    end

    # Build exchange operator only using occupied orbitals (to save on FFTs)
    occk_occ    = @view occk[maskk_occ]
    ψk_occ      = @view ψk[:, maskk_occ]
    ψk_occ_real = @view ψk_real[:, :, :, maskk_occ]
    Vxk = ExchangeOperator(basis, kpt, interaction_kernel, occk_occ, ψk_occ, ψk_occ_real)

    # Apply ACE compression using as sketch space all orbitals the user wants for compression
    occk_comp    = @view occk[fftmask]
    ψk_comp      = @view ψk[:, fftmask]
    ψk_comp_real = @view ψk_real[:, :, :, fftmask]
    (; opk, Mk) = compress_exchange(Vxk, ψk_comp, ψk_comp_real)

    # The compression computes Wnk = Vxk * ψnk  for all passed ψnk.
    # Therefore [Mk]_{nm} is just < ψ_{nk}, Vxk ψmk>, which means that the
    # energy contribution from this k-point can be computed as
    #     1/2 * ∑_n occ_{nk} [Mk]_{nn}
    Ek = 1/T(2) * real(tr(Diagonal(Mk) * Diagonal(occk_comp)))

    (; Ek, opk)
end

# Sketch exchange operator of a particular k-point using the passed orbitals
# in real and Fourier space
function compress_exchange(Vxk::ExchangeOperator, ψk::AbstractMatrix,
                           ψk_real::AbstractArray{T,4}) where {T}
    basis = Vxk.basis
    kpt = Vxk.kpoint

    Wk = similar(ψk)
    Wnk_real_tmp = similar(ψk_real[:, :, :, 1])
    for (Wnk, ψnk_real) in zip(eachcol(Wk), eachslice(ψk_real, dims=4))
        # Compute Wnk = Vxk * ψnk in real space
        Wnk_real_tmp .= 0
        apply!((; real=Wnk_real_tmp), Vxk, (; real=ψnk_real))
        fft!(Wnk, basis, kpt, Wnk_real_tmp)
    end

    Mk  = Hermitian(ψk' * Wk)
    Bk  = InverseNegatedMap(cholesky(-Mk))
    opk = NonlocalOperator(basis, kpt, Wk, Bk)
    (; opk, Mk, Bk, Wk)
end

struct InverseNegatedMap{T}
    B::T
end
Base.:*(op::InverseNegatedMap, x) = -(op.B \ x)


function exx_energy_only(basis::PlaneWaveBasis{T}, kpt, interaction_kernel, ψk_real, occk) where {T}
    # Naive algorithm for computing the exact exchange energy only.

    Ek = zero(T)
    for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
        for (m, ψmk_real) in enumerate(eachslice(ψk_real, dims=4))
            m > n && continue
            ρmn_real = conj(ψmk_real) .* ψnk_real
            ρmn_fourier = fft(basis, kpt, ρmn_real) # actually we need a q-point here

            # Exact exchange is quadratic in occupations but linear in spin,
            # hence we need to undo the fact that in DFTK for non-spin-polarized calcuations
            # orbitals are considered as spin orbitals and thus occupations run from 0 to 2
            # We do this by dividing by the filled_occupation.
            fac_mn = occk[n] * occk[m] / filled_occupation(basis.model)

            fac_mn *= (m != n ? 2 : 1) # factor 2 because we skipped m>n
            Ek -= 1/T(2) * fac_mn * real(dot(ρmn_fourier .* interaction_kernel, ρmn_fourier))
        end
    end
    Ek
end
