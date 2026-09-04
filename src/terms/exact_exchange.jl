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
- [`SphericallyTruncatedCoulomb`](@ref) and 
- [`WignerSeitzTruncatedCoulomb`](@ref) for a Coulomb kernels with truncated range, 
  that converge faster with the size of the Born-von Karman cell.
- [`ShortRangeCoulomb`](@ref) the `erf`-truncated short-range Coulomb kernel
- [`LongRangeCoulomb`](@ref) the `erf`-truncated long-range Coulomb kernel
"""
@kwdef struct ExactExchange
    scaling_factor::Real = 1.0
    kernel = Coulomb()
end
(ex::ExactExchange)(basis) = TermExactExchange(basis, ex.scaling_factor, ex.kernel)
breaks_symmetries(::ExactExchange) = true  # TODO: make ExactExchange fit for symmetries 

struct TermExactExchange{T, Tkernel, Tq, Tmap} <: Term
    scaling_factor::T             # scaling factor, absorbed into interaction_kernels
    interaction_kernels::Tkernel  # Vector{Array{T,3}}: kernel on the FFT cube, one per q
    q_points::Tq                  # Vector{Kpoint{T}}
    kprime_mapping::Tmap          # Matrix{Int}: find index for k'=k-q
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor, kernel) where {T}
    if mpi_nprocs(basis.comm_kpts) > 1
        error("ExactExchange currently does not support MPI parallelization over k-points.")
    end
    fac::T = scaling_factor 

    q_points, kprime_mapping = build_qpoints(basis)
    if any(iszero, kprime_mapping)
        error("ExactExchange requires a k-point set which is closed under differences, " *
              "i.e. for each k-point k and momentum transfer q = k - k' the point k - q " *
              "has to be a k-point of the basis as well. This is the case for a full " *
              "(non-symmetry-reduced) k-point grid.")
    end
    interaction_kernels = fac .* compute_kernel_fourier(kernel, basis, q_points)

    TermExactExchange(fac, interaction_kernels, q_points, kprime_mapping)
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

    # get occupied orbitals only 
    mask_occ = occupied_empty_masks(occupation, occupation_threshold).mask_occ
    occ_occ = [occupation[ik][mask_occ[ik]] for ik in 1:length(basis.kpoints)]

    # Precompute all occupied real-space orbitals once per SCF step
    ψ_occ_real = map(1:length(basis.kpoints)) do ik
        kpt = basis.kpoints[ik]
        mask = mask_occ[ik]
        ψk_occ = @view ψ[ik][:, mask]
        ψk_real = similar(ψ[ik], basis.fft_size..., length(mask))
        for (ψnk_real, ψnk) in zip(eachslice(ψk_real; dims=4), eachcol(ψk_occ))
            ifft!(ψnk_real, basis, kpt, ψnk)
        end
        ψk_real
    end

    E = zero(T)
    ops = []
    for (ik, kpt) in enumerate(basis.kpoints)
        (; Ek, opk) = build_exx(exxalg, basis, kpt, term, ψ, occupation, mask_occ,
                                ψ_occ_real, occ_occ)
        push!(ops, opk)
        E += Ek * basis.kweights[ik]
    end
    (; E, ops)
end


"""
Plain vanilla Fock exchange implementation without any tricks.
"""
struct VanillaExx <: ExxAlgorithm end
function build_exx(::VanillaExx, basis::PlaneWaveBasis{T}, kpt, term::TermExactExchange,
                     ψ, occupation, mask_occ, ψ_occ_real, occ_occ) where {T}
    
    Ek = exx_energy_only(basis, kpt, term.interaction_kernels, term.q_points, 
                         term.kprime_mapping, ψ_occ_real, occ_occ)

    opk = ExchangeOperator(basis, kpt, term.interaction_kernels, term.q_points, 
                           term.kprime_mapping, ψ_occ_real, occ_occ)
    (; Ek, opk)
end

# TODO: Should probably define an energy-only function, which directly calls into
#       exx_energy_only for both ACE and Vanilla version.


# TODO this is currently called only with interaction_kernel for one q (see build_exx)
# Naive algorithm for computing the exact exchange energy only.
function exx_energy_only(basis::PlaneWaveBasis{T}, kpt, interaction_kernels, q_points,
                         kprime_mapping, ψ_occ_real, occ_occ) where {T}

    # get occupied orbitals at k
    ik = findfirst(isequal(kpt), basis.kpoints)
    ψk_real = ψ_occ_real[ik]

    Ek = zero(T)

    # outer loop over q
    for iq in 1:length(q_points)

        # construct k' = k - q
        ikp = kprime_mapping[ik, iq]
        ikp == 0 && continue 
        
        # get the Coulomb kernel Fourier components G+q
        qpt = q_points[iq]
        kernel_q = interaction_kernels[iq]
        
        # get occupied orbitals at k'
        ψkp_real = ψ_occ_real[ikp]

        # Calculate the reciprocal lattice shift Gb from BZ folding
        # k - q = k' + G_b  =>  G_b = k - q - k'
        kpt_prime = basis.kpoints[ikp]
        Gb_frac = kpt.coordinate - kpt_prime.coordinate - qpt.coordinate
        Gb = round.(Int, Gb_frac)
        phase_shift = map(r -> cis2pi(dot(Gb, r)), r_vectors(basis))

        for (n, ψnk_real) in enumerate(eachslice(ψk_real, dims=4))
            for (m, ψmkp_real) in enumerate(eachslice(ψkp_real, dims=4))
                m > n && ik == ikp && continue
                
                ρmn_real = conj(ψmkp_real) .* ψnk_real .* phase_shift
                ρmn_fourier = fft(basis, ρmn_real) 

                # Exact exchange is quadratic in occupations but linear in spin,
                # hence we need to undo the fact that in DFTK for non-spin-polarized calcuations
                # orbitals are considered as spin orbitals and thus occupations run from 0 to 2
                # We do this by dividing by the filled_occupation.
                fac_mn = occ_occ[ik][n] * occ_occ[ikp][m] / filled_occupation(basis.model)
                
                fac_mn *= basis.kweights[ikp] # k'-weight 
                fac_mn *= ((m != n && ik == ikp) ? 2 : 1)  # factor 2 because we skipped m>n    

                Ek -= 1/T(2) * fac_mn * real(dot(ρmn_fourier .* kernel_q, ρmn_fourier)) 
            end
        end
    end
    Ek
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
function build_exx(ace::AceExx, basis::PlaneWaveBasis{T}, kpt, term::TermExactExchange,
                   ψ, occupation, mask_occ, ψ_occ_real, occ_occ) where {T}
    # Occupied views using mask_occ
    ψ_occ = [@view ψ[ik][:, mask_occ[ik]] for ik in 1:length(basis.kpoints)]
    occ_occ = [occupation[ik][mask_occ[ik]] for ik in 1:length(basis.kpoints)]

    # Build the ExchangeOperator K acting on orbital at k
    Kk = ExchangeOperator(basis, kpt, term.interaction_kernels, term.q_points, 
                          term.kprime_mapping, ψ_occ_real, occ_occ)

    # Build mask for ACE sketch orbitals
    ik = findfirst(isequal(kpt), basis.kpoints)
    mask_sketch = ace.sketch_with_extra_orbitals ? (1:size(ψ[ik], 2)) : mask_occ[ik]

    # FFT only the target sketch orbitals to real space
    ψk_sketch = @view ψ[ik][:, mask_sketch]
    ψk_sketch_real = similar(ψ[ik], basis.fft_size..., length(mask_sketch))
    for (ψnk_real, ψnk) in zip(eachslice(ψk_sketch_real; dims=4), eachcol(ψk_sketch))
        ifft!(ψnk_real, basis, kpt, ψnk)
    end

    # Apply ACE compression using as sketch space all orbitals the user wants for compression
    (; opk, Mk) = compress_exchange(Kk, ψk_sketch, ψk_sketch_real)

    # Energy computation
    #
    # The compression computes Wnk = Kk * ψnk  for all passed ψnk.
    # Therefore [Mk]_{nm} is just < ψ_{nk}, Kk ψmk>, which means that the
    # energy contribution from this k-point can be computed as
    #     1/2 * ∑_n occ_{nk} [Mk]_{nn}
    occk_sketch = @view occupation[ik][mask_sketch]
    Ek = 1/T(2) * real(tr(Diagonal(Mk) * Diagonal(occk_sketch)))

    (; Ek, opk)
end

# Sketch exchange operator of a particular k-point using the passed orbitals
# in real and Fourier space
function compress_exchange(Kk::ExchangeOperator, ψk::AbstractMatrix,
                           ψk_real::AbstractArray{T,4}) where {T}
    basis = Kk.basis
    kpt = Kk.kpoint

    Wk = similar(ψk)
    Wnk_real_tmp = similar(ψk_real, size(ψk_real)[1:3]...)
    for (Wnk, ψnk_real) in zip(eachcol(Wk), eachslice(ψk_real, dims=4))
        # Compute Wnk = Kk * ψnk in real space
        Wnk_real_tmp .= 0
        apply!((; real=Wnk_real_tmp), Kk, (; real=ψnk_real))
        fft!(Wnk, basis, kpt, Wnk_real_tmp)
    end

    Mk = Hermitian(ψk' * Wk)
    # -Mk is positive semi-definite. We use shifted_cholesky to handle
    # potentially rank-deficient cases (e.g. HEG or extra bands).
    Bk = InverseNegatedMap(shifted_cholesky(-Mk))
    opk = NonlocalOperator(basis, kpt, Wk, Bk)
    (; opk, Mk, Bk, Wk)
end

struct InverseNegatedMap{T}
    B::T
end
Base.:*(op::InverseNegatedMap, x) = -(op.B \ x)
