"""Abstract types for different algorithms to evaluate exact exchange"""
abstract type ExxAlgorithm end

@doc raw"""
Term for (possibly screened) Hartree-Fock exact exchange energy of the form
```math
-1/2 ∑_{nm} f_n f_m ∫∫ ψ_n^*(r)ψ_m^*(r') kernel(r, r')  ψ_n(r')ψ_m(r) dr dr'
```
where the `kernel` keyword argument is an [`InteractionKernel`](@ref) , typically
- the untruncated, unscreened [`BareCoulomb`](@ref) kernel `G(r, r') = 1/|r - r'|` for
  Hartree-Fock exact exchange, wrapped in a singularity treatment, by default
  [`ProbeCharge`](@ref).
- [`SphericallyTruncatedCoulomb`](@ref) and 
- [`WignerSeitzTruncatedCoulomb`](@ref) for a Coulomb kernels with truncated range, 
  that converge faster with the size of the Born-von Karman cell.
- [`ShortRangeCoulomb`](@ref) the `erf`-truncated short-range Coulomb kernel
- [`LongRangeCoulomb`](@ref) the `erf`-truncated long-range Coulomb kernel
"""
@kwdef struct ExactExchange
    scaling_factor::Real = 1.0
    kernel = ProbeCharge(BareCoulomb())
end
(ex::ExactExchange)(basis) = TermExactExchange(basis, ex.scaling_factor, ex.kernel)
breaks_symmetries(::ExactExchange) = true  # TODO: make ExactExchange fit for symmetries 

struct TermExactExchange{T, Tkernel, Tq} <: Term
    scaling_factor::T             # scaling factor, absorbed into interaction_kernels
    interaction_kernels::Tkernel  # Vector{Array{T,3}}: kernel on the FFT cube, one per q
    q_points::Tq                  # Vector{Vec3{T}}: momentum transfers (reduced coordinates)
end
function TermExactExchange(basis::PlaneWaveBasis{T}, scaling_factor, kernel) where {T}
    if mpi_nprocs(basis.comm_kpts) > 1
        error("ExactExchange currently does not support MPI parallelization over k-points.")
    end
    if !(basis.kgrid isa MonkhorstPack) || basis.n_irreducible_kpoints != length(basis.kgrid)
        error("ExactExchange requires a full (non-symmetry-reduced) Monkhorst-Pack grid.")
    end
    fac::T = scaling_factor 

    # Exchange couples each k to k - q for all q. For a full Monkhorst-Pack grid every
    # k - q is again a k-point of the basis (up to a G vector).
    q_points = build_qpoints(basis)
    interaction_kernels = eval_kernel_fourier(kernel, basis, q_points)
    if !all(kernel_fourier -> all(isfinite, kernel_fourier), interaction_kernels)
        error("Interaction kernel $(nameof(typeof(kernel))) is singular at G+q=0. Wrap it " *
              "in a singularity treatment, e.g. ProbeCharge($(nameof(typeof(kernel)))()).")
    end

    TermExactExchange(fac, fac .* interaction_kernels, q_points)
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
    exx_operator(exxalg, basis, term.interaction_kernels, term.q_points,
                 ψ, occupation, mask_occ)
end

# Transform the orbitals selected by masks[ik] at each k-point to real space
function _orbitals_real(basis::PlaneWaveBasis, ψ, masks)
    map(basis.kpoints, ψ, masks) do kpt, ψk, mask
        ψk_real = similar(ψk, basis.fft_size..., length(mask))
        for (ψnk_real, ψnk) in zip(eachslice(ψk_real; dims=4), eachcol(@view ψk[:, mask]))
            ifft!(ψnk_real, basis, kpt, ψnk)
        end
        ψk_real
    end
end


"""
Plain vanilla Fock exchange implementation without any tricks.
"""
struct VanillaExx <: ExxAlgorithm end
function exx_operator(::VanillaExx, basis::PlaneWaveBasis{T}, interaction_kernels, q_points,
                      ψ, occupation, mask_occ) where {T}
    occ_occ    = [to_cpu(occupation[ik][mask_occ[ik]]) for ik in 1:length(basis.kpoints)]
    ψ_occ_real = _orbitals_real(basis, ψ, mask_occ)

    E = sum(1:length(basis.kpoints)) do ik
        basis.kweights[ik] * exx_energy_only(basis, ik, interaction_kernels, q_points,
                                             ψ_occ_real, occ_occ)
    end
    ops = [ExchangeOperator(basis, kpt, interaction_kernels, q_points, ψ_occ_real, occ_occ)
           for kpt in basis.kpoints]
    (; E, ops)
end

# The energy alone does not need the exchange operator (nor its ACE compression),
# so we evaluate it directly from the occupied orbitals, independent of the ExxAlgorithm.
@timing "energy: ExactExchange" function energy(term::TermExactExchange,
                                                basis::PlaneWaveBasis{T}, ψ, occupation;
                                                occupation_threshold=zero(T),
                                                kwargs...) where {T}
    if isnothing(ψ) || isnothing(occupation)
        @warn "Exact exchange requires orbitals and occupation, return zero energy."
        return zero(T)
    end

    mask_occ   = occupied_empty_masks(occupation, occupation_threshold).mask_occ
    occ_occ    = [to_cpu(occupation[ik][mask_occ[ik]]) for ik in 1:length(basis.kpoints)]
    ψ_occ_real = _orbitals_real(basis, ψ, mask_occ)
    sum(1:length(basis.kpoints)) do ik
        basis.kweights[ik] * exx_energy_only(basis, ik, term.interaction_kernels,
                                             term.q_points, ψ_occ_real, occ_occ)
    end
end

# Naive algorithm for computing the exact exchange energy only.
function exx_energy_only(basis::PlaneWaveBasis{T}, ik, interaction_kernels, q_points,
                         ψ_occ_real, occ_occ::Vector{<:Vector}) where {T}

    # get occupied orbitals at k
    kpt = basis.kpoints[ik]
    ψk_real = ψ_occ_real[ik]

    # Energy contribution of each occupied orbital n at k, such that the threads
    # in the loop over n below never write to the same entry
    Ek_n = zeros(T, size(ψk_real, 4))

    # kernel_q holds the Fourier components of the interaction kernel at G+q
    for (q, kernel_q) in zip(q_points, interaction_kernels)
        # k - q is equivalent to the basis k-point k' = k - q + ΔG
        ikp, ΔG = find_equivalent_kpt(basis, kpt.coordinate - q, kpt.spin)
        phase_shift = map(r -> cis2pi(-dot(ΔG, r)), r_vectors(basis))

        # get occupied orbitals at k'
        ψkp_real = ψ_occ_real[ikp]

        parallel_loop_over_range(1:size(ψk_real, 4)) do n
            ψnk_real = @view ψk_real[:, :, :, n]
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

                Ek_n[n] -= 1/T(2) * fac_mn * real(dot(ρmn_fourier .* kernel_q, ρmn_fourier))
            end
        end
    end
    sum(Ek_n)
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
function exx_operator(ace::AceExx, basis::PlaneWaveBasis{T}, interaction_kernels, q_points,
                      ψ, occupation, mask_occ) where {T}
    mask_sketch = ace.sketch_with_extra_orbitals ? [1:size(ψk, 2) for ψk in ψ] : mask_occ
    occ_occ     = [to_cpu(occupation[ik][mask_occ[ik]]) for ik in 1:length(basis.kpoints)]

    # FFT the sketch orbitals once. They include the occupied orbitals, which are the
    # leading columns (mask_occ[ik] = 1:n_occ), so the latter can be taken as views.
    ψ_sketch_real = _orbitals_real(basis, ψ, mask_sketch)
    ψ_occ_real = [@view ψk_sketch_real[:, :, :, mask]
                  for (ψk_sketch_real, mask) in zip(ψ_sketch_real, mask_occ)]

    E = zero(T)
    ops = []
    for (ik, kpt) in enumerate(basis.kpoints)
        # Build the ExchangeOperator Vxk acting on orbitals at k
        Vxk = ExchangeOperator(basis, kpt, interaction_kernels, q_points,
                               ψ_occ_real, occ_occ)

        # ACE compression using as sketch space all orbitals the user wants for compression
        ψk_sketch = @view ψ[ik][:, mask_sketch[ik]]
        (; opk, Mk) = compress_exchange(Vxk, ψk_sketch, ψ_sketch_real[ik])
        push!(ops, opk)

        # Energy computation
        #
        # The compression computes Wnk = Vxk * ψnk  for all passed ψnk.
        # Therefore [Mk]_{nm} is just < ψ_{nk}, Vxk ψmk>, which means that the
        # energy contribution from this k-point can be computed as
        #     1/2 * ∑_n occ_{nk} [Mk]_{nn}
        occk_sketch = @view occupation[ik][mask_sketch[ik]]
        Ek = 1/T(2) * real(tr(Diagonal(Mk) * Diagonal(occk_sketch)))
        E += Ek * basis.kweights[ik]
    end
    (; E, ops)
end

# Sketch exchange operator of a particular k-point using the passed orbitals
# in real and Fourier space
function compress_exchange(Vxk::ExchangeOperator, ψk::AbstractMatrix,
                           ψk_real::AbstractArray{T,4}) where {T}
    basis = Vxk.basis
    kpt = Vxk.kpoint

    # Compute Wnk = Vxk * ψnk in real space, threaded over the sketch orbitals
    Wk = similar(ψk)
    allocate_local_storage() = (; Wnk_real=similar(ψk_real, size(ψk_real)[1:3]...))
    parallel_loop_over_range(1:size(ψk, 2); allocate_local_storage) do n, storage
        storage.Wnk_real .= 0
        apply!((; real=storage.Wnk_real), Vxk, (; real=@view ψk_real[:, :, :, n]))
        fft!(@view(Wk[:, n]), basis, kpt, storage.Wnk_real)
    end

    # The compressed operator is Wk Mk⁺ Wk'. Mk is Hermitian, but depending on the kernel
    # possibly indefinite and nearly singular, so we use its pseudo-inverse. For a Hermitian
    # matrix pinv works via an eigendecomposition and drops eigenvalues below its default
    # tolerance, size(Mk, 1) * eps relative to the largest one.
    Mk = ψk' * Wk
    Mk_pinv = pinv(Hermitian(Mk))
    opk = NonlocalOperator(basis, kpt, Wk, Mk_pinv)
    (; opk, Mk, Mk_pinv, Wk)
end
