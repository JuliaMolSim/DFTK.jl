using SparseArrays

"""
Compute the index mapping between the global grids of two bases.
Returns an iterator of 8 pairs `(block_in, block_out)`. Iterated over these pairs
`x_out_fourier[block_out, :] = x_in_fourier[block_in, :]` does the transfer from
the Fourier coefficients `x_in_fourier` (defined on `basis_in`) to
`x_out_fourier` (defined on `basis_out`, equally provided as Fourier coefficients).
"""
function transfer_mapping(basis_in::PlaneWaveBasis, basis_out::PlaneWaveBasis)
    @assert basis_in.model.lattice == basis_out.model.lattice

    # TODO This logic feels rather convoluted ... think if there are ways to simplify
    idcs = map(basis_in.fft_size, basis_out.fft_size) do fft_in, fft_out
        if fft_in <= fft_out
            a = cld(fft_in, 2)
            b = fld(fft_in, 2)
            return (; in=(1:a,         (a+1):fft_in ),
                     out=(1:a, (fft_out+1-b):fft_out))
        else
            a = cld(fft_out, 2)
            b = fld(fft_out, 2)
            return (; in=(1:a, (fft_in+1-b):fft_in ),
                     out=(1:a,        (a+1):fft_out))
        end
    end

    idcs_in  = CartesianIndices.(Iterators.product(idcs[1].in,  idcs[2].in,  idcs[3].in))
    idcs_out = CartesianIndices.(Iterators.product(idcs[1].out, idcs[2].out, idcs[3].out))
    zip(idcs_in, idcs_out)
end

"""
Compute the index mapping between two bases. Returns two arrays
`idcs_in` and `idcs_out` such that `ψk_out[idcs_out] = ψk_in[idcs_in]` does
the transfer from `ψk_in` (defined on `basis_in` and `kpt_in`) to `ψk_out`
(defined on `basis_out` and `kpt_out`).

Note that `kpt_out` does not have to belong to `basis_out` as long as it is equivalent to
some other point in it (`kpt_out = kpt_in + ΔG`).
In that case, the transfer does not change the Bloch wave ``ψ``.
It does change the periodic part ``u``:
``e^{i k·x} u_k(x) = e^{i (k+ΔG)·x} (e^{-i ΔG·x} u_k(x))``.
Beware: this is a lossy conversion in general.
"""
function transfer_mapping(basis_in::PlaneWaveBasis,  kpt_in::Kpoint,
                          basis_out::PlaneWaveBasis, kpt_out::Kpoint)
    @assert basis_in.model.lattice == basis_out.model.lattice
    ΔG = kpt_out.coordinate .- kpt_in.coordinate  # kpt_out = kpt_in + ΔG
    @assert all(is_approx_integer.(ΔG))
    ΔG = round.(Int, ΔG)

    idcs_in  = 1:length(G_vectors(basis_in, kpt_in))  # All entries from idcs_in
    kpt_in == kpt_out && return idcs_in, idcs_in

    # Get indices of the G vectors of the old basis inside the new basis.
    idcs_out = index_G_vectors.(Ref(basis_out), G_vectors(basis_in, kpt_in) .- Ref(ΔG))

    # In the case where G_vectors(basis_in.kpoints[ik]) are bigger than vectors
    # in the fft_size box of basis_out, we need to filter out the "nothings" to
    # make sure that the index linearization works. It is not an issue to
    # filter these vectors as this can only happen if Ecut_in > Ecut_out.
    if any(isnothing, idcs_out)
        idcs_in  = idcs_in[idcs_out  .!= nothing]
        idcs_out = idcs_out[idcs_out .!= nothing]
    end
    idcs_out = getindex.(Ref(LinearIndices(basis_out.fft_size)), idcs_out)

    # Map to the indices of the corresponding G-vectors in
    # G_vectors(basis_out, kpt_out) this array might contains some nothings if
    # basis_out has less G_vectors than basis_in at this k-point
    idcs_out = indexin(idcs_out, kpt_out.mapping)
    if any(isnothing, idcs_out)
        idcs_in  = idcs_in[idcs_out  .!= nothing]
        idcs_out = idcs_out[idcs_out .!= nothing]
    end

    idcs_in, idcs_out
end

"""
Return a sparse matrix that maps quantities given on `basis_in` and `kpt_in`
to quantities on `basis_out` and `kpt_out`.
"""
function compute_transfer_matrix(basis_in::PlaneWaveBasis,  kpt_in::Kpoint,
                                 basis_out::PlaneWaveBasis, kpt_out::Kpoint)
    idcs_in, idcs_out = transfer_mapping(basis_in, kpt_in, basis_out, kpt_out)
    sparse(idcs_out, idcs_in, true)
end

"""
Return a list of sparse matrices (one per ``k``-point) that map quantities given in the
`basis_in` basis to quantities given in the `basis_out` basis.
"""
function compute_transfer_matrix(basis_in::PlaneWaveBasis, basis_out::PlaneWaveBasis)
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik = 1:length(basis_in.kpoints))
    [compute_transfer_matrix(basis_in, kpt_in, basis_out, kpt_out)
     for (kpt_in, kpt_out) in zip(basis_in.kpoints, basis_out.kpoints)]
end

"""
Transfer an array `ψk` defined on basis_in ``k``-point kpt_in to basis_out ``k``-point
`kpt_out`; see [`transfer_mapping`](@ref).
"""
function transfer_blochwave_kpt(ψk_in, basis_in::PlaneWaveBasis, kpt_in::Kpoint,
                                basis_out::PlaneWaveBasis, kpt_out::Kpoint)
    kpt_in == kpt_out && return copy(ψk_in)
    @assert length(G_vectors(basis_in, kpt_in)) == size(ψk_in, 1)
    idcsk_in, idcsk_out = transfer_mapping(basis_in, kpt_in, basis_out, kpt_out)

    n_bands = size(ψk_in, 2)
    ψk_out  = similar(ψk_in, length(G_vectors(basis_out, kpt_out)), n_bands)
    ψk_out .= 0
    ψk_out[idcsk_out, :] .= ψk_in[idcsk_in, :]

    ψk_out
end

"""
Transfer Bloch wave between two basis sets. Limited feature set.
"""
function transfer_blochwave(ψ_in, basis_in::PlaneWaveBasis, basis_out::PlaneWaveBasis)
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik = 1:length(basis_in.kpoints))

    # If, for some kpt ik, basis_in has less vectors than basis_out, then idcs_out[ik] is
    # the array of the indices of the G_vectors from basis_in in basis_out.
    # It is then of size G_vectors(basis_in.kpoints[ik]) and the transfer can be done with
    # ψ_out[ik] .= 0
    # ψ_out[ik][idcs_out[ik], :] .= ψ_in[ik]

    # Otherwise, if, for some kpt ik, basis_in has more vectors than basis_out, then
    # idcs_out[ik] just keep the indices of the G_vectors from basis_in that are in basis_out.
    # It is then of size G_vectors(basis_out.kpoints[ik]) and the transfer can be done with
    # ψ_out[ik] .= ψ_in[ik][idcs_in[ik], :]

    map(enumerate(basis_out.kpoints)) do (ik, kpt_out)
        transfer_blochwave_kpt(ψ_in[ik], basis_in, basis_in.kpoints[ik], basis_out, kpt_out)
    end
end

@doc raw"""
Transfer density (in real space) between two basis sets.

This function is fast by transferring only the Fourier coefficients from the small basis
to the big basis.

Note that this implies that for even-sized small FFT grids doing the
transfer small -> big -> small is not an identity (as the small basis has an unmatched
Fourier component and the identity ``c_G = c_{-G}^\ast`` does not fully hold).

Note further that for the direction big -> small employing this function does not give
the same answer as using first `transfer_blochwave` and then `compute_density`.
"""
function transfer_density(ρ_in, basis_in::PlaneWaveBasis{T},
                          basis_out::PlaneWaveBasis{T}) where {T}
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(size(ρ_in)) ∈ (3, 4)

    ρ_freq_in  = fft(basis_in, ρ_in)
    ρ_freq_out = zeros_like(ρ_freq_in, basis_out.fft_size..., size(ρ_in, 4))

    for (block_in, block_out) in transfer_mapping(basis_in, basis_out)
        ρ_freq_out[block_out, :] .= ρ_freq_in[block_in, :]
    end

    irfft(basis_out, ρ_freq_out)
end

"""
Find the equivalent index of the coordinate `kcoord` ∈ ℝ³ in a list `kcoords` ∈ [-½, ½)³.
`ΔG` is the vector of ℤ³ such that `kcoords[index] = kcoord + ΔG`.
"""
function find_equivalent_kpt(basis::PlaneWaveBasis{T}, kcoord, spin; tol=sqrt(eps(T))) where {T}
    kcoord_red = normalize_kpoint_coordinate(kcoord .+ eps(T))

    ΔG = kcoord_red - kcoord
    # ΔG should be an integer.
    @assert all(is_approx_integer.(ΔG))
    ΔG = round.(Int, ΔG)

    indices_σ = krange_spin(basis, spin)
    kcoords_σ = getfield.(basis.kpoints[indices_σ], :coordinate)
    # Unique by construction.
    index::Int = findfirst(isapprox(kcoord_red; atol=tol), kcoords_σ) + (indices_σ[1] - 1)

    return (; index, ΔG)
end

"""
Returns a permutation `indices` of the ``k``-points in `basis` such that
`kpoints[ik].coordinate + q` is equivalent to `kpoints[indices[ik]].coordinate`.
"""
function k_to_kpq_permutation(basis::PlaneWaveBasis, q)
    kpoints = basis.kpoints
    indices = [find_equivalent_kpt(basis, kpt.coordinate + q, kpt.spin).index
               for kpt in kpoints]
    @assert isperm(indices)
    indices
end

@doc raw"""
Return the Fourier coefficients for the Bloch waves ``f^{\rm real}_{q} ψ_{k-q}`` in an
element of `basis.kpoints` equivalent to ``k-q``.
"""
@views function multiply_ψ_by_blochwave(basis::PlaneWaveBasis, ψ, f_real, q)
    fψ = zero.(ψ)
    # First, express ψ_{[k-q]} in the basis of k-q points…
    ψ_minus_q = transfer_blochwave_equivalent_to_actual(basis, ψ, -q)
    for (ik, kpt) in enumerate(basis.kpoints)
        # … then perform the multiplication with f in real space and get the Fourier
        # coefficients.
        for n = 1:size(ψ[ik], 2)
            fψ[ik][:, n] = fft(basis, kpt,
                               ifft(basis, ψ_minus_q[ik].kpt, ψ_minus_q[ik].ψk[:, n])
                                 .* f_real[:, :, :, kpt.spin])
        end
    end
    fψ
end

"""
For Bloch waves ``ψ`` such that `ψ[ik]` is defined in a point in `basis.kpoints` equivalent
to `basis.kpoints[ik] + q`, return the Bloch waves `ψ_plus_q[ik]` defined on `kpt_plus_q[ik]`.
"""
@views function transfer_blochwave_equivalent_to_actual(basis, ψ_plus_q_equivalent, q)
    k_to_k_plus_q = k_to_kpq_permutation(basis, q)
    map(enumerate(basis.kpoints)) do (ik, kpt)
        # Express ψ_plus_q_equivalent_{[k-q]} in the basis of k-q points.
        kpt_plus_q, equivalent_kpt_plus_q = get_kpoint(basis, kpt.coordinate + q, kpt.spin)
        ψk_plus_q = transfer_blochwave_kpt(ψ_plus_q_equivalent[k_to_k_plus_q[ik]], basis,
                                           equivalent_kpt_plus_q, basis, kpt_plus_q)
        (; kpt=kpt_plus_q, ψk=ψk_plus_q)
    end
end
