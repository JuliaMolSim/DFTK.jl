using SparseArrays

"""
Compute the index mapping between two bases. Returns two arrays
`idcs_in` and `idcs_out` such that `ψkout[idcs_out] = ψkin[idcs_in]` does
the transfer from `ψkin` (defined on `basis_in` and `kpt_in`) to `ψkout`
(defined on `basis_out` and `kpt_out`).
"""
function transfer_mapping(basis_in::PlaneWaveBasis{T},  kpt_in::Kpoint,
                          basis_out::PlaneWaveBasis{T}, kpt_out::Kpoint) where {T}
    idcs_in  = 1:length(G_vectors(basis_in, kpt_in))  # All entries from idcs_in
    kpt_in == kpt_out && return idcs_in, idcs_in

    # Get indices of the G vectors of the old basis inside the new basis.
    idcs_out = index_G_vectors.(Ref(basis_out), G_vectors(basis_in, kpt_in))

    # In the case where G_vectors(basis_in.kpoints[ik]) are bigger than vectors
    # in the fft_size box of basis_out, we need to filter out the "nothings" to
    # make sure that the index linearization works. It is not an issue to
    # filter these vectors as this can only happen if Ecut_in > Ecut_out.
    if any(isnothing, idcs_out)
        idcs_in  = idcs_in[idcs_out .!= nothing]
        idcs_out = idcs_out[idcs_out .!= nothing]
    end
    idcs_out = getindex.(Ref(LinearIndices(basis_out.fft_size)), idcs_out)

    # Map to the indices of the corresponding G-vectors in
    # G_vectors(basis_out, kpt_out) this array might contains some nothings if
    # basis_out has less G_vectors than basis_in at this k-point
    idcs_out = indexin(idcs_out, kpt_out.mapping)
    if any(isnothing, idcs_out)
        idcs_in  = idcs_in[idcs_out .!= nothing]
        idcs_out = idcs_out[idcs_out .!= nothing]
    end

    idcs_in, idcs_out
end

"""
Return a sparse matrix that maps quantities given on `basis_in` and `kpt_in`
to quantities on `basis_out` and `kpt_out`.
"""
function compute_transfer_matrix(basis_in::PlaneWaveBasis{T}, kpt_in::Kpoint,
                                 basis_out::PlaneWaveBasis{T}, kpt_out::Kpoint) where {T}
    idcs_in, idcs_out = transfer_mapping(basis_in, kpt_in, basis_out, kpt_out)
    sparse(idcs_out, idcs_in, true)
end

"""
Return a list of sparse matrices (one per ``k``-point) that map quantities given in the
`basis_in` basis to quantities given in the `basis_out` basis.
"""
function compute_transfer_matrix(basis_in::PlaneWaveBasis{T}, basis_out::PlaneWaveBasis{T}) where {T}
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))
    [compute_transfer_matrix(basis_in, kpt_in, basis_out, kpt_out)
     for (kpt_in, kpt_out) in zip(basis_in.kpoints, basis_out.kpoints)]
end

"""
Transfer an array `ψk` defined on basis_in ``k``-point kpt_in to basis_out ``k``-point kpt_out.
"""
function transfer_blochwave_kpt(ψk_in, basis_in::PlaneWaveBasis{T}, kpt_in::Kpoint,
                                basis_out::PlaneWaveBasis{T}, kpt_out::Kpoint) where {T}
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
Transfer an array `ψk_in` expanded on `kpt_in`, and produce ``ψ(r) e^{i ΔG·r}`` expanded on
`kpt_out`. It is mostly useful for phonons.
Beware: `ψk_out` can lose information if the shift `ΔG` is large or if the `G_vectors`
differ between `k`-points.
"""
function transfer_blochwave_kpt(ψk_in, basis::PlaneWaveBasis, kpt_in, kpt_out, ΔG)
    ψk_out = zeros(eltype(ψk_in), length(G_vectors(basis, kpt_out)), size(ψk_in, 2))
    for (iG, G) in enumerate(G_vectors(basis, kpt_in))
        # e^i(kpt_in + G)r = e^i(kpt_out + G')r, where
        # kpt_out + G' = kpt_in + G = kpt_out + ΔG + G, and
        # G' = G + ΔG
        idx_Gp_in_kpoint = index_G_vectors(basis, kpt_out, G - ΔG)
        if !isnothing(idx_Gp_in_kpoint)
            ψk_out[idx_Gp_in_kpoint, :] = ψk_in[iG, :]
        end
    end
    ψk_out
end

"""
Find the equivalent index of the coordinate `kcoord` ∈ ℝ³ in a list `kcoords` ∈ [-½, ½)³.
`ΔG` is the vector of ℤ³ such that `kcoords[index] = kcoord + ΔG`.
"""
function find_equivalent_kpt(kcoords::Vector{Vec3{T}}, kcoord; tol=100*eps(T)) where {T}
    kcoord_normalized = mod.(Vector(kcoord), 1)  # coordinate in [0, 1)³
    kcoord_normalized[kcoord_normalized .≥ 0.5  - tol] .-= 1  # coordinate in [-½, ½)

    indices = findall(e -> is_approx_integer(e - kcoord_normalized; tol), kcoords)
    index = only(indices)

    ΔG = kcoord_normalized - kcoord

    # ΔG should be an integer.
    @assert all(is_approx_integer.(ΔG; tol))
    ΔG = round.(Int, ΔG)

    return (; index=index, ΔG)
end

"""
Return the Fourier coefficients for `ψk · e^{i q·r}` in the basis of `kpt_out`, where `ψk`
is defined on a basis `kpt_in`.
"""
function multiply_by_expiqr(basis, kpt_in, q, ψk; transfer_fn=transfer_blochwave_kpt)
    kcoords = getfield.(basis.kpoints, :coordinate)
    shifted_kcoord = kpt_in.coordinate .+ q  # coordinate of ``k``-point in ℝ
    (index, ΔG) = find_equivalent_kpt(kcoords, shifted_kcoord)
    kpt_out = basis.kpoints[index]
    return transfer_fn(ψk, basis, kpt_in, kpt_out, ΔG)
end

"""
Transfer Bloch wave between two basis sets. Limited feature set.
"""
function transfer_blochwave(ψ_in, basis_in::PlaneWaveBasis{T},
                            basis_out::PlaneWaveBasis{T}) where {T}
    @assert basis_in.model.lattice == basis_out.model.lattice
    @assert length(basis_in.kpoints) == length(basis_out.kpoints)
    @assert all(basis_in.kpoints[ik].coordinate == basis_out.kpoints[ik].coordinate
                for ik in 1:length(basis_in.kpoints))

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
