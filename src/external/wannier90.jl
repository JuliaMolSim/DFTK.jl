using Dates
using Printf: @sprintf

"""
Return the indices of the bands to keep for Wannierization.

Usually the semicore states are excluded from the Wannierization.

# Arguments
- `n_bands`: the total number of bands computed in the DFT calculation.
- `exclude_bands`: the band indices to be excluded.
"""
function _get_keep_bands(n_bands::Integer, exclude_bands::Union{AbstractVector{Int},Nothing})
    keep_bands = 1:n_bands
    if !isnothing(exclude_bands)
        keep_bands = setdiff(keep_bands, exclude_bands)
    end
    keep_bands
end

"""
See [`compute_mmn`](@ref) for the meaning of the arguments.
"""
function _get_keep_bands(
    ψ::AbstractVector{Matrix},
    exclude_bands::Union{AbstractVector{Int},Nothing}
)
    n_bands = size(ψ[1], 2)
    _get_keep_bands(n_bands, exclude_bands)
end

"""
Return the kpoint index shift for spin-polarized case.

For spin-polarized calculation, first half of kpoints are for spin up,
and the second half are for spin down.
"""
function _get_ik_shift(basis::PlaneWaveBasis, spin::Int)
    ik_shift = 0
    n_spin = basis.model.n_spin_components
    n_kpts = length(basis.kpoints) ÷ n_spin
    # only for spin down case, the kpoint index needs to be shifted
    spin == 2 && (ik_shift = n_kpts)
    ik_shift
end

@doc raw"""
Computes the overlap matrix for Wannierization.

``[M^{k,b}]_{m,n} = \langle u_{m,k} | u_{n,k+b} \rangle``
for given kpoint ``k`` and ``b`` vector.

The ``k+b`` vector is actually shifted to its periodic image, denoted by
another kpoint ``k`` inside the uniform kgrid, together with a shifting
vector ``G_shift``.
`G_shift` is the "shifting" vector, correction due to the periodicity conditions
imposed on ``k \to  ψ_k``.
It is non zero if ``k+b`` is taken in another unit cell of the reciprocal lattice.
We use here that:
``u_{n(k + G_{\rm shift})}(r) = e^{-i G_{\rm shift} \cdot r} u_{nk}``.

# Arguments
- `basis`: the `PlaneWaveBasis`
- `ψ`: the Bloch states, a vector of length `n_kpts`, each element is
    a matrixof size `(n_G, n_bands)`. In the spin-polarized case, the
    vector is of length `2*n_kpts`, and the first half is for spin-up,
    the second half is for spin-down.
- `kpb_k`: the kpoint indices of k+b, size: `(n_bvecs, n_kpts)`.
    1st index: b-vector, 2nd index: kpoint, e.g., `kpb_k[ib, ik]` is the
    corresponding kpoint index of k+b for `ib`-th b-vector at `ik`-th kpoint.
- `kpb_G`: the shifting vectors of k+b, size: `(3, n_bvecs, n_kpts)`.
    1st index: x,y,z component of b-vector,
    2nd index: b-vector,
    3rd index: shifting vector to get the shifted k-point `kpb_k[ib,ik]` to
    the actual k+b point, thus the actual k+b is `kpb_k[ib,ik] + kpb_G[:,ib,ik]`.

# Keyword arguments
- `exclude_bands`: the bands to be excluded from the Wannierization.
- `spin`: the spin channel to be used, either `1` or `2`.
"""
@timing function compute_mmn(
    basis::PlaneWaveBasis,
    ψ::AbstractVector{Matrix{Complex}},
    kpb_k::AbstractArray{Int,2},
    kpb_G::AbstractArray{Int,3};
    exclude_bands::Union{AbstractVector{Int},Nothing},
    spin::Int=1,
)
    keep_bands = _get_keep_bands(ψ, exclude_bands)
    n_bands = length(keep_bands)

    n_bvecs, n_kpts = size(kpb_k)
    @assert size(kpb_G) == (3, n_bvecs, n_kpts)

    kpts = krange_spin(basis, spin)
    @assert length(kpts) == n_kpts
    ik_shift = _get_ik_shift(basis, spin)

    M = zeros(eltype(ψ[1]), n_bands, n_bands, n_bvecs, n_kpts)

    for ik in kpts
        k = basis.kpoints[ik]
        ik0 = ik - ik_shift
        for ib in 1:n_bvecs
            ikpb0 = kpb_k[ib, ik0]
            ikpb = ikpb0 + ik_shift
            kpb = basis.kpoints[ikpb]
            G_shift = kpb_G[:, ib, ik0]
            # Search for common Fourier modes and their resp. indices in
            # Bloch states k and kpb
            # TODO Check if this can be improved using the G vector mapping in the kpoints
            equivalent_G_vectors = [
                (iGk, index_G_vectors(basis, kpb, Gk + G_shift))
                for (iGk, Gk) in enumerate(G_vectors(basis, k))
            ]
            iGk = [eqG[1] for eqG in equivalent_G_vectors if !isnothing(eqG[2])]
            iGkpb = [eqG[2] for eqG in equivalent_G_vectors if !isnothing(eqG[2])]

            # Compute overlaps
            for (i, n) in enumerate(keep_bands)
                for (j, m) in enumerate(keep_bands)
                    # Select the coefficient in right order
                    M[j, i, ib, ik0] = dot(ψ[ik][iGk, m], ψ[ikpb][iGkpb, n])
                end
            end
        end
    end
    M
end

@doc raw"""
Radial functions from solutions of Hydrogenic Schrödinger equation.

Same as Wannier90 user guide Table 3.3.

# Arguments
- `r`: radial grid
- `n`: principal quantum number
- `α`: diffusivity, ``\frac{Z}/{a}`` where ``Z`` is the atomic number and
    ``a`` is the Bohr radius.
"""
function radial_hydrogenic(r::AbstractVector{T}, n::Integer, α::Real=1.0) where {T<:Real}
    @assert n > 0
    if n == 1
        f = 2 * α^(3/2) * exp(-α * r)
    elseif n == 2
        f = 2^(-3/2) * α^(3/2) * (2 - α * r) * exp(-α * r/2)
    elseif n == 3
        f = sqrt(4/27) * α^(3/2) * (1 - 2/3 * α * r + 2/27 * α^2 * r.^2) * exp(-α * r/3)
    else
        error("n = $n is not supported")
    end
    f
end

"""
Compute radial Fourier transform of a radial function.

`4π Σ{k} j_l(q r[k]) (r[k]^2 R_{l}(r[k]) dr[k])`
where `q = k + G`, `R_{l}` is the radial function, `j_l` is spherical Bessel
function and `l` is angular momentum, `r[k]` is the radial grid and
`k` is the grid index.
"""
function radial_fourier(
    R::AbstractVector{T}, r::AbstractVector{T}, dr::AbstractVector{T},
    q::AbstractVector{Vec3}, l::Integer) where {T<:Real}
    @assert length(R) == length(r)
    r2_R_dr = r.^2 .* R .* dr
    f = zeros(T, length(q))
    @inbounds for (iq, qvec) in enumerate(q)
        for ir = eachindex(r2_R_dr)
            f[iq] += sphericalbesselj_fast(l, qvec * r[ir]) * r2_R_dr[ir]
        end
    end
    4π * f
end

@raw doc"""
Compute Fourier transform of a projector function.

A projector function is ``R(r) * Y_{lm}(\theta, \phi)``.
The `Rq` argument is the radial Fourier transform of `R(r)`, computed by
[`radial_fourier`](@ref).

`(-i)^l R(q) Y_{lm}(q)`
"""
function projector_fourier(Rq::AbstractVector{T},
    q::AbstractVector{Vec3}, l::Integer, m::Integer) where {T<:Real}
    ylm = zeros(T, length(q))
    @inbounds for (iq, qvec) in enumerate(q)
        ylm[iq] = ylm_real(l, m, qvec)
    end
    (-im)^l .* Rq .* ylm
end

@doc raw"""
Given an orbital ``f``, the periodized orbital is defined by:
```math
\phi_{\mathbf{k}} = \sum \limits_{R \in {\rm lattice}}
    \exp(- \mathbf{G} \cdot \mathbf{R}) f(\mathbf{k + G}).
```
"""
@timing function bloch_sum(kpt::Kpoint, fq::AbstractArray{T}, 
    τ::Vec3{Real}) where {T<:Complex}
    ϕk = zeros(T, length(kpt.G_vectors))

    @inbounds for (iG, G) in enumerate(kpt.G_vectors)
        ϕk[iG] += exp(-im * dot(kpt.coordinate + G, τ)) * fq[iG]
    end
    # Functions are l^2 normalized in Fourier, in DFTK conventions.
    ϕk / norm(ϕk)
end

@doc raw"""
Compute random Gaussain initial guess for Wannier functions.

With specified WF centers given as a vector of length `n_wann`,
each element is a `Vec3` which is in lattice coordinates.
"""
@timing function guess_wannier_hydrogenic(
    kpt::Kpoint;
    centers::AbstractVector{Vec3},
    l::AbstractVector{Integer}, m::AbstractVector{Integer},
    n::AbstractVector{Integer}=ones(Integer, length(centers)),
    α::AbstractVector{Real}=ones(Real, length(centers))
)
    n_wann = length(centers)
    @assert n_wann == length(l) == length(m) == length(n) == length(α)
    @assert all(n .> 0)
    @assert all(α .> 0)
    @assert all(0 .<= l .<= n .- 1)
    @assert all(-l .<= m .<= l)

    r = 1:0.1:10  # TODO update
    dr = 0.1 .* ones(length(r))
    q = map(G -> G + kpt.coordinate, kpt.G_vectors)
    # all q = k+G in reduced coordinates
    # qs = vec(map(G -> G .+ kpt.coordinate, G_vectors(basis)))

    for (i, (ni, li, mi, αi)) in enumerate(zip(n, l, m, α))
        R = radial_hydrogenic(r, ni, αi)
        Rq = radial_fourier(R, r, dr, q, li)
        fq = projector_fourier(Rq, q, li, mi)
        ϕk = bloch_sum(kpt, fq, centers[i])
    end
    ϕk
end

@doc raw"""
Compute the initial projection matrix for Wannierization.

``A_{m,n,k} = \langle ψ_{m,k} | g_{n,k} \rangle``
where ``g_{n,k}`` are Bloch sums of some real-space localized orbitals.

# Keyword arguments
- `guess`: the initial guess for the Wannier functions.
    A function that accepts a `Kpoint` `kpt` as arguments,
    and return a vector of length `n_wann` containing the initial guess for
    the MLWFs at that kpoint, see [`compute_amn_random`](@ref) for examples.
"""
@timing function compute_amn(
    basis::PlaneWaveBasis,
    ψ::AbstractVector{Matrix{Complex}},
    guess::Function;
    exclude_bands::Union{AbstractVector{Int},Nothing},
    spin::Integer=1,
)
    kpts = krange_spin(basis, spin)
    ψs = ψ[kpts]  # ψ for the selected spin

    n_kpts = length(kpts)
    keep_bands = _get_keep_bands(ψs, exclude_bands)
    n_bands = length(keep_bands)
    # I call once `guess` to get the n_wann, to avoid having to pass `n_wann`
    # as an argument.
    ϕk = guess(basis.kpoints[kpts[1]])
    n_wann = length(ϕk)
    A = zeros(eltype(ψs[1]), n_bands, n_wann, n_kpts)

    # G_vectors in reduced coordinates.
    # The dot product is computed in the Fourier space.
    for (ik, kpt) in enumerate(basis.kpoints[kpts])
        ψk = ψs[ik]
        ik != 1 && (ϕk = guess(kpt))
        length(ϕk) == n_wann || error("guess function returns wrong length")
        for m in keep_bands
            for (n, ϕnk) in enumerate(ϕk)
                A[m, n, ik] = dot(ψk[:, m], ϕnk)
            end
        end
    end
    A
end

"""
Random Gaussain initial guess for Wannier functions.
"""
@timing function compute_amn_random(
    basis::PlaneWaveBasis,
    ψ::AbstractVector{Matrix{Complex}},
    n_wann::Integer;
    kwargs...
)
    @assert n_wann > 0

    random_centers = [rand(3) for _ in 1:n_wann]
    l = [0 for _ in 1:n_wann]
    m = [0 for _ in 1:n_wann]
    guess(kpt) = guess_wannier_hydrogenic(kpt; random_centers, l, m)

    compute_amn(basis, ψ, guess; kwargs...)
end

"""
Return a matrix of eigenvalues on a uniform grid for Wannierization.

# Arguments
- `basis`: the `PlaneWaveBasis`
- `eigenvalues`: the eigenvalues of the Hamiltonian, a vector of length
    `n_kpts`, each element is a vector of length `n_bands`. In the spin-polarized
    case, the vector is of length `2*n_kpts`, and the first half is for spin-up,
    the second half is for spin-down.

For the explanation of keyword arguments, see [`compute_mmn`](@ref).
"""
function compute_eig(
    basis::PlaneWaveBasis,
    eigenvalues::AbstractVector{Vector{Real}};
    exclude_bands::Union{AbstractVector{Int},Nothing}=nothing,
    spin::Int=1,
)
    kpts = krange_spin(basis, spin)
    eigs = eigenvalues[kpts]

    n_kpts = length(kpts)
    n_bands = length(eigs[1])
    keep_bands = _get_keep_bands(n_bands, exclude_bands)
    n_bands = length(keep_bands)
    E = zeros(eltype(eigs[1]), n_bands, n_kpts)

    for (ik, εk) in enumerate(eigs)
        for (n, εnk) in enumerate(εk[keep_bands])
            E[n, ik] = auconvert(Unitful.eV, εnk).val
        end
    end
    E
end

"""
Compute real-space wavefunctions that can be used for plotting Wannier functions.

Since real-space wavefunctions are memory intensive, a `Kpoint` `kpt` needs
to be specified to compute the periodic part of the wavefunction at a given
kpoint.
Since the `Kpoint.spin` already contains the spin information, no need
to specify `spin` here, as in `compute_mmn`, `compute_eig`, etc.

For the explanation of the `ψ` and keyword arguments, see `compute_mmn`.
"""
@timing function compute_unk(
    basis::PlaneWaveBasis,
    ψ::AbstractVector{Matrix{Complex}},
    kpt::Kpoint;
    exclude_bands::Union{AbstractVector{Int},Nothing},
)
    fft_size = basis.fft_size
    keep_bands = _get_keep_bands(ψ, exclude_bands)
    n_bands = length(keep_bands)
    unk = zeros(eltype(ψ[1]), fft_size[1], fft_size[2], fft_size[3], n_bands)

    ik = findfirst(basis.kpoints .== kpt)
    for ib in 1:n_bands
        unk[:, :, :, ib] = ifft(basis, kpt, @view ψ[ik][:, ib])
    end
    unk
end

"""
Generate a `NamedTuple` of input parameters for a Wannier90 `win` file.
Parameters to Wannier90 can be added as kwargs: e.g. `num_iter=500`.
"""
function get_wannier90_win(basis::PlaneWaveBasis; kwargs...)
    isnothing(basis.kgrid) && error("The basis must be constructed from a MP grid.")
    # Required parameter
    @assert :num_wann in keys(kwargs)
    @assert :num_bands in keys(kwargs)

    win = Dict{Symbol, Any}(kwargs)

    # columnwise vectors, and to angstrom
    unit_cell_cart = auconvert(Unitful.angstrom, basis.model.lattice).val
    push!(win, :unit_cell_cart => unit_cell_cart)

    atom_labels = [a.symbol for a in basis.model.atoms]
    # to columnwise vectors, and fractional coordinates
    atoms_frac = hcat(basis.model.positions...)
    push!(win, :atoms_frac => atoms_frac, :atom_labels => atom_labels)

    # deepcopy for safety
    mp_grid = deepcopy(basis.kgrid)
    if basis.model.spin_polarization == :collinear
        kpoints = filter(k -> k.spin == 1, basis.kpoints)
    else
        kpoints = basis.kpoints[1:2:end] #?
    end
    # columnwise vectors, fractional coordinates
    kpoints = hcat([k.coordinate for k in kpoints]...)
    @assert size(kpoints) == (3, prod(mp_grid))
    push!(win, :mp_grid => mp_grid, :kpoints => kpoints)

    if get(kwargs, :wannier_plot, false)
        push!(win, :wvfn_formatted => true)
    end

    if get(kwargs, :bands_plot, false)
        kpath  = irrfbz_path(basis.model)
        length(kpath.paths) > 1 || @warn(  # TODO check
            "Only first kpath branch considered in write_wannier90_win")
        path = kpath.paths[1]

        kpoint_path = []
        for i in 1:length(path)-1
            A, B = path[i:i+1]  # write segment A -> B
            A_k = A => kpath.points[A]
            B_k = B => kpath.points[B]
            push!(kpoint_path, [A_k, B_k])
        end
        push!(win, :kpoint_path => kpoint_path)
    end

    win
end

function _unfold_scfres(
    scfres::NamedTuple,
    exclude_bands::Union{AbstractArray{Integer},Nothing}=nothing,
)
    @assert scfres.basis.model.spin_polarization in (:none, :spinless, :collinear)

    # Undo symmetry operations to get full k-point list
    # for collinear calculation, I assume there is no symmetry
    if scfres.basis.model.spin_polarization == :collinear
        @assert scfres.basis.symmetry == :none  # TODO check
        scfres_unfold = scfres
    else
        scfres_unfold = unfold_bz(scfres)
    end
    basis = scfres_unfold.basis
    ψ = scfres_unfold.ψ
    eigenvalues = scfres_unfold.eigenvalues

    if !isnothing(exclude_bands)
        n_bands = length(eigenvalues[1])
        keep_bands = _get_keep_bands(n_bands, exclude_bands)
        ψ = [ψk[:, keep_bands] for ψk in ψ]
        eigenvalues = [e[keep_bands] for e in eigenvalues]
    end

    (; basis, ψ, eigenvalues)
end

"""
Compute matrices for Wannierization.

# Keyword arguments
- `nnkp`: a tuple of `(kpb_k, kpb_G)` specifying the ``b``-vectors.
    Can be generated either by `Wannier90` or `Wannier.jl`.
- `fileprefix`: the filename for saving the matrices, e.g.,
    `amn` will be written to `wannier/silicon.amn` if
    `fileprefix = "wannier/silicon"`. If `nothing`, no files are saved.
- `unk`: whether to compute real-space wavefunction files `UNK`.
    Warning: this takes a lot of memory and time.
"""
function compute_wannier(
    basis::PlaneWaveBasis,
    ψ::AbstractVector{Matrix{Complex}},
    eigenvalues::AbstractVector{Vector{T}};
    n_wann::Integer,
    nnkp::NamedTuple,
    spin::Integer=1,
    fileprefix::Union{Nothing,AbstractString}=nothing,
    unk::Bool=false,
    kwargs...,
) where {T<:Real}
    unk && isnothing(fileprefix) && error("Must specify `fileprefix` when `unk=true`")

    # Make wannier directory ...
    if !isnothing(fileprefix)
        dir, prefix = dirname(fileprefix), basename(fileprefix)
        mkpath(dir)
    end

    n_bands = length(eigenvalues[1])
    win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands, kwargs...)
    !isnothing(fileprefix) && WannierIO.write_win("$(prefix).win"; win...)

    M = compute_mmn(basis, ψ, nnkp.kpb_k, nnkp.kpb_G; spin=spin)
    if !isnothing(fileprefix)
        fname = "$(prefix).mmn"
        header = "Generated by DFTK.jl at $(now())"
        WannierIO.write_mmn(fname, M, nnkp.kpb_k, nnkp.kpb_G, header)
    end

    # This is just a demonstration using random initial projections,
    # in practice one should use a better guess.
    A = compute_amn_random(basis, ψ, n_wann; spin=spin)
    if !isnothing(fileprefix)
        fname = "$(prefix).amn"
        header = "Generated by DFTK.jl at $(now())"
        WannierIO.write_amn(fname, A)
    end

    E = compute_eig(basis, eigenvalues; spin=spin)
    if !isnothing(fileprefix)
        fname = "$(prefix).eig"
        WannierIO.write_eig(fname, E)
    end

    # Writing the unk files is expensive (requires FFTs), so only do if needed.
    if unk
        kpts = krange_spin(basis, spin)
        for (ik, kpt) in enumerate(basis.kpoints[kpts])
            unk = compute_unk(basis, ψ, kpt)
            fname = joinpath(dir, @sprintf "UNK%05d.%1d" ik spin)
            WannierIO.write_w90_unk(fname, ik, unk)
        end
    end

    (; win, M, A, E)
end

function _run_wannier90_jll(fileprefix::AbstractString, postproc::Bool=false)
    dir, prefix = dirname(fileprefix), basename(fileprefix)
    pp = postproc ? "-pp" : ""
    wannier90_jll.wannier90(exe -> run(Cmd(`$exe $pp $prefix`; dir)))
end

function _default_exclude_bands(scfres::NamedTuple)
    n_bands = length(scfres.eigenvalues[1])
    exclude_bands = (scfres.n_bands_converge+1):n_bands
    exclude_bands
end

"""
Call Wannier90 to Wannierize the results.

For `kwargs` see [`compute_wannier`](@ref).
"""
function run_wannier90(
    scfres::NamedTuple;
    fileprefix::AbstractString,
    exclude_bands::Union{AbstractArray{Integer},Nothing}=_default_exclude_bands(scfres),
    kwargs...,
)
    # Make wannier directory ...
    dir, prefix = dirname(fileprefix), basename(fileprefix)
    mkpath(dir)

    basis, ψ, eigenvalues = _unfold_scfres(scfres, exclude_bands)

    # Files for main Wannierization run
    if basis.model.spin_polarization == :collinear
        prefixes = ["$(fileprefix)_up", "$(fileprefix)_dn"]
    else
        prefixes = [fileprefix]
    end

    for (spin, prefix) in enumerate(prefixes)
        if isnothing(nnkp)
            _run_wannier90_jll(prefix, true)
            nnkp = WannierIO.read_nnkp("$(prefix).nnkp")
        end

        @timing "Compute Wannier matrices" compute_wannier(
            basis, ψ, eigenvalues; n_wann, nnkp, spin, fileprefix=prefix, kwargs...
        )

        # Run Wannierisation procedure
        @timing "Wannierization" _run_wannier90_jll(prefix)
    end
    prefixes
end

"""
Call `Wannier.jl` to Wannierize the results.

For `kwargs` see [`compute_wannier`](@ref).
"""
function run_wannier(
    scfres::NamedTuple;
    exclude_bands::Union{AbstractArray{Integer},Nothing}=_default_exclude_bands(scfres),
    kwargs...,
)
    basis, ψ, eigenvalues = _unfold_scfres(scfres, exclude_bands)

    # Files for main Wannierization run
    fileprefix = get(kwargs, :fileprefix, nothing)
    if isnothing(fileprefix)
        prefixes = [nothing]
    else
        if basis.model.spin_polarization == :collinear
            prefixes = ["$(fileprefix)_up", "$(fileprefix)_dn"]
        else
            prefixes = [fileprefix]
        end
    end

    n_bands = length(eigenvalues[1])
    @assert haskey(kwargs, :n_wann) "Must specify `n_wann` in `kwargs`"
    n_wann = kwargs[:n_wann]
    win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands, kwargs...)
    nnkp = Wannier.get_bvectors(win.unit_cell_cart, win.kpoints)  # TODO check

    models = []
    for (spin, prefix) in enumerate(prefixes)
        @timing "Compute Wannier matrices"  win, M, A, E = compute_wannier(
            basis, ψ, eigenvalues; n_wann, nnkp, spin, fileprefix=prefix, kwargs...
        )

        # Run Wannierisation procedure
        @timing "Wannierization" begin
            model = Wannier.Model(win, M, A, E)  # TODO check
            model.U .= Wannier.disentangle(model)
            push!(models, model)
        end
    end
    models
end
