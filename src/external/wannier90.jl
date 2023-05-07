using Dates
using Printf: @sprintf

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
- `spin`: the spin channel to be used, either `1` or `2`.
"""
@timing function compute_mmn(
    basis::PlaneWaveBasis, ψ::AbstractVector{Matrix{Complex}},
    kpb_k::AbstractArray{Int,2}, kpb_G::AbstractArray{Int,3};
    spin::Int=1,
)
    n_bands = size(ψ[1], 2)
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
            for n in 1:n_bands
                for m in 1:n_bands
                    M[m, n, ib, ik0] = dot(ψ[ik][iGk, m], ψ[ikpb][iGkpb, n])
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
Generate a fake `PspUpf` for Hydrogenic orbitals.
"""
function radial_hydrogenic_upf(n::Integer, l::Integer, α::Real)
    # this is same as QE pw2wannier90, r = exp(x) / α
    xmin = -6.0
    dx = 0.025
    rmax = 10.0
    n_r = round(Int, (log(rmax) - xmin) / dx) + 1
    x = range(; start=xmin, length=n_r, step=dx)
    # rgrid depends on α
    r = exp.(x) ./ α
    dr = r .* dx
    R = radial_hydrogenic(r, n, α)

    T = Float64
    pswfcs = Vector{Vector{Vector{T}}}()
    pswfc_occs = Vector{Vector{T}}()
    for _ in 0:(l-1)
        push!(pswfcs, [T[]])
        push!(pswfc_occs, T[])
    end
    push!(pswfcs, [R])
    push!(pswfc_occs, T[])

    PspUpf{T,T}(
        0, l+1, r, dr, T[], [[T[]]], [Matrix{T}()], pswfcs, pswfc_occs,
        T[], T[], T, [T[]], 0, 0, T[], [[T[]]], T[], T[], "",
        "hydrogenic n=$(n) l=$(l) α=$(α)"
    )
end

@doc raw"""
Project Bloch wavefunctions to hydrogenic orbitals for the initial guess of
Wannier functions.

# Arguments
- `basis`: PlaneWaveBasis
- `centers`: centers of Wannier functions, in fractional coordinates
- `l`: angular momentum
- `n`: principal quantum number
- `α`: diffusivity, see [`radial_hydrogenic`](@ref)

# Examples
Project onto a ``s`` orbital centered at (0, 0, 0) and
three ``p`` (i.e., ``p_y, p_z, p_x``) orbitals centered at (0.5, 0.5, 0.5).
```julia
guess = guess_amn_hydrogenic(basis; [[0, 0, 0], [0.5, 0.5, 0.5]], [0, 1])
```

See <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics>
for the order of orbitals.
"""
@timing function guess_amn_hydrogenic(
    basis::PlaneWaveBasis; centers::AbstractVector{Vec3},
    l::AbstractVector{Integer},
    n::AbstractVector{Integer}=ones(Integer, length(centers)),
    α::AbstractVector{Real}=ones(Real, length(centers))
)
    @assert length(centers) == length(l) == length(n) == length(α)
    @assert all(n .> 0)
    @assert all(α .> 0)
    @assert all(0 .<= l .<= n .- 1)

    # I generate a fake `PspUpf` so as to reuse `build_projection_vectors_pswfcs`
    labels = collect(zip(n, l, α))
    psp_groups = [findall(Ref(lab) .== labels) for lab in Set(labels)]
    psps = [radial_hydrogenic_upf(labels[first(group)]...) for group in psp_groups]
    psp_positions = [centers[group] for group in psp_groups]

    # note that on output the order of initial projections are different from
    # the input order of (centers, l, n, α), due to reordering of psps
    guess(kpt) = build_projection_vectors_pswfcs(basis, kpt, psps, psp_positions)

    # we permute the order so that it matches the user input
    # note the outputs also contain `m` angular momentum
    perm = Int[]
    for group in psp_groups
        for g in group
            start = sum(2 * l[i] + 1 for i in 1:(g-1); init=0)
            append!(perm, start .+ (1:(2 * l[g] + 1)))
        end
    end
    guess_perm(kpt) = permute!(guess(kpt), perm)

    # Lowdin orthonormalization
    ortho_lowdin ∘ guess_perm
end

"""
Allow specifying the `m` angular momentum of hydrogenic orbitals.

See <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics>
for the correspondence between `m` and orbital symbols, e.g.,
- for p orbitals (l=1), `m=-1,0,1` map to ``p_y, p_z, p_x``
- for d orbitals (l=2), `m=-2,-1,0,1,2` map to ``d_{xy}, d_{yz}, d_{z^2}, d_{xz}, d_{x^2-y^2}``

# Examples
For two atoms centered at (0, 0, 0) and (0.5, 0.5, 0.5), the following code
Projects onto a ``s`` orbital centered at (0, 0, 0) and
``p_x, p_y`` orbitals centered at (0.5, 0.5, 0.5).
```julia
guess = guess_amn_hydrogenic(
    basis; [[0, 0, 0], [0.5, 0.5, 0.5]], [0, 1, 1], [0, 1, -1]
)
```
"""
@timing function guess_amn_hydrogenic(
    basis::PlaneWaveBasis; centers::AbstractVector{Vec3},
    l::AbstractVector{Integer}, m::AbstractVector{Integer},
    n::AbstractVector{Integer}=ones(Integer, length(centers)),
    α::AbstractVector{Real}=ones(Real, length(centers))
)
    @assert length(centers) == length(l) == length(m) == length(n) == length(α)
    @assert all(-l .<= m .<= l)

    # find the indices of specified `l, m, n, α` in the output of
    # `guess_amn_hydrogenic`, which computes all the `m`.
    # e.g., with input `l=[0, 1, 1], m=[0, 1, -1]`, the returned orbitals
    # of `guess_amn_hydrogenic` are
    # `(l=0, m=0), (l=1, m=-1), (l=1, m=0), (l=1, m=1)`, so the computed
    # `indices = [1, 4, 2]`.
    indices = Int[]
    lnα = []
    for (ll, mm, nn, αα) in zip(l, m, n, α)
        q = (ll, nn, αα)
        !(q in lnα) && push!(lnα, q)
        iq = findfirst(Ref(q) .== lnα)
        lq = lnα[iq][1]
        @info mm, lq
        im = findfirst(mm .== (-lq:lq))
        start = sum(2 * lnα[i][1] + 1 for i in 1:(iq-1); init=0)
        push!(indices, start + im)
    end

    l = [q[1] for q in lnα]
    n = [q[2] for q in lnα]
    α = [q[3] for q in lnα]
    guess = guess_amn_hydrogenic(basis; centers, l, n, α)
    guess_filtered(kpt) = guess(kpt)[indices]
    guess_filtered
end

"""
Random Gaussain initial guess for Wannier functions.
"""
@timing function guess_amn_random(basis::PlaneWaveBasis, n_wann::Integer)
    @assert n_wann > 0
    centers = [rand(3) for _ in 1:n_wann]
    l = zeros(Int, n_wann)
    guess_amn_hydrogenic(basis; centers, l)
end

"""
Pseudo-atomic-orbitals initial guess for Wannier functions.

The pseudo-atomic-orbitals are atomic orbitals from pseudopotentials,
centered on each atom.
"""
@timing function guess_amn_psp(basis::PlaneWaveBasis)
    model = basis.model

    # keep only pseudopotential atoms and positions
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    psps = [model.atoms[first(group)].psp for group in psp_groups]
    psp_positions = [model.positions[group] for group in psp_groups]

    isempty(psp_groups) && error("No pseudopotential atoms found in the model.")
    guess(kpt) = build_projection_vectors_pswfcs(basis, kpt, psps, psp_positions)

    # Lowdin orthonormalization
    ortho_lowdin ∘ guess
end

@doc raw"""
Compute the initial projection matrix for Wannierization.

``A_{m,n,k} = \langle ψ_{m,k} | g_{n,k} \rangle``
where ``g_{n,k}`` are Bloch sums of some real-space localized orbitals.

# Keyword arguments
- `guess`: the initial guess for the Wannier functions.
    A function that accepts a `Kpoint` `kpt` as arguments,
    and return a vector of length `n_wann` containing the initial guess for
    the MLWFs at that kpoint, see [`guess_amn_hydrogenic`](@ref),
    [`guess_amn_random`](@ref), [`guess_amn_psp`](@ref)
    for examples.
"""
@timing function compute_amn(
    basis::PlaneWaveBasis, ψ::AbstractVector{Matrix{Complex}}, guess::Function;
    spin::Integer=1,
)
    kpts = krange_spin(basis, spin)
    ψs = ψ[kpts]  # ψ for the selected spin

    n_kpts = length(kpts)
    n_bands = size(ψs[1], 2)
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
        for m in 1:n_bands
            for (n, ϕnk) in enumerate(ϕk)
                A[m, n, ik] = dot(ψk[:, m], ϕnk)
            end
        end
    end
    A
end

"""
Return a matrix of eigenvalues on a uniform grid for Wannierization.

# Arguments
- `basis`: the `PlaneWaveBasis`
- `eigenvalues`: the eigenvalues of the Hamiltonian, a vector of length
    `n_kpts`, each element is a vector of length `n_bands`. In the spin-polarized
    case, the vector is of length `2*n_kpts`, and the first half is for spin-up,
    the second half is for spin-down.
"""
function compute_eig(
    basis::PlaneWaveBasis, eigenvalues::AbstractVector{Vector{Real}}; spin::Int=1,
)
    kpts = krange_spin(basis, spin)
    eigs = eigenvalues[kpts]

    n_kpts = length(kpts)
    n_bands = length(eigs[1])
    E = zeros(eltype(eigs[1]), n_bands, n_kpts)

    for (ik, εk) in enumerate(eigs)
        for (n, εnk) in enumerate(εk)
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
to specify `spin`, which is needed in `compute_mmn`, `compute_eig`, etc.

For the explanation of the `ψ`, see [`compute_mmn`](@ref).
"""
@timing function compute_unk(
    basis::PlaneWaveBasis, ψ::AbstractVector{Matrix{Complex}}, kpt::Kpoint
)
    fft_size = basis.fft_size
    ik = findfirst(Ref(kpt) .== basis.kpoints)
    n_bands = size(ψ[ik], 2)
    unk = zeros(eltype(ψ[ik]), fft_size..., n_bands)

    for n in 1:n_bands
        unk[:, :, :, n] = ifft(basis, kpt, @view ψ[ik][:, n])
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
    kpoints = filter(k -> k.spin == 1, basis.kpoints)
    # columnwise vectors, fractional coordinates
    kpoints = hcat([k.coordinate for k in kpoints]...)
    @assert size(kpoints) == (3, prod(mp_grid))
    push!(win, :mp_grid => mp_grid, :kpoints => kpoints)

    if get(kwargs, :wannier_plot, false)
        push!(win, :wvfn_formatted => true)
    end

    if get(kwargs, :bands_plot, false)
        kpath  = irrfbz_path(basis.model)
        kpoint_path = []
        for path in kpath.paths
            for i in 1:length(path)-1
                A, B = path[i:i+1]  # write segment A -> B
                A_k = A => kpath.points[A]
                B_k = B => kpath.points[B]
                push!(kpoint_path, [A_k, B_k])
            end
        end
        push!(win, :kpoint_path => kpoint_path)
    end

    win
end

"""
Preprocess the scf results for Wannierization.

Unfold symmetries, exclude bands.

# Arguments
- `exclude_bands`: the band indices to be excluded from the Wannierization,
    usually the semicore states.
"""
function unfold_scfres_wannier(
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
        keep_bands = setdiff(1:n_bands, exclude_bands)
        ψ = [ψk[:, keep_bands] for ψk in ψ]
        eigenvalues = [e[keep_bands] for e in eigenvalues]
    end

    (; basis, ψ, eigenvalues)
end

"""
Compute & save matrices for Wannierization.

# Arguments
- `basis`: `PlaneWaveBasis`
- `ψ`: the wavefunctions
- `eigenvalues`: the eigenvalues
    usually obtained from
    `basis, ψ, eigenvalues = _unfold_scfres(scfres, exclude_bands)`
- `fileprefix`: the filename prefix for saving the matrices, e.g., if
    `fileprefix = "wannier/silicon"`, `amn` will be written to
    `wannier/silicon.amn`.

# Keyword arguments
- `nnkp`: a tuple of `(kpb_k, kpb_G)` specifying the ``b``-vectors.
    Can be generated either by `Wannier90` or `Wannier.jl`.
- `unk`: whether to compute real-space wavefunction files `UNK`.
    Warning: this takes a lot of memory and time.

!!! warning

    For simplicity, here only use random initial guess for Wannier functions.
    In principle, one should use better initial guesses, e.g.,
    [`guess_amn_hydrogenic`](@ref), [`guess_amn_psp`](@ref).
"""
function save_wannier(
    basis::PlaneWaveBasis, ψ::AbstractVector{T}, eigenvalues::AbstractVector{I};
    fileprefix::AbstractString,
    n_wann::Integer,
    nnkp::NamedTuple,
    spin::Integer=1,
    unk::Bool=false,
) where {T<:AbstractMatrix{Complex}, I<:AbstractVector{Real}}
    # Make wannier directory ...
    dir = dirname(fileprefix)
    isempty(dir) || mkpath(dir)

    n_bands = length(eigenvalues[1])
    win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands)
    WannierIO.write_win("$(fileprefix).win"; win...)

    M = compute_mmn(basis, ψ, nnkp.kpb_k, nnkp.kpb_G; spin)
    fname = "$(fileprefix).mmn"
    header = "Generated by DFTK.jl at $(now())"
    WannierIO.write_mmn(fname, M, nnkp.kpb_k, nnkp.kpb_G, header)

    E = compute_eig(basis, eigenvalues; spin)
    fname = "$(fileprefix).eig"
    WannierIO.write_eig(fname, E)

    # This is just a demonstration using random initial projections,
    # in practice one should use a better guess.
    guess = guess_amn_random(basis, n_wann)
    A = compute_amn(basis, ψ, guess; spin)
    fname = "$(fileprefix).amn"
    header = "Generated by DFTK.jl at $(now())"
    WannierIO.write_amn(fname, A)

    # Writing the unk files is expensive (requires FFTs), so only done if required.
    if unk
        kpts = krange_spin(basis, spin)
        for (ik, kpt) in enumerate(basis.kpoints[kpts])
            unk = compute_unk(basis, ψ, kpt)
            fname = joinpath(dir, @sprintf "UNK%05d.%1d" ik spin)
            WannierIO.write_w90_unk(fname, ik, unk)
        end
    end

    @info "Wannier matrices written to $(fileprefix)"

    (; win, M, A, E)
end

function _default_exclude_bands(scfres::NamedTuple)
    n_bands = length(scfres.eigenvalues[1])
    exclude_bands = (scfres.n_bands_converge+1):n_bands
    exclude_bands
end

function _run_wannier90_jll(fileprefix::AbstractString, postproc::Bool=false)
    dir, prefix = dirname(fileprefix), basename(fileprefix)
    pp = postproc ? "-pp" : ""
    wannier90_jll.wannier90(exe -> run(Cmd(`$exe $pp $prefix`; dir)))
end

"""
Call Wannier90 to Wannierize the results.

# Arguments
- `scfres`: the result of scf.

`kwargs` will be passed to [`save_wannier`](@ref).
"""
function run_wannier90(
    scfres::NamedTuple;
    fileprefix::AbstractString="wannier/wannier",
    exclude_bands::AbstractArray{Integer}=_default_exclude_bands(scfres),
    kwargs...,
)
    basis, ψ, eigenvalues = unfold_scfres_wannier(scfres, exclude_bands)

    n_bands = length(eigenvalues[1])
    @assert haskey(kwargs, :n_wann) "Must specify `n_wann` in `kwargs`"
    n_wann = kwargs[:n_wann]

    # Save files for Wannierization
    if basis.model.spin_polarization == :collinear
        prefixes = ["$(fileprefix)_up", "$(fileprefix)_dn"]
    else
        prefixes = [fileprefix]
    end

    for (spin, prefix) in enumerate(prefixes)
        @timing "Compute b-vectors" begin
            win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands, kwargs...)
            WannierIO.write_win("$(prefix).win"; win...)
            _run_wannier90_jll(prefix, true)
            nnkp = WannierIO.read_nnkp("$(prefix).nnkp")
        end

        @timing "Compute & save Wannier matrices" save_wannier(
            basis, ψ, eigenvalues; fileprefix=prefix, nnkp, spin, kwargs...)

        @timing "Run Wannierization" _run_wannier90_jll(prefix)
    end
    prefixes
end

"""
Call `Wannier.jl` to Wannierize the results.

# Arguments
- `scfres`: the result of scf.

`kwargs` will be passed to [`save_wannier`](@ref).
"""
function run_wannier(
    scfres::NamedTuple;
    fileprefix::AbstractString="wannier/wannier",
    exclude_bands::AbstractArray{Integer}=_default_exclude_bands(scfres),
    kwargs...,
)
    basis, ψ, eigenvalues = unfold_scfres_wannier(scfres, exclude_bands)

    n_bands = length(eigenvalues[1])
    @assert haskey(kwargs, :n_wann) "Must specify `n_wann` in `kwargs`"
    n_wann = kwargs[:n_wann]

    # Although with Wannier.jl we can pass matrices in-memory,
    # however I still save the files so that we can restart later.
    if basis.model.spin_polarization == :collinear
        prefixes = ["$(fileprefix)_up", "$(fileprefix)_dn"]
    else
        prefixes = [fileprefix]
    end

    @timing "Compute b-vectors" begin
        win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands, kwargs...)
        nnkp = Wannier.get_bvectors(win.unit_cell_cart, win.kpoints)
    end

    εF = auconvert(u"eV", scfres.εF)

    models = []
    for (spin, prefix) in enumerate(prefixes)
        @timing "Compute & save Wannier matrices" win, M, A, E = save_wannier(
            basis, ψ, eigenvalues; fileprefix=prefix, nnkp, spin, kwargs...)

        @timing "Run Wannierization" begin
            model = Wannier.Model(win, M, A, E)
            # I simply call disentangle which works for both isolated and
            # entangled cases. Also set frozen window just to be 1eV above Fermi.
            Wannier.set_froz_win!(model, εF + 1.0)
            model.U .= Wannier.disentangle(model)
            push!(models, model)
        end
    end
    models
end
