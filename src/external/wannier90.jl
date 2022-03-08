using Dates
using DelimitedFiles
using WriteVTK

"""
Write a win file at the indicated prefix.
Parameters to Wannier90 can be added as kwargs: e.g. `num_iter=500`.
"""
function write_w90_win(fileprefix::String, basis::PlaneWaveBasis;
                       bands_plot=false, wannier_plot=false, kwargs...)
    @assert :num_bands in keys(kwargs)  # Required parameter
    @assert :num_wann  in keys(kwargs)  # Required parameter
    isnothing(basis.kgrid) && error("The basis must be constructed from a MP grid.")

    open(fileprefix *".win", "w") do fp
        println(fp, "! Generated by DFTK.jl at $(now())\n")

        # Drop parameters
        for (key, value) in pairs(kwargs)
            @printf fp "%-20s =   %-30s\n" key value
        end
        if wannier_plot
            println(fp, "wvfn_formatted = True")
            println(fp, "wannier_plot   = True")
        end

        # Delimiter for system
        println(fp, "\n", "!"^20 * " System \n\n")

        # Lattice vectors are in rows in Wannier90
        println(fp, "begin unit_cell_cart\nbohr")
        for vec in eachcol(basis.model.lattice)
            @printf fp "%10.6f %10.6f %10.6f \n" vec...
        end
        println(fp, "end unit_cell_cart \n")

        println(fp, "begin atoms_frac")
        for (element, positions) in basis.model.atoms
            for position in positions
                @printf fp "%-2s %10.6f %10.6f %10.6f \n" element.symbol position...
            end
        end
        println(fp, "end atoms_frac\n")

        # Delimiter for k-points block
        println(fp, "!"^20 * " k_points\n")

        if bands_plot
            kpathdata = high_symmetry_kpath(basis.model)
            length(kpathdata.kpath) > 1 || @warn(
                "Only first kpath branch considered in write_w90_win")
            path = kpathdata.kpath[1]

            println(fp, "begin kpoint_path")
            for i in 1:length(path)-1
                A, B = path[i:i+1]  # write segment A -> B
                @printf(fp, "%s %10.6f %10.6f %10.6f  ", A, round.(kpathdata.klabels[A], digits=5)...)
                @printf(fp, "%s %10.6f %10.6f %10.6f\n", B, round.(kpathdata.klabels[B], digits=5)...)
            end
            println(fp, "end kpoint_path")
            println(fp, "bands_plot = true\n")
        end

        println(fp, "mp_grid : $(basis.kgrid[1]) $(basis.kgrid[2]) $(basis.kgrid[3])\n")
        println(fp, "begin kpoints")
        for kpt in basis.kpoints
            @printf fp  "%10.6f %10.6f %10.6f\n" kpt.coordinate...
        end
        println(fp, "end kpoints")
    end
    nothing
end


"""
Read the .nnkp file provided by the preprocessing routine of Wannier90
(i.e. "wannier90.x -pp prefix")
Returns:
1) the array 'nnkpts' of k points, their respective nearest neighbors and associated
   shifing vectors (non zero if the neighbor is located in another cell).
2) the number 'nntot' of neighbors per k point.

TODO: add the possibility to exclude bands
"""
function read_w90_nnkp(fileprefix::String)
    fn = fileprefix * ".nnkp"
    !isfile(fn) && error("Expected file $fileprefix.nnkp not found.")

    # Extract nnkp block
    lines = readlines(fn)
    ibegin_nnkp = findfirst(lines .== "begin nnkpts")
    iend_nnkp   = findfirst(lines .== "end nnkpts")
    @assert !isnothing(ibegin_nnkp) && !isnothing(iend_nnkp)

    nntot  = parse(Int, lines[ibegin_nnkp + 1])
    nnkpts = map(ibegin_nnkp+2:iend_nnkp-1) do iline
        splitted = parse.(Int, split(lines[iline], ' ', keepempty=false))
        @assert length(splitted) == 5

        # Meaning of the entries:
        # 1st: Index of k-point
        # 2nd: Index of periodic image of k+b k-point
        # 3rd: Shift vector to get k-point of ikpb to the actual k+b point required
        (ik=splitted[1], ikpb=splitted[2], G_shift=splitted[3:5])
    end
    (; nntot, nnkpts)
end


"""
Write the eigenvalues in a format readable by Wannier90.
"""
function write_w90_eig(fileprefix::String, eigenvalues; n_bands)
    open("$fileprefix.eig", "w") do fp
        for (k, εk) in enumerate(eigenvalues)
            for (n, εnk) in enumerate(εk[1:n_bands])
                @printf fp "%3i  %3i   %25.18f \n" n k auconvert(Unitful.eV, εnk).val
            end
        end
    end
    nothing
end


@timing function write_w90_unk(fileprefix::String, basis, ψ; n_bands, spin=1)
    fft_size = basis.fft_size

    for ik in krange_spin(basis, spin)
        open(dirname(fileprefix) * (@sprintf "/UNK%05i.%i" ik spin), "w") do fp
            println(fp, "$(fft_size[1]) $(fft_size[2]) $(fft_size[3]) $ik $n_bands")
            for iband in 1:n_bands
                ψnk_real = G_to_r(basis, basis.kpoints[ik], @view ψ[ik][:, iband])
                for iz in 1:fft_size[3], iy in 1:fft_size[2], ix in 1:fft_size[1]
                    println(fp, real(ψnk_real[ix, iy, iz]), " ", imag(ψnk_real[ix, iy, iz]))
                end
            end
        end
    end
    nothing
end


@doc raw"""
Computes the matrix ``[M^{k,b}]_{m,n} = \langle u_{m,k} | u_{n,k+b} \rangle``
for given k, kpb = k+b.

`G_shift` is the "shifting" vector, correction due to the periodicity conditions
imposed on k -> ψ_k.
It is non zero if kpb is taken in another unit cell of the reciprocal lattice.
We use here that : ``u_{n(k + G_shift)}(r) = e^{-i*\langle G_shift,r \rangle} u_{nk}``
"""
@views function overlap_Mmn_k_kpb(basis::PlaneWaveBasis, ψ, ik, ikpb, G_shift, n_bands)
    # Search for common Fourier modes and their resp. indices in Bloch states k and kpb
    # TODO Check if this can be improved using the G vector mapping in the kpoints
    k   = basis.kpoints[ik]
    kpb = basis.kpoints[ikpb]
    equivalent_G_vectors = [(iGk, DFTK.index_G_vectors(basis, kpb, Gk + G_shift))
                            for (iGk, Gk) in enumerate(G_vectors(basis, k))]
    iGk   = [eqG[1] for eqG in equivalent_G_vectors if !isnothing(eqG[2])]
    iGkpb = [eqG[2] for eqG in equivalent_G_vectors if !isnothing(eqG[2])]

    # Compute overlaps
    # TODO This should be improved ...
    Mkb = zeros(ComplexF64, (n_bands, n_bands))
    for n in 1:n_bands
        for m in 1:n_bands
            # Select the coefficient in right order
            Mkb[m, n] = dot(ψ[ik][iGk, m], ψ[ikpb][iGkpb, n])
        end
    end
    iszero(Mkb) && return Matrix(I, n_bands, n_bands)
    Mkb
end

@timing function write_w90_mmn(fileprefix::String, basis::PlaneWaveBasis, ψ, nnkp; n_bands)
    open(fileprefix * ".mmn", "w") do fp
        println(fp, "Generated by DFTK at $(now())")
        println(fp, "$n_bands  $(length(ψ))  $(nnkp.nntot)")
        for (ik, ikpb, G_shift) in nnkp.nnkpts
            @printf fp "%i  %i  %i  %i  %i \n" ik ikpb G_shift...
            for ovlp in overlap_Mmn_k_kpb(basis, ψ, ik, ikpb, G_shift, n_bands)
                @printf fp "%22.18f %22.18f \n" real(ovlp) imag(ovlp)
            end
        end
    end
    nothing
end


@doc raw"""
Compute the matrix ``[A_k]_{m,n} = \langle ψ_m^k | g^{\text{per}}_n \rangle``

``g^{per}_n`` are periodized gaussians whose respective centers are given as an
 (num_bands,1) array [ [center 1], ... ].

Centers are to be given in lattice coordinates and G_vectors in reduced coordinates.
The dot product is computed in the Fourier space.

Given an orbital ``g_n``, the periodized orbital is defined by :
 ``g^{per}_n =  \sum\limits_{R \in {\rm lattice}} g_n( \cdot - R)``.
The  Fourier coefficient of ``g^{per}_n`` at any G
is given by the value of the Fourier transform of ``g_n`` in G.
"""
function compute_Ak_gaussian_guess(basis::PlaneWaveBasis, ψk, kpt, centers, n_bands)
    n_wannier = length(centers)
    # TODO This function should be improved in performance

    # associate a center with the fourier transform of the corresponding gaussian
    fourier_gn(center, qs) = [exp( 2π*(-im*dot(q, center) - dot(q, q) / 4) ) for q in qs]
    qs = vec(map(G -> G .+ kpt.coordinate, G_vectors(basis)))  # all q = k+G in reduced coordinates
    Ak = zeros(eltype(ψk), (n_bands, n_wannier))

    # Compute Ak
    for n in 1:n_wannier
        # Functions are l^2 normalized in Fourier, in DFTK conventions.
        norm_gn_per = norm(fourier_gn(centers[n], qs), 2)
        # Fourier coeffs of gn_per for k+G in common with ψk
        coeffs_gn_per = fourier_gn(centers[n], qs[kpt.mapping]) ./ norm_gn_per
        # Compute overlap
        for m in 1:n_bands
            # TODO Check the ordering of m and n here!
            Ak[m, n] = dot(ψk[:, m], coeffs_gn_per)
        end
    end
    Ak
end


@timing function write_w90_amn(fileprefix::String, basis::PlaneWaveBasis, ψ; n_bands, centers)
    open(fileprefix * ".amn", "w") do fp
        println(fp, "Generated by DFTK at $(now())")
        println(fp, "$n_bands   $(length(basis.kpoints))  $(length(centers))")

        for (ik, (ψk, kpt)) in enumerate(zip(ψ, basis.kpoints))
            Ak = compute_Ak_gaussian_guess(basis, ψk, kpt, centers, n_bands)
            for n in 1:size(Ak, 2)
                for m in 1:size(Ak, 1)
                    @printf(fp, "%3i %3i %3i  %22.18f %22.18f \n",
                            m, n, ik, real(Ak[m, n]), imag(Ak[m, n]))
                end
            end
        end
    end
    nothing
end

"""
Default random Gaussian guess for maximally-localised wannier functions
generated in reduced coordinates.
"""
default_wannier_centres(n_wannier) = [rand(1, 3) for _ in 1:n_wannier]

@timing function run_wannier90(scfres;
                               n_bands=size(scfres.ψ[1], 2) - scfres.n_ep_extra,
                               n_wannier=n_bands,
                               centers=default_wannier_centres(n_wannier),
                               fileprefix=joinpath("wannier90", "wannier"),
                               wannier_plot=false, kwargs...)
    # TODO None of the routines consider spin at the moment
    @assert scfres.basis.model.spin_polarization in (:none, :spinless)
    @assert length(centers) == n_wannier

    # Undo symmetry operations to get full k-point list
    scfres_unfold = unfold_bz(scfres)
    basis = scfres_unfold.basis
    ψ = scfres_unfold.ψ

    # Make wannier directory ...
    dir, prefix = dirname(fileprefix), basename(fileprefix)
    mkpath(dir)

    # Write input file and launch Wannier90 preprocessing
    write_w90_win(fileprefix, basis;
                  num_wann=length(centers), num_bands=n_bands, wannier_plot, kwargs...)
    wannier90_jll.wannier90(exe -> run(Cmd(`$exe -pp $prefix`; dir)))
    nnkp = read_w90_nnkp(fileprefix)

    # Files for main wannierization run
    write_w90_eig(fileprefix, scfres_unfold.eigenvalues; n_bands)
    write_w90_amn(fileprefix, basis, ψ; n_bands, centers)
    write_w90_mmn(fileprefix, basis, ψ, nnkp; n_bands)

    # Writing the unk files is expensive (requires FFTs), so only do if needed.
    if wannier_plot
        write_w90_unk(fileprefix, basis, ψ; n_bands, spin=1)
    end

    # Run Wannierisation procedure
    @timing "Wannierization" wannier90_jll.wannier90(exe -> run(Cmd(`$exe $prefix`; dir)))
    fileprefix
end

function read_w90_output_mat(file, n_bands_tot)
    Uks = readdlm(file); num_kpts, num_ψout, num_ψin = Uks[2,1:3];
    len_Uk = num_ψin*num_ψout

    list_Uk = []
    for k in 1:num_kpts
        i_Uk = 4 + (k-1)*(len_Uk+1)
        Uk = reshape([x+im*y for (x,y) in eachrow(Uks[i_Uk:i_Uk+(len_Uk-1), 1:2])],
                     num_ψin, num_ψout)
        (norm(Uk'Uk - I) > 1e-8) && (error("U matrix has non-orthogonal columns"))
        # Complete U matrices by zero to ease the writing of `apply_U_matrices`
        (n_bands_tot ≠ num_ψin) &&
            ( Uk = vcat(Uk, zeros(ComplexF64, n_bands_tot - num_ψin, num_ψout)) )
        push!(list_Uk, Uk)
    end
    list_Uk
end

@doc raw"""
For a given set of bands ``\{Ψ_{nk}\}_{1\leq n \leq N_in}``
and a Wannier90 file containing unitary matrices returns a set of band
``{Ψ_out_{nk}}_{1\leq n \leq N_out}``
with:
  - ``Ψ_out_{nk} = \sum\limits_{m=1}^{N_in} U_{mn}Ψ_{nk}``
  - ``N_in > N_out`` for disentanglement.
  - ``N_in = N_out`` for wannierization.
"""
function apply_U_matrices(file, ψ)
    # read .mat file
    n_bands_tot = size(ψ[1], 2)
    num_kpts = length(ψ)
    Uks = read_w90_output_mat(file, n_bands_tot)
    [ψ[k]*Uks[k] for k in 1:num_kpts]
end

@doc raw"""
Apply wannierization U matrices to return Wannier functions as a table ``W[k][G,n]``
The prefix in argument is the one returned by the `run_wannier90` routine above.
As in the wannierization process, the scfres has to be unfolded.
"""
function extract_wannier_functions(prefix, scfres::NamedTuple)
    scfres_unfold = unfold_bz(scfres); ψ_unfold = scfres_unfold.ψ
    # Apply disentanglement if needed
    isfile("$(prefix)_u_dis.mat") && (ψ_unfold =
                       apply_U_matrices("$(prefix)_u_dis.mat", ψ_unfold))
    # Construct Wannier functions
    apply_U_matrices("$(prefix)_u.mat", ψ_unfold)
end
        
"""
Produce vts files that are readable with Paraview to plot Wannier functions.
The wannier functions are given as a Bloch decomposition in the unit cell.
"""
function plot_wannier_function(basis, w_unit_cell, prefix, n_band)
    basis_unfold = DFTK.unfold_bz(basis)
    basis_supercell = cell_to_supercell(basis_unfold)

    # Adapt w to the ifft on supercell framework
    num_kpt = prod(basis.kgrid); num_wann = size(w_unit_cell[1],2)
    w_supercell = cell_to_supercell(w_unit_cell, basis_unfold, basis_supercell)
    wn_supercell = sum(w_supercell[:,ik*num_wann + n_band] for ik in 0:num_kpt-1)

    wn_real = G_to_r(basis_supercell, basis_supercell.kpoints[1], wn_supercell)

    # Extract real supercell grid and plot
    r_vec = collect(r_vectors_cart(basis_supercell))
    x = [r[1] for r in r_vec]; y = [r[2] for r in r_vec]; z = [r[3] for r in r_vec];
    vtk_grid(prefix*"_wannier_$(n_band)", x, y, z) do vtk
        vtk["value"]= abs2.(wn_real)
    end
    nothing
end
plot_wannier_function(scfres::NamedTuple, prefix::String, n_band) =
    plot_wannier_function(scfres.basis, extract_wannier_functions(prefix, scfres),
                          prefix, n_band)
