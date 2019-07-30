using PyCall

#
# =================  Sandbox and playground to be moved elsewhere =================
#

function sample_kpoints_monkhorst_pack(grid_size)
    # TODO Rename this function
    # Paper: Monkhorst1976
    # Generate a Monkhorst-Pack uniformly spaced k-Point grid
    grid_size = Vec3{Int}(grid_size)
    start = -ceil.(Int, (grid_size .- 1) ./ 2)
    stop  = floor.(Int, (grid_size .- 1) ./ 2)
    kpoints = [Vec3([i, j, k] .// grid_size) for i=start[1]:stop[1],
               j=start[2]:stop[2], k=start[3]:stop[3]]
    ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))] for _ in 1:length(kpoints)]
    vec(kpoints), ksymops
end


function build_spglib_cell(lattice, composition...)
    n_species = sum(length(positions) for (spec, positions) in composition)
    spg_numbers = Vector{Int}(undef, n_species)
    spg_positions = Matrix{Float64}(undef, n_species, 3)

    offset = 0
    nextnumber = 1
    for (spec, positions) in composition
        for (i, pos) in enumerate(positions)
            # assign the same number to all species with this position
            spg_numbers[offset + i] = nextnumber
            spg_positions[offset + i, :] = pos
        end
        offset += length(positions)
        nextnumber += 1
    end

    # Note: In the python interface of spglib the lattice vectors
    #       are given in rows, but DFTK uses columns
    #       For future reference: The C interface spglib also uses columns.
    (lattice', spg_positions, spg_numbers)
end


function sample_kpoints_symmetry_reduced(grid_size, lattice, composition...;
                                         tol_symmetry=1e-5)
    # TODO Rename this function
    spglib = pyimport_conda("spglib", "spglib")
    cell = build_spglib_cell(lattice, composition...)

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_symops = spglib.get_symmetry(cell, symprec=tol_symmetry)
    mapping, grid = spglib.get_stabilized_reciprocal_mesh(
        grid_size, spg_symops["rotations"], is_shift=[0, 0, 0], is_time_reversal=true
    )

    # Convert irreducible k-Points to DFTK conventions
    grid_size = Vec3{Int}(grid_size)
    kpoints = [Vec3{Int}(grid[ik + 1, :]) .// grid_size for ik in unique(mapping)]

    # Find the indices of the corresponding reducible k-Points in `grid`, which one of the
    # irreducible k-Points in `kpoints` generates.
    k_all_reducible = [findall(isequal(elem), mapping) for elem in unique(mapping)]

    # The list of symmetry operations (for each irreducible k-Point) needed to do the
    # respective mapping irreducible -> reducible implied by `k_all_reducible`.
    SymOp = Tuple{Mat3{Int}, Vec3{Float64}}
    ksymops = Vector{Vector{SymOp}}(undef, length(kpoints))

    for (ik, k) in enumerate(kpoints)
        ksymops[ik] = Vector{SymOp}()
        for ired in k_all_reducible[ik]
            kred = Vec3(grid[ired, :]) .// grid_size

            isym = findfirst(1:size(spg_symops["rotations"], 1)) do idx
                # Are kred and S' * k equal k-Points ?
                S = view(spg_symops["rotations"], idx, :, :)
                all(elem.den == 1 for elem in (kred - (S' * k)))
            end

            if isym === nothing
                error("No corresponding symop found for $k -> $kred")
            else
                S = spg_symops["rotations"][isym, :, :]
                τ = spg_symops["translations"][isym, :]
                push!(ksymops[ik], (S', -inv(S) * τ))
            end
        end
    end

    kpoints, ksymops
end


function test_reduction(grid_size, irred_kpoints, ksymops)
    # TODO This should become a proper test!
    red_kpoints, _ = sample_kpoints_monkhorst_pack(grid_size)

    # Make this a proper unit test
    # Test reproducing the full set of kpoints
    red_kpoints = [Vec3(k) for k in red_kpoints]
    irred_kpoints = [Vec3(k) for k in irred_kpoints]

    all_kpoints = Array{Vec3{Float64}}(undef, 0)
    for (ik, k) in enumerate(irred_kpoints) 
        append!(all_kpoints, [S * k for (S, τ) in ksymops[ik]])
    end

    grid_size = Vec3(grid_size)
    red_kpoints = [mod.(round.(Int, k .* grid_size), grid_size) for k in red_kpoints]
    all_kpoints = [mod.(round.(Int, k .* grid_size), grid_size) for k in all_kpoints]

    all_kpoints = sort(Vector.(all_kpoints))
    red_kpoints = sort(Vector.(red_kpoints))
    @assert all_kpoints == red_kpoints
end


function test_density_mapping(kgrid, lattice, composition...; Ecut=5, n_bands=4, tol=1e-8)
    function get_bands(basis, composition...; n_bands=4, tol=1e-8)
        ham = Hamiltonian(
            basis,
            pot_local=build_local_potential(basis, composition...),
            pot_nonlocal=build_nonlocal_projectors(basis, composition...),
            pot_hartree=PotHartree(basis),
            pot_xc=PotXc(basis, :lda_xc_teter93)
        )

        ρ = guess_gaussian_sad(basis, composition...)
        values_hartree = empty_potential(ham.pot_hartree)
        values_xc = empty_potential(ham.pot_xc)
        energies = Dict{Symbol, real(eltype(ρ))}()
        update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
        update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

        res = lobpcg(ham, n_bands; pot_hartree_values=values_hartree,
                     pot_xc_values=values_xc, prec=PreconditionerKinetic(ham, α=0.1),
                     tol=tol)
        res.X, res.λ
    end

    @assert n_bands >= 4
    occupation = [2ones(4); zeros(n_bands - 4)]

    println("Running full k-Point diagonalisation ... please wait.")
    kfull, knosym = sample_kpoints_monkhorst_pack(kgrid)
    kones = [length(symops) for symops in knosym]
    kones = kones / sum(kones)
    Ggrid = determine_grid_size(lattice, Ecut; kpoints=kfull) * ones(3)
    basis_full = PlaneWaveBasis(lattice, Ggrid, Ecut, kfull, kones, knosym, kgrid)
    Psi_full, orben_full = get_bands(basis_full, composition..., n_bands=n_bands, tol=tol)

    println("Running irreducible k-Point diagonalisation ... please wait.")
    kpoints, ksymops = sample_kpoints_symmetry_reduced(kgrid, lattice, composition...)
    kweights = [length(symops) for symops in ksymops]
    kweights = kweights / sum(kweights)
    basis_ir = PlaneWaveBasis(lattice, Ggrid, Ecut, kpoints, kweights, ksymops, kgrid)
    Psi_ir, orben_ir = get_bands(basis_ir, composition..., n_bands=n_bands, tol=tol)

    function getindex_G(grid_size, G)
        start = -ceil.(Int, (grid_size .- 1) ./ 2)
        stop  = floor.(Int, (grid_size .- 1) ./ 2)

        if !all(start .<= G .<= stop)
            return nothing
        end

        strides = [1, grid_size[1], grid_size[1] * grid_size[2]]
        sum((G .+ stop) .* strides) + 1
    end

    ρ_accu = similar(Psi_ir[1][:, 1], prod(basis_ir.grid_size))
    ρ_accu .= 0

    println("Running comparison.")
    for (ik, k) in enumerate(basis_ir.kpoints)
        ρ_k = build_partial_density(basis_ir, ik, Psi_ir[ik], occupation)

        for (S, τ) in basis_ir.ksymops[ik]
            kred = S * k
            ikfull = findfirst(1:length(basis_full.kpoints)) do idx
                all(elem.den == 1 for elem in rationalize.((kred -basis_full.kpoints[idx])))
            end
            @assert ikfull !== nothing

            # Orbital energies should agree (by symmetry)
            @assert orben_full[ikfull] ≈ orben_ir[ik] atol=tol

            # Compare partial waves with appropriate mapping
            ρ_kred = build_partial_density(basis_full, ikfull, Psi_full[ikfull], occupation)

            # Transform ρ_k -> ρ_kred and compare with the reference obtained above
            ρ_test = similar(ρ_kred)
            ρ_test .= 0
            for (ig, G) in enumerate(basis_ρ(basis_ir))
                Gired = Vec3{Int}(inv(S) * G)
                igired = getindex_G(basis_ir.grid_size, Gired)
                if igired != nothing
                    ρ_test[ig] += cis(2π * dot(G, τ)) * ρ_k[igired]
                end
            end

            actdiff = maximum(abs.(ρ_test - ρ_kred))
            println("$kred -> $k   ",  actdiff, " < ", 10tol)
            @assert actdiff < 10tol

            ρ_accu .+= ρ_test

            n_k_touched += 1
        end
    end

    # TODO Might remove this
    @assert n_bands >= 4
    occupation_full = [[2ones(4); zeros(n_bands - 4)] for _ in 1:length(basis_full.kpoints)]
    ρ_full = compute_density(basis_full, Psi_full, occupation_full;
                             tolerance_orthonormality=tol)
    @assert maximum(abs.(ρ_accu / prod(kgrid) - ρ_full)) < tol
end


function test_compute_density(kgrid, lattice, composition...; Ecut=5, n_bands=4, tol=1e-8)
    function get_bands(basis, composition...; n_bands=4, tol=1e-8)
        ham = Hamiltonian(
            basis,
            pot_local=build_local_potential(basis, composition...),
            pot_nonlocal=build_nonlocal_projectors(basis, composition...),
            pot_hartree=PotHartree(basis),
            pot_xc=PotXc(basis, :lda_xc_teter93)
        )

        ρ = guess_gaussian_sad(basis, composition...)
        values_hartree = empty_potential(ham.pot_hartree)
        values_xc = empty_potential(ham.pot_xc)
        energies = Dict{Symbol, real(eltype(ρ))}()
        update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
        update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

        res = lobpcg(ham, n_bands; pot_hartree_values=values_hartree,
                     pot_xc_values=values_xc, prec=PreconditionerKinetic(ham, α=0.1),
                     tol=tol)
        res.X, res.λ
    end

    println("Running full k-Point diagonalisation ... please wait.")
    kfull, knosym = sample_kpoints_monkhorst_pack(kgrid)
    kones = [length(symops) for symops in knosym]
    kones = kones / sum(kones)
    Ggrid = determine_grid_size(lattice, Ecut; kpoints=kfull) * ones(3)
    basis_full = PlaneWaveBasis(lattice, Ggrid, Ecut, kfull, kones, knosym, kgrid)
    Psi_full, orben_full = get_bands(basis_full, composition..., n_bands=n_bands, tol=tol)

    @assert n_bands >= 4
    occupation_full = [[2ones(4); zeros(n_bands - 4)] for _ in 1:length(basis_full.kpoints)]
    ρ_full = compute_density(basis_full, Psi_full, occupation_full;
                             tolerance_orthonormality=tol)

    println("Running irreducible k-Point diagonalisation ... please wait.")
    kpoints, ksymops = sample_kpoints_symmetry_reduced(kgrid, lattice, composition...)
    kweights = [length(symops) for symops in ksymops]
    kweights = kweights / sum(kweights)
    basis_ir = PlaneWaveBasis(lattice, Ggrid, Ecut, kpoints, kweights, ksymops, kgrid)
    Psi_ir, orben_ir = get_bands(basis_ir, composition..., n_bands=n_bands, tol=tol)

    occupation_ir = [[2ones(4); zeros(n_bands - 4)] for _ in 1:length(basis_ir.kpoints)]
    ρ_ir = compute_density(basis_ir, Psi_ir, occupation_ir; tolerance_orthonormality=tol)

    @assert maximum(abs.(ρ_ir - ρ_full)) < 10tol
end



#
# ==================== Actual code ======================
#


function build_partial_density(pw, ik, Ψk, occupation; tolerance_orthonormality=-1)
    n_states = size(Ψk, 2)
    @assert n_states == length(occupation)

    # Fourier-transform the wave functions to real space
    Ψk_real = similar(Ψk[:, 1], size(pw.FFT)..., n_states)
    for ist in 1:n_states
        G_to_r!(pw, Ψk[:, ist], view(Ψk_real, :, :, :, ist), gcoords=pw.basis_wf[ik])
    end

    # TODO I am not quite sure why this is needed here
    #      maybe this points at an error in the normalisation of the
    #      Fourier transform
    Ψk_real /= sqrt(pw.unit_cell_volume)

    if tolerance_orthonormality > 0
        # TODO These assertions should go to a test case
        # Check for orthonormality of the Ψ_k_reals
        n_fft = prod(size(pw.FFT))
        T = real(eltype(Ψk_real))
        Ψk_real_mat = reshape(Ψk_real, n_fft, n_states)
        Ψk_real_overlap = adjoint(Ψk_real_mat) * Ψk_real_mat
        nondiag = Ψk_real_overlap - I * (n_fft / pw.unit_cell_volume)
        @assert maximum(abs.(nondiag)) < max(1000 * eps(T), tolerance_orthonormality)
    end

    # Build the partial density for this k-Point
    ρk_real = similar(Ψk[:, 1], size(pw.FFT)...)
    ρk_real .= 0
    for ist in 1:n_states
        @. @views begin
            ρk_real += occupation[ist] * Ψk_real[:, :, :, ist] * conj(Ψk_real[:, :, :, ist])
        end
    end
    Ψk_real = nothing

    # Check ρk is real and positive and properly normalized
    T = real(eltype(ρk_real))
    @assert maximum(imag(ρk_real)) < 100 * eps(T)
    @assert minimum(real(ρk_real)) ≥ 0

    n_electrons = sum(ρk_real) * pw.unit_cell_volume / prod(size(pw.FFT))
    @assert abs(n_electrons - sum(occupation)) < sqrt(eps(T))

    ρk = similar(Ψk[:, 1], prod(pw.grid_size))
    r_to_G!(pw, ρk_real, ρk)
    ρk
end


"""
    compute_density(pw::PlaneWaveBasis, Psi::AbstractVector, occupation::AbstractVector;
                    tolerance_orthonormality)

Compute the density for a wave function `Psi` discretised on the plane-wave grid `pw`,
where the individual k-Points are occupied according to `occupation`. `Psi` should
be one coefficient matrix per k-Point. If `tolerance_orthonormality` is ≥ 0, some
orthonormality properties are verified explicitly.
"""
function compute_density(pw::PlaneWaveBasis, Psi::AbstractVector,
                         occupation::AbstractVector; tolerance_orthonormality=-1)
    # TODO This function could be made in-place

    n_k = length(pw.kpoints)
    @assert n_k == length(Psi)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(pw.basis_wf[ik]) == size(Psi[ik], 1)
        @assert length(occupation[ik]) == size(Psi[ik], 2)
    end
    @assert n_k > 0

    # TODO Not sure this is reasonable
    @assert all(occupation[ik] == occupation[1] for ik in 1:n_k)

    function getindex_G(grid_size, G)  # This feels a little strange
        start = -ceil.(Int, (grid_size .- 1) ./ 2)
        stop  = floor.(Int, (grid_size .- 1) ./ 2)

        if !all(start .<= G .<= stop)
            return nothing
        end

        strides = [1, grid_size[1], grid_size[1] * grid_size[2]]
        sum((G .+ stop) .* strides) + 1
    end

    ρ = similar(Psi[1][:, 1], prod(pw.grid_size))
    ρ .= 0
    for (ik, k) in enumerate(pw.kpoints)
        # TODO Using the kweights here instead of ρ / prod(pw.kgrid)
        # in the end does not work here ... not sure why at the moment
        # pw.kweights[ik] * 
        ρ_k = build_partial_density(
            pw, ik, Psi[ik], occupation[ik],
            tolerance_orthonormality=tolerance_orthonormality
        )
        for (S, τ) in pw.ksymops[ik]
            # TODO If τ == [0,0,0] and if S == id
            #      this routine can be simplified or even skipped

            # Transform ρ_k -> to the partial density at S * k
            for (ig, G) in enumerate(basis_ρ(pw))
                Gired = Vec3{Int}(inv(S) * G)
                igired = getindex_G(pw.grid_size, Gired)
                if igired != nothing
                    ρ[ig] += cis(2π * dot(G, τ)) * ρ_k[igired]
                end
            end # ig, G
        end # S, τ
    end  # ik, k

    ρ / prod(pw.kgrid)
end
