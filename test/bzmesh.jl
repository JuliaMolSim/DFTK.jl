using DFTK
using LinearAlgebra
using PyCall
using Test
using Unitful
include("testcases.jl")


@testset "bzmesh_uniform agrees with spglib" begin
    function test_against_spglib(kgrid_size; kshift=[0, 0, 0])
        kgrid_size = Vec3(kgrid_size)
        is_shift = ifelse.(kshift .== 0, 0, 1)
        n_kpts, _, grid =
            DFTK.spglib_get_stabilized_reciprocal_mesh(kgrid_size, [Matrix{Int64}(I, 3, 3)],
                                                       is_shift=is_shift)

        kcoords_spglib = [(kshift .+ grid[ik]) .// kgrid_size
                          for ik in 1:n_kpts]
        kcoords_spglib = DFTK.normalize_kpoint_coordinate.(kcoords_spglib)
        sort!(kcoords_spglib)

        kcoords, _ = bzmesh_uniform(kgrid_size, kshift=kshift)
        sort!(kcoords)

        @test kcoords == kcoords_spglib
    end

    test_against_spglib([ 2,  3,  2])
    test_against_spglib([ 3,  3,  3])
    test_against_spglib([ 3,  3,  3], kshift=[1//2, 0, 0])
    test_against_spglib([ 2,  3,  4])
    test_against_spglib([ 9, 11, 13])
end

@testset "bzmesh_ir_wedge is correct reduction" begin
    function test_reduction(system, kgrid_size, kirredsize;
                            supercell=[1, 1, 1], kshift=[0, 0, 0])
        lattice = system.lattice
        atoms = [ElementCoulomb(system.atnum) => system.positions]
        if supercell != [1, 1, 1]  # Make a supercell
            pystruct = pymatgen_structure(lattice, atoms)
            pystruct.make_supercell(supercell)
            lattice = load_lattice(pystruct)
            el = ElementCoulomb(system.atnum)
            atoms = [el => [s.frac_coords for s in pystruct.sites]]
        end

        red_kcoords, _ = bzmesh_uniform(kgrid_size, kshift=kshift)
        symmetries = DFTK.symmetry_operations(lattice, atoms)
        irred_kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, symmetries; kshift=kshift)

        @test length(irred_kcoords) == kirredsize

        # Try to reproduce all kcoords from irred_kcoords
        all_kcoords = Vector{Vec3{Rational{Int}}}()
        for (ik, k) in enumerate(irred_kcoords)
            append!(all_kcoords, [S * k for (S, τ) in ksymops[ik]])
        end

        # Normalize the obtained k-points and test for equality
        red_kcoords = sort([mod.(k .* kgrid_size, kgrid_size) for k in red_kcoords])
        all_kcoords = sort([mod.(k .* kgrid_size, kgrid_size) for k in all_kcoords])
        @test all_kcoords == red_kcoords
    end

    test_reduction(silicon, [ 2,  3,  2],   6)
    test_reduction(silicon, [ 3,  3,  3],   4)
    test_reduction(silicon, [ 2,  3,  4],  14)
    test_reduction(silicon, [ 9, 11, 13], 644)

    test_reduction(silicon, [ 3,  3,  3], 6, kshift=[1//2, 1//2, 1//2])
    test_reduction(silicon, [ 3,  3,  3], 6, kshift=[1//2, 0, 1//2])
    test_reduction(silicon, [ 3,  3,  3], 6, kshift=[0, 1//2, 0])

    test_reduction(silicon, [ 1,  4,  4],    7, supercell=[2, 1, 1])
    test_reduction(silicon, [ 1,  16,  16], 73, supercell=[4, 1, 1])

    test_reduction(magnesium, [ 2,  3,  2],   8)
    test_reduction(magnesium, [ 3,  3,  3],   6)
    test_reduction(magnesium, [ 2,  3,  4],  12)
    test_reduction(magnesium, [ 9, 11, 13], 350)

    test_reduction(platinum_hcp, [5, 5, 5], 63)
end

@testset "standardize_atoms" begin
    # Test unperturbed structure
    atoms = [ElementCoulomb(:Si) => silicon.positions]
    slattice, satoms = standardize_atoms(silicon.lattice, atoms, primitive=true)
    @test length(atoms) == 1
    @test atoms[1][1] == ElementCoulomb(:Si)
    @test length(atoms[1][2]) == 2
    @test atoms[1][2][1] == ones(3) ./ 8
    @test atoms[1][2][2] == -ones(3) ./ 8

    # Perturb structure
    plattice = silicon.lattice .+ 1e-8rand(3, 3)
    patoms = [ElementCoulomb(:Si) => [p + 1e-8rand(3) for p in silicon.positions]]
    plattice, patoms = standardize_atoms(plattice, patoms, primitive=true)

    # And check we get the usual silicon primitive cell back:
    a = plattice[1, 2]
    @test plattice == [0  a  a; a 0 a; a a 0]
    @test length(atoms) == 1
    @test atoms[1][1] == ElementCoulomb(:Si)
    @test length(atoms[1][2]) == 2
    @test atoms[1][2][1] - atoms[1][2][2] == ones(3) ./ 4
end

@testset "kgrid_from_minimal_spacing" begin
    # Test that units are stripped from both the lattice and the spacing
    lattice = [[-1.0 1 1]; [1 -1  1]; [1 1 -1]]
    @test kgrid_from_minimal_spacing(lattice * u"angstrom", 0.5 / u"angstrom") == [9; 9; 9]
end

@testset "kgrid_from_minimal_n_kpoints" begin
    lattice = [[-1.0 1 1]; [1 -1  1]; [1 1 -1]]
    @test kgrid_from_minimal_n_kpoints(lattice * u"Å", 1000) == [10, 10, 10]

    @test kgrid_from_minimal_n_kpoints(magnesium.lattice, 1) == [1, 1, 1]
    for n_kpt in [10, 20, 100, 400, 900, 1200]
        @test prod(kgrid_from_minimal_n_kpoints(magnesium.lattice, n_kpt)) ≥ n_kpt
    end

    lattice = diagm([4., 10, 0])
    @test kgrid_from_minimal_n_kpoints(lattice, 1000) == [50, 20, 1]
    @test kgrid_from_minimal_n_kpoints(diagm([10, 0, 0]), 913) == [913, 1, 1]
end
