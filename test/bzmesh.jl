using DFTK: bzmesh_uniform, bzmesh_ir_wedge, ElementCoulomb, Vec3, Mat3
using DFTK: pymatgen_structure, load_lattice, standardize_atoms
using LinearAlgebra
using PyCall
using Test
include("testcases.jl")


@testset "bzmesh_uniform agrees with spglib" begin

    function test_against_spglib(kgrid_size; kshift=[0, 0, 0])
        kgrid_size = Vec3(kgrid_size)
        identity = collect(reshape(Matrix{Int64}(I, 3, 3), 3, 3, 1))
        is_shift = ifelse.(kshift .== 0, 0, 1)
        num_kpts, _, grid = DFTK.spglib_get_stabilized_reciprocal_mesh(kgrid_size, identity,
                                                        is_shift=is_shift)
        
        kcoords_spglib = [(kshift .+ grid[ik]) .// kgrid_size
                          for ik in 1:length(grid)]
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
    function test_reduction(system, kgrid_size; supercell=[1, 1, 1], kshift=[0, 0, 0])
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
        irred_kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, DFTK.symmetry_operations(lattice, atoms); kshift=kshift)

        # Try to reproduce all kcoords from irred_kcoords
        all_kcoords = Vector{Vec3{Rational{Int}}}()
        for (ik, k) in enumerate(irred_kcoords)
            append!(all_kcoords, [S * k for (S, τ) in ksymops[ik]])
        end

        # Normalize the obtained k-Points and test for equality
        red_kcoords = sort([mod.(k .* kgrid_size, kgrid_size) for k in red_kcoords])
        all_kcoords = sort([mod.(k .* kgrid_size, kgrid_size) for k in all_kcoords])
        @test all_kcoords == red_kcoords
    end

    test_reduction(silicon, [ 2,  3,  2])
    test_reduction(silicon, [ 3,  3,  3])
    test_reduction(silicon, [ 2,  3,  4])
    test_reduction(silicon, [ 9, 11, 13])

    test_reduction(silicon, [ 3,  3,  3], kshift=[1//2, 1//2, 1//2])
    test_reduction(silicon, [ 3,  3,  3], kshift=[1//2, 0, 1//2])
    test_reduction(silicon, [ 3,  3,  3], kshift=[0, 1//2, 0])

    test_reduction(silicon, [ 1,  4,  4], supercell=[2, 1, 1])
    test_reduction(silicon, [ 1,  16,  16], supercell=[4, 1, 1])

    test_reduction(magnesium, [ 2,  3,  2])
    test_reduction(magnesium, [ 3,  3,  3])
    test_reduction(magnesium, [ 2,  3,  4])
    test_reduction(magnesium, [ 9, 11, 13])
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
