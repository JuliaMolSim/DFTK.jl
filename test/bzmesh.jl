@testitem "MonkhorstPack reducible_kcoords agrees with spglib" begin
    using DFTK: normalize_kpoint_coordinate
    using DFTK
    using Spglib
    using LinearAlgebra

    function test_against_spglib(kgrid_size; kshift=[0, 0, 0])
        kgrid_size = Vec3(kgrid_size)
        is_shift = ifelse.(kshift .== 0, false, true)
        diagonal = Matrix{Int64}(I, 3, 3)
        spg_mesh = Spglib.get_stabilized_reciprocal_mesh([diagonal], kgrid_size;
                                                         is_shift, is_time_reversal=false)
        kcoords_spglib = normalize_kpoint_coordinate.(Spglib.eachpoint(spg_mesh))
        sort!(kcoords_spglib)

        (; kcoords) = DFTK.reducible_kcoords(MonkhorstPack(kgrid_size, kshift))
        @test sort!(kcoords) == kcoords_spglib
    end

    test_against_spglib([ 2,  3,  2])
    test_against_spglib([ 3,  3,  3])
    test_against_spglib([ 3,  3,  3], kshift=[1//2, 0, 0])
    test_against_spglib([ 2,  3,  4])
    test_against_spglib([ 9, 11, 13])
end

# PythonCall does not play nicely with MPI.
@testitem "MonkhorstPack irreducible_kcoords is correct reduction" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using Logging
    using ASEconvert
    (; silicon, magnesium, platinum_hcp) = TestCases.all_testcases

    function test_reduction(testcase, kgrid_size, kirredsize;
                            supercell=(1, 1, 1), kshift=[0, 0, 0])
        system = atomic_system(testcase.lattice, testcase.atoms, testcase.positions)

        # Make supercell
        if supercell != (1, 1, 1)
            ase_atoms = with_logger(() -> convert_ase(system), NullLogger())
            system = pyconvert(AbstractSystem, ase_atoms * pytuple(supercell))
        end
        kgrid = MonkhorstPack(kgrid_size, kshift)
        symmetries = symmetry_operations(system)
        sym_preserving_grid = DFTK.symmetries_preserving_kgrid(symmetries, kgrid)

        red_kcoords   = DFTK.reducible_kcoords(kgrid).kcoords
        irred_kcoords = DFTK.irreducible_kcoords(kgrid, sym_preserving_grid).kcoords
        @test length(irred_kcoords) == kirredsize

        # Try to reproduce all kcoords from irred_kcoords
        all_kcoords = Vector{Vec3{Rational{Int}}}()
        for k in irred_kcoords
            append!(all_kcoords, [symop.S * k for symop in sym_preserving_grid])
        end

        # Normalize the obtained k-points and test for equality
        red_kcoords = unique(sort([mod.(k .* kgrid_size, kgrid_size) for k in red_kcoords]))
        all_kcoords = unique(sort([mod.(k .* kgrid_size, kgrid_size) for k in all_kcoords]))
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

@testitem "standardize_atoms" setup=[TestCases] begin
    silicon = TestCases.silicon

    # Test unperturbed structure
    std = standardize_atoms(silicon.lattice, silicon.atoms, silicon.positions, primitive=true)
    @test length(std.atoms) == 2
    @test std.atoms[1].symbol == :Si
    @test std.atoms[2].symbol == :Si
    @test length(std.positions) == 2
    @test std.positions[1] - std.positions[2] ≈ 0.25ones(3)
    @test std.lattice ≈ silicon.lattice

    # Perturb structure
    plattice   = silicon.lattice .+ 1e-8rand(3, 3)
    patoms     = silicon.atoms
    ppositions = [p + 1e-8rand(3) for p in silicon.positions]
    std = standardize_atoms(plattice, patoms, ppositions, primitive=true)

    # And check we get the usual silicon primitive cell back:
    a = std.lattice[1, 2]
    @test std.lattice == [0  a  a; a 0 a; a a 0]
    @test length(std.atoms) == 2
    @test std.atoms[1].symbol == :Si
    @test std.atoms[2].symbol == :Si
    @test length(std.positions) == 2
    @test std.positions[1] - std.positions[2] ≈ 0.25ones(3)
end

@testitem "kgrid_from_maximal_spacing" begin
    using DFTK
    using Unitful

    # Test that units are stripped from both the lattice and the spacing
    lattice = [[-1.0 1 1]; [1 -1  1]; [1 1 -1]]
    @test kgrid_from_maximal_spacing(lattice * u"Å", 0.5 / u"Å").kgrid_size == [9, 9, 9]
end

@testitem "kgrid_from_minimal_n_kpoints" setup=[TestCases] begin
    using DFTK
    using Unitful
    using LinearAlgebra
    magnesium = TestCases.magnesium

    lattice = [[-1.0 1 1]; [1 -1  1]; [1 1 -1]]
    @test kgrid_from_minimal_n_kpoints(lattice * u"Å", 1000).kgrid_size == [10, 10, 10]

    @test kgrid_from_minimal_n_kpoints(magnesium.lattice, 1).kgrid_size == [1, 1, 1]
    for n_kpt in [10, 20, 100, 400, 900, 1200]
        @test length(kgrid_from_minimal_n_kpoints(magnesium.lattice, n_kpt)) ≥ n_kpt
    end

    lattice = diagm([4., 10, 0])
    @test kgrid_from_minimal_n_kpoints(lattice, 1000).kgrid_size          == [50, 20, 1]
    @test kgrid_from_minimal_n_kpoints(diagm([10, 0, 0]), 913).kgrid_size == [913, 1, 1]
end
