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

@testitem "MonkhorstPack irreducible_kcoords is correct reduction" setup=[TestCases] begin
    using AtomsBuilder
    using DFTK
    using Logging
    (; silicon, magnesium, platinum_hcp) = TestCases.all_testcases

    function test_reduction(testcase, kgrid_size, kirredsize;
                            supercell=(1, 1, 1), kshift=[0, 0, 0])
        system = atomic_system(testcase.lattice, testcase.atoms, testcase.positions)
        if supercell != (1, 1, 1)  # Make supercell
            system = system * supercell
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

    test_reduction(silicon, [ 1,  1,  1],   1)
    test_reduction(silicon, [ 1,  1,  5],   3)
    test_reduction(silicon, [ 2,  3,  2],   6)
    test_reduction(silicon, [ 3,  3,  3],   4)
    test_reduction(silicon, [ 2,  3,  4],  14)
    test_reduction(silicon, [ 9, 11, 13], 644)

    test_reduction(silicon, [ 3,  3,  3], 6, kshift=[1//2, 1//2, 1//2])
    test_reduction(silicon, [ 3,  3,  3], 6, kshift=[1//2, 0, 1//2])
    test_reduction(silicon, [ 3,  3,  3], 6, kshift=[0, 1//2, 0])

    test_reduction(silicon, [ 1,  4,  4],    7, supercell=(2, 1, 1))
    test_reduction(silicon, [ 1,  16,  16], 73, supercell=(4, 1, 1))

    test_reduction(magnesium, [ 2,  3,  2],   8)
    test_reduction(magnesium, [ 3,  3,  3],   6)
    test_reduction(magnesium, [ 2,  3,  4],  12)
    test_reduction(magnesium, [ 9, 11, 13], 350)

    test_reduction(platinum_hcp, [5, 5, 5], 63)
end

@testitem "MonkhorstPack: equivalent kshifts give identical grids" setup=[TestCases] begin
    using DFTK
    using DFTK: normalize_kpoint_coordinate, reducible_kcoords, irreducible_kcoords
    using DFTK: symmetries_preserving_kgrid, breaks_time_reversal_symmetry
    silicon = TestCases.silicon
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)
    kgrid_size = [3, 3, 3]

    function sorted_kcoords(kgrid)
        sort(normalize_kpoint_coordinate.(reducible_kcoords(kgrid).kcoords))
    end

    # Different ways of specifying the same half shift (equal up to integers)
    half_shift_grids = [MonkhorstPack(kgrid_size; kshift=[0,  1//2,  0]),
                        MonkhorstPack(kgrid_size; kshift=[0, -1//2,  0]),
                        MonkhorstPack(kgrid_size; kshift=[0,   0.5,  0]),
                        MonkhorstPack(kgrid_size; kshift=(0,   1/2,  0)),
                        MonkhorstPack(kgrid_size; kshift=[1,  3//2, -2]),
                        MonkhorstPack(kgrid_size, [0, 0.5, 0])]
    ref = first(half_shift_grids)
    syms_ref = symmetries_preserving_kgrid(model.symmetries, ref)
    @assert length(syms_ref) > 1

    for kgrid in half_shift_grids
        @test kgrid.kgrid_size == ref.kgrid_size
        @test kgrid.kshift     == ref.kshift
        @test sorted_kcoords(kgrid) == sorted_kcoords(ref)
        @test !breaks_time_reversal_symmetry(kgrid)

        # Half shifts keep the symmetry reduction working: this should reduce to 6 points
        syms = symmetries_preserving_kgrid(model.symmetries, kgrid)
        @test length(syms) == length(syms_ref)
        @test length(irreducible_kcoords(kgrid, syms).kcoords) == 6
    end

    # Equivalent generic shift, which break time-reversal and (since Spglib does not)
    # support it, we remove all symmetries in symmetries_preserving_kgrid.
    generic_shift_grids = [MonkhorstPack(kgrid_size; kshift=[ 1//3, 0, 1//4]),
                           MonkhorstPack(kgrid_size; kshift=[-2//3, 1, 0.25]),
                           MonkhorstPack(kgrid_size; kshift=( 1/3,  0, 1/4))]
    ref = first(generic_shift_grids)
    for kgrid in generic_shift_grids
        @test kgrid.kshift == ref.kshift
        @test sorted_kcoords(kgrid) == sorted_kcoords(ref)
        @test breaks_time_reversal_symmetry(kgrid)

        syms = symmetries_preserving_kgrid(model.symmetries, kgrid)
        @test length(syms) == 1
        @test isone(only(syms))
        @test length(irreducible_kcoords(kgrid, syms).kcoords) == prod(kgrid_size)
    end
end

@testitem "standardize_atoms" setup=[TestCases] begin
    using AtomsBase
    using DFTK: standardize_atoms
    silicon = TestCases.silicon

    # Test unperturbed structure
    std = standardize_atoms(silicon.lattice, silicon.atoms, silicon.positions, primitive=true)
    @test length(std.atoms) == 2
    @test species(std.atoms[1]) == ChemicalSpecies(:Si)
    @test species(std.atoms[2]) == ChemicalSpecies(:Si)
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
    @test species(std.atoms[1]) == ChemicalSpecies(:Si)
    @test species(std.atoms[2]) == ChemicalSpecies(:Si)
    @test length(std.positions) == 2
    @test std.positions[1] - std.positions[2] ≈ 0.25ones(3)
end

@testitem "KgridSpacing" setup=[TestCases] begin
    using DFTK
    using Unitful

    @testset "Simple lattice with units" begin
        # Test that units are stripped from both the lattice and the spacing
        kgen = KgridSpacing(0.5 / u"Å")
        lattice = [[-1.0 1 1]; [1 -1  1]; [1 1 -1]]
        @test DFTK.build_kgrid(lattice * u"Å", kgen).kgrid_size == [9, 9, 9]
    end
    @testset "Magnesium system" begin
        kgen = KgridSpacing(0.5 / u"Å")
        magnesium = TestCases.magnesium
        @test DFTK.build_kgrid(magnesium.lattice, kgen).kgrid_size == [5, 5, 3]
    end
end

@testitem "KgridTotalNumber" setup=[TestCases] begin
    using DFTK
    using Unitful
    using LinearAlgebra

    @testset "Simple lattice with units" begin
        lattice = [[-1.0 1 1]; [1 -1  1]; [1 1 -1]]
        kgen = KgridTotalNumber(1000)
        @test DFTK.build_kgrid(lattice * u"Å", kgen).kgrid_size == [10, 10, 10]
    end

    @testset "Magnesium system" begin
        magnesium = TestCases.magnesium
        @test DFTK.build_kgrid(magnesium.lattice, KgridTotalNumber(1)).kgrid_size == [1, 1, 1]
        for n_kpt in [10, 20, 100, 400, 900, 1200]
            kgrid = DFTK.build_kgrid(magnesium.lattice, KgridTotalNumber(n_kpt))
            @test length(kgrid) ≥ n_kpt
        end
    end

    @testset "Reduced dimension" begin
        lattice = diagm([4., 10, 0])
        @test DFTK.build_kgrid(lattice, KgridTotalNumber(1000)). kgrid_size == [50, 20, 1]
        lattice = diagm([10, 0, 0])
        @test DFTK.build_kgrid(lattice, KgridTotalNumber(913)).kgrid_size   == [913, 1, 1]
    end
end
