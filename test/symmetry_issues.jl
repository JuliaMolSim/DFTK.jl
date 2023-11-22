# This file collects issues with respect to our symmetry determination,
# which we collected and solved.


@testitem "Symmetry issues" begin
    using DFTK
    using DFTK: spglib_dataset
    using Unitful
    using UnitfulAtomic
    using AtomsBase

    @testset "CuO2" begin
        a = 4.474
        lattice = [[0, a, a], [a, 0, a], [a, a, 0]]u"bohr"
        x = 6.711
        y = 2.237
        atoms = [
            Atom(:Cu, [0, 0, 0]u"bohr", magnetic_moment=0),
            Atom(:O,  [x, y, x]u"bohr", magnetic_moment=0),
            Atom(:O,  [x, y, y]u"bohr", magnetic_moment=0),
        ]
        system = periodic_system(atoms, lattice)

        symmetries = DFTK.symmetry_operations(system; check_symmetry=true)
        @test length(symmetries) == 48
        @test spglib_dataset(system).spacegroup_number == 225
    end
end
