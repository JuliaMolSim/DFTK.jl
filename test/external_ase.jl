using Test
using PyCall
using DFTK
using Unitful
using UnitfulAtomic

ase = pyimport_e("ase")
if !ispynull(ase)
    py"""
    import ase
    """

    @testset "ase.Atoms -> DFTK lattice / atoms / magnetisation -> ase.Atoms" begin
        py"""
        import ase
        cell = [[3.21, 0.0, 0.0], [-1.605, 2.7799415461480477, 0.0], [0.0, 0.0, 5.21304]]
        positions = [[0.0, 0.0, 0.0], [0.0, 1.85329, 2.60652]]
        magmoms = [1.0, 2.0]
        atoms = ase.Atoms(symbols='Mg2', pbc=True, cell=cell,
                          positions=positions, magmoms=magmoms)
        """

        lattice = load_lattice(py"atoms")
        @test lattice[:, 1] ≈ austrip(1u"Å") * [3.21, 0.0, 0.0]
        @test lattice[:, 2] ≈ austrip(1u"Å") * [-1.605, 2.7799415461480477, 0.0]
        @test lattice[:, 3] ≈ austrip(1u"Å") * [0.0, 0.0, 5.21304]

        @test load_lattice(py"atoms") == load_lattice(py"atoms.cell")

        atoms = load_atoms(py"atoms")
        @test length(atoms) == 1
        @test all(at isa ElementCoulomb for (at, positions) in atoms)
        @test atoms[1][1].symbol == :Mg
        @test atoms[1][2][1] ≈ [0, 0, 0]
        @test atoms[1][2][2] ≈ [1/3, 2/3, 1/2] atol=1e-5

        magnetic_moments = load_magnetic_moments(py"atoms")
        @test length(magnetic_moments) == 1
        @test all(at isa ElementCoulomb for (at, magmoms) in magnetic_moments)
        @test magnetic_moments[1][1].symbol == :Mg
        @test magnetic_moments[1][2][1] == 1.0
        @test magnetic_moments[1][2][2] == 2.0

        newatoms = ase_atoms(lattice, atoms, magnetic_moments)
        @test all(py"$newatoms.cell == atoms.cell")
        @test all(py"$newatoms.symbols == atoms.symbols")
        @test py"$newatoms.get_positions()" ≈  py"atoms.get_positions()"
        @test (  py"$newatoms.get_initial_magnetic_moments()"
               ≈ py"atoms.get_initial_magnetic_moments()")
    end
end
