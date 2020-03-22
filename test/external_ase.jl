using Test
using PyCall
using DFTK

ase = pyimport_e("ase")
if !ispynull(ase)
    py"""
    import ase
    """

    @testset "ase.Atoms -> DFTK lattice / atoms" begin
        py"""
        import ase
        cell = [[3.21, 0.0, 0.0], [-1.605, 2.7799415461480477, 0.0], [0.0, 0.0, 5.21304]]
        positions = [[0.0, 0.0, 0.0], [0.0, 1.85329, 2.60652]]
        atoms = ase.Atoms(symbols='Mg2', pbc=True, cell=cell, positions=positions)
        """

        lattice = load_lattice(py"atoms")
        @test lattice[:, 1] ≈ DFTK.units.Ǎ * [3.21, 0.0, 0.0]
        @test lattice[:, 2] ≈ DFTK.units.Ǎ * [-1.605, 2.7799415461480477, 0.0]
        @test lattice[:, 3] ≈ DFTK.units.Ǎ * [0.0, 0.0, 5.21304]

        @test load_lattice(py"atoms") == load_lattice(py"atoms.cell")

        atoms = load_atoms(py"atoms")
        @test length(atoms) == 1
        @test all(at isa ElementCoulomb for (at, positions) in atoms)
        @test atoms[1][1].symbol == :Mg
        @test atoms[1][2][1] ≈ [0, 0, 0]
        @test atoms[1][2][2] ≈ [1/3, 2/3, 1/2] atol=1e-5
    end
end
