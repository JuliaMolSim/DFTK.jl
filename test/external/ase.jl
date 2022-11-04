using Test
using PythonCall
using DFTK
using Unitful
using UnitfulAtomic

@testset "ase.Atoms -> DFTK lattice / atoms / magnetisation -> ase.Atoms" begin
    pyatoms = pyexec(@NamedTuple{atoms::Py}, """
    import ase
    cell = [[3.21, 0.0, 0.0], [-1.605, 2.7799415461480477, 0.0], [0.0, 0.0, 5.21304]]
    positions = [[0.0, 0.0, 0.0], [0.0, 1.85329, 2.60652]]
    magmoms = [1.0, 2.0]

    atoms = ase.Atoms(symbols='Mg2', pbc=True, cell=cell,
                      positions=positions, magmoms=magmoms)
    """, Main).atoms

    lattice = load_lattice(pyatoms)
    @test lattice[:, 1] ≈ austrip(1u"Å") * [3.21, 0.0, 0.0]
    @test lattice[:, 2] ≈ austrip(1u"Å") * [-1.605, 2.7799415461480477, 0.0]
    @test lattice[:, 3] ≈ austrip(1u"Å") * [0.0, 0.0, 5.21304]

    atoms = load_atoms(pyatoms)
    @test length(atoms) == 2
    @test all(at isa ElementCoulomb for at in atoms)
    @test atoms[1].symbol == :Mg
    @test atoms[2].symbol == :Mg

    positions = load_positions(pyatoms)
    @test positions[1] ≈ [0, 0, 0]
    @test positions[2] ≈ [1/3, 2/3, 1/2] atol=1e-5

    magnetic_moments = load_magnetic_moments(pyatoms)
    @test magnetic_moments[1] == [0.0, 0.0, 1.0]
    @test magnetic_moments[2] == [0.0, 0.0, 2.0]

    newatoms = ase_atoms(lattice, atoms, positions, magnetic_moments)
    @test all(py"$newatoms.cell == atoms.cell")
    @test all(py"$newatoms.symbols == atoms.symbols")
    @test py"$newatoms.get_positions()" ≈  py"atoms.get_positions()"
    @test (  py"$newatoms.get_initial_magnetic_moments()"
           ≈ py"atoms.get_initial_magnetic_moments()")
end
