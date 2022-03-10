using Test
using DFTK: load_psp, pymatgen_lattice, pymatgen_structure, load_lattice, load_atoms
using DFTK: ElementCoulomb
using Unitful
using UnitfulAtomic
using PyCall

py"""
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
"""

@testset "Lattice DFTK -> pymatgen -> DFTK" begin
    reference = randn(3, 3)
    output = load_lattice(pymatgen_lattice(reference))
    @test output ≈ reference atol=1e-14
end

@testset "Lattice pymatgen -> DFTK -> pymatgen" begin
    data = randn(9)
    reference = py"Lattice($data)"
    output = pymatgen_lattice(load_lattice(reference))
    output = py"$output.matrix.ravel()" .+ 0
    @test data ≈ output atol=1e-14
end

@testset "Lattice DFTK -> pymatgen" begin
    a = randn(3)
    b = randn(3)
    c = randn(3)
    reference = [a b c]

    # Convert the lattice to python, make it flat and convert
    # to Julia array
    output = pymatgen_lattice(reference)
    outlatt = py"$output.matrix.ravel()" .+ 0
    @test a ≈ austrip.(outlatt[1:3] * u"Å") atol=1e-14
    @test b ≈ austrip.(outlatt[4:6] * u"Å") atol=1e-14
    @test c ≈ austrip.(outlatt[7:9] * u"Å") atol=1e-14
end

@testset "Structure pymatgen -> DFTK -> pymatgen" begin
    data = randn(9)
    reflattice = py"Lattice($data)"
    species = [1, 1, 6, 6, 6, 8]
    positions = [randn(3), randn(3), randn(3), randn(3), randn(3), randn(3)]

    reference = py"Structure($reflattice, $species, $positions)"
    atoms = load_atoms(reference)
    @test length(atoms) == 6
    @test all(at isa ElementCoulomb for at in atoms)
    @test atoms[1].symbol == :H
    @test atoms[2].symbol == :H
    @test atoms[3].symbol == :C
    @test atoms[4].symbol == :C
    @test atoms[5].symbol == :C
    @test atoms[6].symbol == :O

    output = pymatgen_structure(load_lattice(reference), atoms, load_positions(reference))
    @test output.lattice == reflattice
    for i in 1:6
        @test output.species[i].number    == species[i]
        @test output.sites[i].frac_coords == positions[i]
    end
end

@testset "Structure DFTK -> pymatgen" begin
    a = randn(3)
    b = randn(3)
    c = randn(3)
    lattice = [a b c]

    H = ElementCoulomb(1)
    O = ElementCoulomb(6)
    C = ElementCoulomb(8)
    atoms = [H, H, O, O, O, C, C]
    positions = [
        randn(3), randn(3),
        randn(3), randn(3), randn(3),
        randn(3), randn(3)
    ]

    # Convert the lattice to python, make it flat and convert
    # to Julia array
    output  = pymatgen_structure(lattice, atoms, positions)
    outlatt = py"$output.lattice.matrix.ravel()" .+ 0
    @test a ≈ austrip.(outlatt[1:3] * u"Å") atol=1e-14
    @test b ≈ austrip.(outlatt[4:6] * u"Å") atol=1e-14
    @test c ≈ austrip.(outlatt[7:9] * u"Å") atol=1e-14

    for i in 1:6
        @test output.species[i].number    == atoms[i].Z
        @test output.sites[i].frac_coords == positions[i]
    end
end
