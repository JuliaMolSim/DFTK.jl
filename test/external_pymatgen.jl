using Test
using DFTK: load_psp, pymatgen_lattice, pymatgen_structure, load_lattice, load_atoms
using DFTK: ElementCoulomb
using Unitful
using UnitfulAtomic
using PyCall

py"""
import numpy as np
import pymatgen as mg
"""

@testset "Lattice DFTK -> pymatgen -> DFTK" begin
    reference = randn(3, 3)
    output = load_lattice(pymatgen_lattice(reference))
    @test output ≈ reference atol=1e-14
end

@testset "Lattice pymatgen -> DFTK -> pymatgen" begin
    data = randn(9)
    reference = py"mg.Lattice($data)"
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
    reflattice = py"mg.Lattice($data)"
    species = [1, 1, 6, 6, 6, 8]
    positions = [randn(3), randn(3), randn(3), randn(3), randn(3), randn(3)]

    reference = py"mg.Structure($reflattice, $species, $positions)"
    atoms = load_atoms(reference)
    @test length(atoms) == 3
    @test all(at isa ElementCoulomb for (at, positions) in atoms)
    @test atoms[1][1].symbol == :H
    @test atoms[2][1].symbol == :C
    @test atoms[3][1].symbol == :O

    output = pymatgen_structure(load_lattice(reference), atoms)
    @test output.lattice == reflattice
    for i in 1:6
        @test output.species[i].number == species[i]
        @test output.sites[i].frac_coords == positions[i]
    end
end

@testset "Structure DFTK -> pymatgen" begin
    a = randn(3)
    b = randn(3)
    c = randn(3)
    lattice = [a b c]

    atoms = [
        ElementCoulomb(1) => [randn(3), randn(3)],
        ElementCoulomb(6) => [randn(3), randn(3), randn(3)],
        ElementCoulomb(8) => [randn(3), randn(3)],
    ]

    # Convert the lattice to python, make it flat and convert
    # to Julia array
    output = pymatgen_structure(lattice, atoms)
    outlatt = py"$output.lattice.matrix.ravel()" .+ 0
    @test a ≈ austrip.(outlatt[1:3] * u"Å") atol=1e-14
    @test b ≈ austrip.(outlatt[4:6] * u"Å") atol=1e-14
    @test c ≈ austrip.(outlatt[7:9] * u"Å") atol=1e-14

    specmap = [1, 1, 2, 2, 2, 3, 3]
    offset = [0, 0, 2, 2, 2, 5, 5]
    for i in 1:6
        specpair = atoms[specmap[i]]
        ired = i - offset[i]
        @test output.species[i].number == specpair.first.Z
        @test output.sites[i].frac_coords == specpair.second[ired]
    end
end
