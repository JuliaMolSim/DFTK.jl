using Test
using DFTK: load_psp, pymatgen_lattice, pymatgen_structure, load_lattice, load_atoms
using DFTK: units, ElementAllElectron
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
    @test a ≈ outlatt[1:3] * units.Ǎ atol=1e-14
    @test b ≈ outlatt[4:6] * units.Ǎ atol=1e-14
    @test c ≈ outlatt[7:9] * units.Ǎ atol=1e-14
end

@testset "Structure pymatgen -> DFTK -> pymatgen" begin
    data = randn(9)
    reflattice = py"mg.Lattice($data)"
    species = [1, 1, 6, 6, 6, 8]
    positions = [randn(3), randn(3), randn(3), randn(3), randn(3), randn(3)]
    pspmap = Dict(1 => "hgh/lda/h-q1",
                  6 => "hgh/lda/c-q4",
                  8 => "hgh/lda/o-q6")

    reference = py"mg.Structure($reflattice, $species, $positions)"
    atoms = load_atoms(reference, pspmap=pspmap)
    @test length(atoms) == 3
    @test atoms[1].first.psp.identifier == "hgh/lda/h-q1"
    @test atoms[2].first.psp.identifier == "hgh/lda/c-q4"
    @test atoms[3].first.psp.identifier == "hgh/lda/o-q6"

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
        ElementAllElectron(1) => [randn(3), randn(3)],
        ElementAllElectron(6) => [randn(3), randn(3), randn(3)],
        ElementAllElectron(8) => [randn(3), randn(3)],
    ]

    # Convert the lattice to python, make it flat and convert
    # to Julia array
    output = pymatgen_structure(lattice, atoms)
    outlatt = py"$output.lattice.matrix.ravel()" .+ 0
    @test a ≈ outlatt[1:3] * units.Ǎ atol=1e-14
    @test b ≈ outlatt[4:6] * units.Ǎ atol=1e-14
    @test c ≈ outlatt[7:9] * units.Ǎ atol=1e-14

    specmap = [1, 1, 2, 2, 2, 3, 3]
    offset = [0, 0, 2, 2, 2, 5, 5]
    for i in 1:6
        specpair = atoms[specmap[i]]
        ired = i - offset[i]
        @test output.species[i].number == specpair.first.Z
        @test output.sites[i].frac_coords == specpair.second[ired]
    end
end
