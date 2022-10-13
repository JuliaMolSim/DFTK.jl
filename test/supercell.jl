using Test
using DFTK
using LinearAlgebra

function create_supercell_by_hand(lattice, atoms, positions, supercell_size)
    lattice_supercell = reduce(hcat, supercell_size .* eachcol(lattice))

    # Compute atoms reduced coordinates in the supercell
    atoms_supercell = eltype(atoms)[]
    positions_supercell = eltype(positions)[]
    nx, ny, nz = supercell_size

    for (atom, position) in zip(atoms, positions)
        append!(positions_supercell, [(position .+ [i;j;k]) ./ [nx, ny, nz]
                                      for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1])
        append!(atoms_supercell, vcat([atom for _ in 1:nx*ny*nz]...))
    end

    (; lattice=lattice_supercell, atoms=atoms_supercell, positions=positions_supercell)
end

@testset "Supercell creation" begin
    Mg        = ElementPsp(:Mg, psp=load_psp("hgh/lda/Mg-q2"))
    Si        = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    X         = ElementGaussian(1.0, 0.5, :X)
    atoms     = [Mg, Si, X]
    positions = [rand(3) for _ in 1:3]
    lattice   = rand(3, 3)

    supercell_size = rand(1:10, 3)
    supercell = create_supercell(lattice, atoms, positions, supercell_size)
    supercell_ref = create_supercell_by_hand(lattice, atoms, positions, supercell_size)

    @testset "Size consistency" begin
        @test length(supercell.positions) == length(supercell_ref.positions)
        @test length(supercell.positions) == length(supercell.atoms)
        @test length(supercell.atoms) == length(supercell_ref.atoms)
    end

    tolerance = sqrt(eps(eltype(lattice)))
    @testset "Values consistency" begin
        @test norm(supercell.lattice - supercell_ref.lattice) < tolerance

        mapping = zeros(Int, length(supercell.positions))
        for (i, position) in enumerate(supercell.positions)
            indices = findall(norm.(supercell_ref.positions .- Ref(position)) .< tolerance)
            mapping[i] = only(indices)
        end
        @test sort(mapping) == 1:length(supercell.positions)

        @test supercell.atoms == supercell_ref.atoms[mapping]
    end
end
