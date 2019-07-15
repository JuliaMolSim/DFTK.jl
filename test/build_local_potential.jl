using Test
using DFTK: PlaneWaveBasis, r_to_G!, build_local_potential, basis_ρ

include("silicon_testcases.jl")

@testset "Check build_local_potential using Coulomb potential" begin
    Ecut = 4
    grid_size = [15, 15, 15]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)

    @testset "Construction using a single species" begin
        Z = 12
        pot_coulomb(G) = -Z / sum(abs2, pw.recip_lattice * G)

        # Shifting by a lattice vector should not make a difference:
        pot0 = build_local_potential(pw, [[0, 0, 0]], pot_coulomb)
        pot1 = build_local_potential(pw, [[0, 1, 0]], pot_coulomb)
        @test pot0.values_real ≈ pot1.values_real

        # Results are additive
        pot1 = build_local_potential(pw, [[0, 1/3, 0]], pot_coulomb)
        pot2 = build_local_potential(pw, [[0, 1/3, 0], [0,0,0]], pot_coulomb)
        @test pot0.values_real + pot1.values_real ≈ pot2.values_real

        # pot3 back to the PW basis to check we get the 1/|G|^2 behaviour
        pot3 = build_local_potential(pw, [[0, 1/8, 0]], pot_coulomb)
        values_fourier = zeros(ComplexF64, prod(pw.grid_size))
        r_to_G!(pw, pot3.values_real, values_fourier)

        lattice_vector = lattice * [0, 1/8, 0]
        reference = [(4π / pw.unit_cell_volume * pot_coulomb(G)
                         * cis(dot(pw.recip_lattice * G, lattice_vector)))
                     for G in basis_ρ(pw)]
        reference[pw.idx_DC] = 0
        @test reshape(reference, :) ≈ values_fourier
    end

    @testset "Construction using multiple species" begin
        pot_coulomb(G, Z) = -Z / sum(abs2, pw.recip_lattice * G)

        # Compute separate potential terms for reference
        pot_Si_1 = build_local_potential(pw, [[0, 1/8, 0]], G -> pot_coulomb(G, 14))
        pot_Si_2 = build_local_potential(pw, [[0, -1/8, 0]], G -> pot_coulomb(G, 14))
        pot_C = build_local_potential(pw, [[0, 1/4, 0]], G -> pot_coulomb(G, 6))
        reference = pot_Si_1.values_real + pot_Si_2.values_real + pot_C.values_real

        posmap = Dict("Si" => [[0, 1/8, 0], [0, -1/8, 0]], "C" => [[0, 1/4, 0]])
        potmap = Dict("Si" => G -> pot_coulomb(G, 14), "C" => G -> pot_coulomb(G, 6))
        pot = build_local_potential(pw, posmap, potmap)
        @test pot.values_real ≈ reference
    end
end
