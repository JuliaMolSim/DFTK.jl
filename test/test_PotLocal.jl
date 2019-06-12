include("testcases_silicon.jl")

@testset "Check PotLocal using Coulomb potential" begin
    Ecut = 4
    grid_size = [15, 15, 15]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)

    @testset "Construction using a single species" begin
        Z = 12
        pot_coulomb(G) = -Z / sum(abs2, G)

        # Shifting by a lattice vector should not make a difference:
        pot0 = PotLocal(pw, [0, 0, 0], pot_coulomb)
        pot1 = PotLocal(pw, [[0, 1, 0]], pot_coulomb)
        @test pot0.values_real ≈ pot1.values_real

        # Results are additive
        pot1 = PotLocal(pw, [[0, 1/3, 0]], pot_coulomb)
        pot2 = PotLocal(pw, [[0, 1/3, 0], [0,0,0]], pot_coulomb)
        @test pot0.values_real + pot1.values_real ≈ pot2.values_real

        # pot3 back to the PW basis to check we get the 1/|G|^2 behaviour
        pot3 = PotLocal(pw, [[0, 1/8, 0]], pot_coulomb)
        values_fourier = zeros(ComplexF64, prod(pw.grid_size))
        DFTK.R_to_G!(pw, pot3.values_real, values_fourier)

        lattice_vector = lattice * [0, 1/8, 0]
        Gvectors = [pw.recip_lattice * G for G in gcoords(pw)]
        reference = [4π / pw.unit_cell_volume * pot_coulomb(G) * cis(dot(G, lattice_vector))
                     for G in Gvectors]
        reference[pw.idx_DC] = 0
        @test reference ≈ values_fourier
    end

    @testset "Construction using multiple species and a parametrised potential" begin
        pot_coulomb(G, Z) = -Z / sum(abs2, G)

        # Compute separate potential terms for reference
        pot_Si_1 = PotLocal(pw, [0, 1/8, 0], G -> pot_coulomb(G, 14))
        pot_Si_2 = PotLocal(pw, [0, -1/8, 0], G -> pot_coulomb(G, 14))
        pot_C = PotLocal(pw, [0, 1/4, 0], G -> pot_coulomb(G, 6))
        reference = pot_Si_1.values_real + pot_Si_2.values_real + pot_C.values_real

        charge_map = Dict("Si" => 14, "C" => 6)
        pot = PotLocal(pw, ("Si" => [[0, 1/8, 0], [0, -1/8, 0]], "C" => [[0, 1/4, 0]]),
                       pot_coulomb, parameters=charge_map)

        @test pot.values_real ≈ reference
    end

    @testset "Construction using multiple species with their own potential" begin
        abs4(x) = (abs2(x)^2)
        pot_coulomb(G, Z) = -Z / sum(abs2, G)
        pot_abs4(G, Z) = -Z / sum(abs4, G)

        # Compute separate potential terms for reference
        pot_Si_1 = PotLocal(pw, [0, 1/8, 0], G -> pot_coulomb(G, 14))
        pot_Si_2 = PotLocal(pw, [0, -1/8, 0], G -> pot_coulomb(G, 14))
        pot_C = PotLocal(pw, [0, 1/4, 0], G -> pot_abs4(G, 6))
        reference = pot_Si_1.values_real + pot_Si_2.values_real + pot_C.values_real

        charge_map = Dict("Si" => 14, "C" => 6)
        potential_map = Dict("Si" => pot_coulomb, "C" => pot_abs4)
        pot = PotLocal(pw, ("Si" => [[0, 1/8, 0], [0, -1/8, 0]], "C" => [[0, 1/4, 0]]),
                       potential_map, parameters=charge_map)
        @test pot.values_real ≈ reference
    end
end
