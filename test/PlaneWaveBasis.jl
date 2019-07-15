using Test
using DFTK: PlaneWaveBasis
using LinearAlgebra

include("silicon_testcases.jl")

function test_pw_cutoffs(lattice, kpoints, kweights, Ecut, grid_size)
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)
    for (ik, k) in enumerate(kpoints)
        for G in pw.basis_wf[ik]
            @test sum(abs2, pw.recip_lattice * (k + G)) ≤ 2 * Ecut
        end
    end
end

@testset "PlaneWaveBasis: Check struct construction" begin
    Ecut = 3
    grid_size = [15, 15, 15]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)

    @test pw.lattice == lattice
    @test pw.recip_lattice ≈ 2π * inv(lattice)
    @test pw.unit_cell_volume ≈ det(lattice)
    @test pw.recip_cell_volume ≈ (2π)^3 * det(inv(lattice))

    @test pw.Ecut == 3
    @test pw.grid_size == grid_size
    @test pw.kpoints == kpoints

    g_start = -ceil.(Int, (pw.grid_size .- 1) ./ 2)
    g_stop  = floor.(Int, (pw.grid_size .- 1) ./ 2)
    for kbasis in pw.basis_wf
        for G in kbasis
            @test g_start <= G <= g_stop
        end
    end
    @test pw.kweights == kweights
end

@testset "PlaneWaveBasis: Energy cutoff is respected" begin
    test_pw_cutoffs(lattice, kpoints, kweights, 4.0, [15, 15, 15])
    test_pw_cutoffs(lattice, kpoints, kweights, 3.0, [15, 13, 13])
    test_pw_cutoffs(lattice, kpoints, kweights, 4.0, [11, 13, 11])
end
