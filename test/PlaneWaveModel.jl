using Test
using DFTK: PlaneWaveModel
using LinearAlgebra

include("testcases.jl")

function test_pw_cutoffs(testcase, Ecut, fft_size)
    model = Model(testcase.lattice, testcase.n_electrons)
    pw = PlaneWaveModel(model, fft_size, Ecut, testcase.kcoords,
                        testcase.kweights, testcase.ksymops)

    for (ik, kpt) in enumerate(pw.kpoints)
        for G in kpt.basis
            @test sum(abs2, model.recip_lattice * (kpt.coordinate + G)) ≤ 2 * Ecut
        end
    end
end

@testset "PlaneWaveBasis: Check struct construction" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.n_electrons)
    pw = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords,
                        silicon.kweights, silicon.ksymops)

    @test pw.model.lattice == lattice
    @test pw.model.recip_lattice ≈ 2π * inv(lattice)
    @test pw.model.unit_cell_volume ≈ det(lattice)
    @test pw.model.recip_cell_volume ≈ (2π)^3 * det(inv(lattice))

    @test pw.Ecut == 3
    @test pw.fft_size == Tuple(fft_size)


    g_start = -ceil.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    g_stop  = floor.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    g_all = vec(collect(basis_Cρ(pw)))

    for (ik, kcoord) in enumerate(silicon.kcoords)
        kpt = pw.kpoints[ik]
        @test kpt.coordinate == kcoord

        for (ig, G) in enumerate(kpt.basis)
            @test g_start <= G <= g_stop
        end
        @test g_all[kpt.mapping] == kpt.basis
    end
    @test pw.kweights == kweights
end

@testset "PlaneWaveBasis: Energy cutoff is respected" begin
    test_pw_cutoffs(silicon, 4.0, [15, 15, 15])
    test_pw_cutoffs(silicon, 3.0, [15, 13, 13])
    test_pw_cutoffs(silicon, 4.0, [11, 13, 11])
end

