using Test
using DFTK: PlaneWaveModel, Model, basis_Cρ, index_Cρ
using LinearAlgebra

include("testcases.jl")

function test_pw_cutoffs(testcase, Ecut, fft_size)
    model = Model(testcase.lattice, testcase.n_electrons)
    pw = PlaneWaveModel(model, fft_size, Ecut, testcase.kcoords, testcase.ksymops)

    for (ik, kpt) in enumerate(pw.kpoints)
        for G in kpt.basis
            @test sum(abs2, model.recip_lattice * (kpt.coordinate + G)) ≤ 2 * Ecut
        end
    end
end

@testset "PlaneWaveModel: Check struct construction" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.n_electrons)
    pw = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords, silicon.ksymops)

    @test pw.model.lattice == silicon.lattice
    @test pw.model.recip_lattice ≈ 2π * inv(silicon.lattice)
    @test pw.model.unit_cell_volume ≈ det(silicon.lattice)
    @test pw.model.recip_cell_volume ≈ (2π)^3 * det(inv(silicon.lattice))

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
    @test pw.kweights == [1, 8, 6, 12] / 27
end

@testset "PlaneWaveModel: Energy cutoff is respected" begin
    test_pw_cutoffs(silicon, 4.0, [15, 15, 15])
    test_pw_cutoffs(silicon, 3.0, [15, 13, 13])
    test_pw_cutoffs(silicon, 4.0, [11, 13, 11])
end

@testset "PlaneWaveModel: Check cubic basis and cubic index" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.n_electrons)
    pw = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords, silicon.ksymops)
    g_all = collect(basis_Cρ(pw))

    for i in 1:15, j in 1:15, k in 1:15
        @test index_Cρ(pw, g_all[i, j, k]) == CartesianIndex(i, j, k)
    end
    @test index_Cρ(pw, [15, 1, 1]) === nothing
    @test index_Cρ(pw, [-15, 1, 1]) === nothing
end
