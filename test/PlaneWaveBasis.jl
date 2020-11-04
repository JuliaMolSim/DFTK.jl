using Test
using DFTK: PlaneWaveBasis, Model, G_vectors, index_G_vectors
using LinearAlgebra

include("testcases.jl")

function test_pw_cutoffs(testcase, Ecut, fft_size)
    model = Model(testcase.lattice, n_electrons=testcase.n_electrons)
    pw = PlaneWaveBasis(model, Ecut, testcase.kcoords, testcase.ksymops; fft_size=fft_size)

    for (ik, kpt) in enumerate(pw.kpoints)
        for G in G_vectors(kpt)
            @test sum(abs2, model.recip_lattice * (kpt.coordinate + G)) ≤ 2 * Ecut
        end
    end
end

@testset "PlaneWaveBasis: Check struct construction" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons)
    pw = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    @test pw.model.lattice == silicon.lattice
    @test pw.model.recip_lattice ≈ 2π * inv(silicon.lattice)
    @test pw.model.unit_cell_volume ≈ det(silicon.lattice)
    @test pw.model.recip_cell_volume ≈ (2π)^3 * det(inv(silicon.lattice))

    @test pw.Ecut == 3
    @test pw.fft_size == Tuple(fft_size)


    g_start = -ceil.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    g_stop  = floor.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    g_all = vec(collect(G_vectors(pw)))

    for (ik, kpt) in enumerate(pw.kpoints)
        kpt = pw.kpoints[ik]
        @test kpt.coordinate == silicon.kcoords[pw.krange_thisproc[ik]]

        for (ig, G) in enumerate(G_vectors(kpt))
            @test g_start <= G <= g_stop
        end
        @test g_all[kpt.mapping] == G_vectors(kpt)
    end
    @test pw.kweights == ([1, 8, 6, 12] / 27)[pw.krange_thisproc]
end

@testset "PlaneWaveBasis: Energy cutoff is respected" begin
    test_pw_cutoffs(silicon, 4.0, [15, 15, 15])
    test_pw_cutoffs(silicon, 3.0, [15, 13, 13])
    test_pw_cutoffs(silicon, 4.0, [11, 13, 11])
end

@testset "PlaneWaveBasis: Check cubic basis and cubic index" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons)
    pw = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    g_all = collect(G_vectors(pw))

    for i in 1:15, j in 1:15, k in 1:15
        @test index_G_vectors(pw, g_all[i, j, k]) == CartesianIndex(i, j, k)
    end
    @test index_G_vectors(pw, [15, 1, 1]) === nothing
    @test index_G_vectors(pw, [-15, 1, 1]) === nothing
end

@testset "PlaneWaveBasis: Check index for kpoints" begin
    Ecut = 3
    fft_size = [7, 9, 11]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    g_all = collect(G_vectors(basis))

    for kpt in basis.kpoints
        for (iball, ifull) in enumerate(kpt.mapping)
            @test index_G_vectors(basis, kpt, g_all[ifull]) == iball
        end

        if kpt.coordinate == [1/3, 1/3, 0]
            @test index_G_vectors(basis, kpt, [-2, -3, -1]) == 62
        else
            @test index_G_vectors(basis, kpt, [-2, -3, -1]) === nothing
        end
        @test index_G_vectors(basis, kpt, [15, 1, 1]) === nothing
        @test index_G_vectors(basis, kpt, [-15, 1, 1]) === nothing
    end
end
