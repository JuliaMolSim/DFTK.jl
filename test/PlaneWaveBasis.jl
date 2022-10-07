using Test
using DFTK
using DFTK: index_G_vectors
using LinearAlgebra

include("testcases.jl")

function test_pw_cutoffs(testcase, Ecut, fft_size)
    model = Model(testcase.lattice; testcase.n_electrons)
    basis = PlaneWaveBasis(model; Ecut, fft_size, kgrid=(2, 5, 5), kshift=[1, 0, 0]/2)

    for (ik, kpt) in enumerate(basis.kpoints)
        for G in G_vectors(basis, kpt)
            @test sum(abs2, model.recip_lattice * (kpt.coordinate + G)) ≤ 2 * Ecut
        end
    end
end

@testset "PlaneWaveBasis: Check struct construction" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)

    @test basis.model.lattice == silicon.lattice
    @test basis.model.recip_lattice ≈ 2π * inv(silicon.lattice)
    @test basis.model.unit_cell_volume ≈ det(silicon.lattice)
    @test basis.model.recip_cell_volume ≈ (2π)^3 * det(inv(silicon.lattice))

    @test basis.Ecut == 3
    @test basis.fft_size == Tuple(fft_size)

    g_start = -ceil.(Int, (Vec3(basis.fft_size) .- 1) ./ 2)
    g_stop  = floor.(Int, (Vec3(basis.fft_size) .- 1) ./ 2)
    g_all = vec(collect(G_vectors(basis)))

    for (ik, kpt) in enumerate(basis.kpoints)
        kpt = basis.kpoints[ik]
        @test kpt.coordinate == silicon.kcoords[basis.krange_thisproc[ik]]

        for (ig, G) in enumerate(G_vectors(basis, kpt))
            @test g_start <= G <= g_stop
        end
        @test g_all[kpt.mapping] == G_vectors(basis, kpt)
    end
    @test basis.kweights == ([1, 8, 6, 12] / 27)[basis.krange_thisproc]
end

@testset "PlaneWaveBasis: Energy cutoff is respected" begin
    test_pw_cutoffs(silicon, 4.0, [15, 15, 15])
    test_pw_cutoffs(silicon, 3.0, [15, 13, 13])
    test_pw_cutoffs(silicon, 4.0, [11, 13, 11])
end

@testset "PlaneWaveBasis: Check cubic basis and cubic index" begin
    model = Model(silicon.lattice; silicon.n_electrons)
    basis = PlaneWaveBasis(model; Ecut=3, fft_size=(15, 15, 15), kgrid=(1, 1, 1))
    g_all = collect(G_vectors(basis))

    for i in 1:15, j in 1:15, k in 1:15
        @test index_G_vectors(basis, g_all[i, j, k]) == CartesianIndex(i, j, k)
    end
    @test index_G_vectors(basis, [15, 1, 1]) === nothing
    @test index_G_vectors(basis, [-15, 1, 1]) === nothing
end

@testset "PlaneWaveBasis: Check index for kpoints" begin
    Ecut = 3
    fft_size = [7, 9, 11]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
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

@testset "PlaneWaveBasis: kpoint mapping" begin
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model; Ecut=3, kgrid=(2, 2, 2), fft_size=[7, 9, 11],
                           kshift=ones(3)/2)

    for kpt in basis.kpoints
        Gs_basis = collect(G_vectors(basis))
        Gs_kpt   = collect(G_vectors(basis, kpt))
        for i in 1:length(kpt.mapping)
            @test Gs_basis[kpt.mapping[i]] == Gs_kpt[i]
        end
        for i in keys(kpt.mapping_inv)
            @test Gs_basis[i] == Gs_kpt[kpt.mapping_inv[i]]
        end
    end
end

@testset "PlaneWaveBasis: Check G_vector-like accessor functions" begin
    Ecut = 3
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)

    # `isapprox` and not `==` because of https://github.com/JuliaLang/julia/issues/46849
    atol = 20eps(eltype(basis))

    @test length(G_vectors(fft_size)) == prod(fft_size)
    @test length(r_vectors(basis))    == prod(fft_size)

    @test G_vectors(basis)      ≈ G_vectors(fft_size) atol=atol
    @test G_vectors_cart(basis) ≈ map(G -> model.recip_lattice * G,
                                      G_vectors(fft_size)) atol=atol
    @test r_vectors_cart(basis) ≈ map(r -> model.lattice * r, r_vectors(basis)) atol=atol

    for kpt in basis.kpoints
        @test length(G_vectors(basis, kpt)) == length(kpt.mapping)

        @test G_vectors_cart(basis, kpt)      ≈ map(G -> model.recip_lattice * G,
                                                    G_vectors(basis, kpt))      atol=atol
        @test Gplusk_vectors(basis, kpt)      ≈ map(G -> G + kpt.coordinate,
                                                    G_vectors(basis, kpt))      atol=atol
        @test Gplusk_vectors_cart(basis, kpt) ≈ map(q -> model.recip_lattice * q,
                                                    Gplusk_vectors(basis, kpt)) atol=atol
    end
end
