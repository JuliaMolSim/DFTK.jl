using Test
using LinearAlgebra
using DFTK

include("testcases.jl")

@testset "Test compute_fft_size on Silicon" begin
    model = Model(silicon.lattice; n_electrons=1)
    @test compute_fft_size(model,  3, supersampling=2)   == (15, 15, 15)
    @test compute_fft_size(model,  4, supersampling=2)   == (15, 15, 15)
    @test compute_fft_size(model,  5, supersampling=2)   == (18, 18, 18)
    @test compute_fft_size(model, 15, supersampling=2)   == (27, 27, 27)
    @test compute_fft_size(model, 25, supersampling=2)   == (36, 36, 36)
    @test compute_fft_size(model, 30, supersampling=2)   == (40, 40, 40)
    @test compute_fft_size(model, 30, supersampling=1.8) == (36, 36, 36)
end

@testset "Test compute_fft_size on skewed lattice" begin
    lattice = Diagonal([1, 1e-12, 1e-12])
    model   = Model(lattice; n_electrons=1)
    @test compute_fft_size(model, 15, supersampling=2)  == ( 5, 1, 1)
    @test compute_fft_size(model, 300, supersampling=2) == (18, 1, 1)
end

@testset "Test compute_fft_size with :precise" begin
    atoms = [ElementPsp(:Si, psp=load_psp(silicon.psp)) => silicon.positions]
    model = Model(silicon.lattice; atoms=atoms, terms=[Kinetic()])

    function fft_precise(model, kcoords, Ecut; supersampling=2)
        fft_size_fast = compute_fft_size(model, Ecut; supersampling, algorithm=:fast)
        fft_size      = compute_fft_size(model, Ecut, kcoords;
                                         supersampling, algorithm=:precise)
        @test all(fft_size .<= fft_size_fast)
        fft_size
    end

    @test fft_precise(model, silicon.kcoords, 20, supersampling=2)   == (30, 30, 30)
    @test fft_precise(model, silicon.kcoords, 20, supersampling=1.6) == (24, 24, 24)
end
