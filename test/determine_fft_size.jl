using Test
using LinearAlgebra
using DFTK: determine_fft_size, determine_fft_size_precise
using DFTK

include("testcases.jl")

@testset "Test determine_fft_size on Silicon" begin
    @test determine_fft_size(silicon.lattice,  3, supersampling=2) == [15, 15, 15]
    @test determine_fft_size(silicon.lattice,  4, supersampling=2) == [15, 15, 15]
    @test determine_fft_size(silicon.lattice,  5, supersampling=2) == [18, 18, 18]
    @test determine_fft_size(silicon.lattice, 15, supersampling=2) == [27, 27, 27]
    @test determine_fft_size(silicon.lattice, 25, supersampling=2) == [36, 36, 36]
    @test determine_fft_size(silicon.lattice, 30, supersampling=2) == [40, 40, 40]

    # Test the model interface as well
    model = Model(silicon.lattice; n_electrons=silicon.n_electrons)
    @test determine_fft_size(model, 30) == [40, 40, 40]
    @test determine_fft_size(model, 30, supersampling=1.8) == [36, 36, 36]
end

@testset "Test determine_fft_size on skewed lattice" begin
    lattice = Diagonal([1, 1e-12, 1e-12])
    @test determine_fft_size(lattice, 15, supersampling=2) == [5, 1, 1]
    @test determine_fft_size(lattice, 300, supersampling=2) == [18, 1, 1]
end

@testset "Test determine_fft_size_precise" begin
    atoms = [ElementPsp(:Si, psp=load_psp(silicon.psp)) => silicon.positions]
    model = Model(silicon.lattice; n_electrons=silicon.n_electrons, atoms=atoms)

    function fft_precise(model, kcoords, Ecut; supersampling=2)
        fft_size_fast = determine_fft_size(model.lattice, Ecut, supersampling=supersampling)
        kpoints = DFTK.build_kpoints(model, fft_size_fast, kcoords, Ecut; variational=true)
        fft_size = determine_fft_size_precise(model.lattice, Ecut, kpoints; supersampling=supersampling)
        @test all(fft_size .<= fft_size_fast)
        fft_size
    end

    @test fft_precise(model, silicon.kcoords, 20, supersampling=2)   == [30, 30, 30]
    @test fft_precise(model, silicon.kcoords, 20, supersampling=1.6) == [24, 24, 24]
end
