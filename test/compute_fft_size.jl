@testitem "Test compute_fft_size on Silicon" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    model = Model(silicon.lattice)
    @test compute_fft_size(model,  3, supersampling=2)   == (15, 15, 15)
    @test compute_fft_size(model,  4, supersampling=2)   == (15, 15, 15)
    @test compute_fft_size(model,  5, supersampling=2)   == (18, 18, 18)
    @test compute_fft_size(model, 15, supersampling=2)   == (27, 27, 27)
    @test compute_fft_size(model, 25, supersampling=2)   == (36, 36, 36)
    @test compute_fft_size(model, 30, supersampling=2)   == (40, 40, 40)
    @test compute_fft_size(model, 30, supersampling=1.8) == (36, 36, 36)
end

@testitem "Test compute_fft_size on skewed lattice" begin
    using DFTK
    using LinearAlgebra

    lattice = Diagonal([1, 1e-12, 1e-12])
    model   = Model(lattice)
    @test compute_fft_size(model, 15, supersampling=2)  == ( 5, 1, 1)
    @test compute_fft_size(model, 300, supersampling=2) == (18, 1, 1)
end

@testitem "Test compute_fft_size with :precise" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon
    kgrid = MonkhorstPack((3, 3, 3))
    model = Model(silicon.lattice, silicon.atoms, silicon.positions; terms=[Kinetic()])

    function fft_precise(model, kgrid, Ecut; supersampling=2)
        fft_size_fast = compute_fft_size(model, Ecut, kgrid;
                                         supersampling, algorithm=:fast)
        fft_size      = compute_fft_size(model, Ecut, kgrid;
                                         supersampling, algorithm=:precise)
        @test all(fft_size .<= fft_size_fast)
        fft_size
    end

    @test fft_precise(model, kgrid, 20, supersampling=2)   == (30, 30, 30)
    @test fft_precise(model, kgrid, 20, supersampling=1.6) == (24, 24, 24)
end
