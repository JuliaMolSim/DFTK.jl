using Test
using DFTK: parse_tm_file

@testset "Check reading" begin
    psp = parse_tm_file("/Users/jasonlehto/git/Dino.jl/dev/DFTK/data/psp/tm/69tm.pspnc")

    @test occursin(r"Tm", psp.description)
    @test occursin(r"Troullier-Martins", psp.description)
    @test psp.atomicNumber == 69
    @test psp.valenceElectrons == 15
    @test psp.lmax == 3
    @test psp.lloc == 0
    @test psp.numGridPoints == 2001
    @test psp.r2well == 0.0
    @test psp.numProjectorFctns == [0,1,1,1]
    @test psp.pspCoreRadius ≈ [2.7274861, 3.4157018, 2.5944632, 2.7965336]
    @test psp.rms ≈ zeros(4)
    @test psp.energiesKB ≈ [zeros(2) for _ in 1:4]
    @test psp.epsatm ≈ zeros(4)
    @test psp.rchrg ≈ 3.70189202588749
    @test psp.fchrg ≈ 0.04664758709060
    @test psp.totCoreChrg ≈ 2.06513006799856
    @test sum(length(vals) for vals in psp.PspVals) ≈ 4 * 2001
    @test sum(length(vals) for vals in psp.firstProjectorVals) ≈ 4 * 2001
    @test isempty(psp.secondProjectorVals)
end