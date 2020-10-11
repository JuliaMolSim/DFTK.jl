using Test
using Unitful
using UnitfulAtomic
using DFTK: uconvert, to_energy

@testset "Check Unitful conversions" begin
    @test uconvert(Unitful.J, 300u"K") == 4.141947e-21u"J"
    @test uconvert(UnitfulAtomic.Eh_au, 300u"K") == 0.0009500434690331574u"Eh_au"
end

@testset "Check energy conversion backed by Unitful" begin
    @test to_energy(300u"K") == 0.0009500434690331574
    @test to_energy(300u"Eh_au") == 300
    # If passing a number, assume it's already in atomic units.
    @test to_energy(300) == 300
end
