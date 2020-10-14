using Test
using Unitful
using UnitfulAtomic
using DFTK: to_energy

@testset "Check energy conversion backed by Unitful" begin
    @test to_energy(300u"K") ≈ 0.0009500434690331574
    @test to_energy(1u"J") ≈ 2.2937122784e17
    @test to_energy(300u"Eh_au") == 300
    # If passing a number, assume it's already in atomic units.
    @test to_energy(300) == 300
end
