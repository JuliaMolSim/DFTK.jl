using Test
using DFTK: load_psp, list_psp

@testset "Check reading all HGH pseudos" begin
    for identifier in list_psp()
        psp = load_psp(identifier)
        @test psp.identifier == identifier
    end
end
