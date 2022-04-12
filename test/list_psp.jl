using Test
using DFTK: load_psp, list_psp

@testset "Check reading all HGH pseudos" begin
    for record in list_psp()
        psp = load_psp(record.identifier)
        @test psp.identifier == record.identifier
        @test psp.Zion == record.n_elec_valence
    end
end
