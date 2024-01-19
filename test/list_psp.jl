@testitem "Check reading all HGH pseudos" tags=[:psp] begin
    using DFTK: load_psp, list_psp

    for record in list_psp()
        psp = load_psp(record.identifier)
        @test psp.identifier == record.identifier
        @test psp.Zion == record.n_elec_valence
    end
end
