using Test
using DFTK: load_psp, list_psp

@testset "Check reading all HGH pseudos" begin
    for record in list_psp()
        psp = load_psp(record.identifier)
        @test psp.identifier == record.identifier
        @test psp.Zion == record.n_elec_valence
    end
end

@testset "Check load_psp routine selectively" begin
    psp_Cu = load_psp(:Cu, functional="lda", family="hgh", core=:semicore)
    @test psp_Cu.identifier == "hgh/lda/cu-q19.hgh"

    psp_Au = load_psp(:Au, functional="pbe", family="hgh", core=:fullcore)
    @test psp_Au.identifier == "hgh/pbe/au-q11.hgh"

    for element in (:Cu, :Ni, :Au)
        sc = load_psp(element, functional="lda", family="hgh", core=:semicore)
        fc = load_psp(element, functional="lda", family="hgh", core=:fullcore)
        @test sc.Zion > fc.Zion
    end
end
