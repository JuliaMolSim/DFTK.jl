using Test
using DFTK: load_psp, Element, charge_nuclear, charge_ionic, n_elec_core, n_elec_valence

@testset "Check constructing species without psp" begin
    spec = Element(12)

    @test spec.Znuc == 12
    @test spec.psp === nothing

    @test charge_nuclear(spec) == 12
    @test charge_ionic(spec) == 12
    @test n_elec_valence(spec) == charge_ionic(spec)
    @test n_elec_core(spec) == 0
end

@testset "Check constructing species with psp" begin
    spec = Element(12, psp=load_psp("hgh/lda/C-q4"))

    @test spec.Znuc == 12
    @test spec.psp !== nothing
    @test spec.psp.identifier == "hgh/lda/c-q4"

    @test charge_nuclear(spec) == 12
    @test charge_ionic(spec) == 4
    @test n_elec_valence(spec) == 4
    @test n_elec_core(spec) == 8
end
