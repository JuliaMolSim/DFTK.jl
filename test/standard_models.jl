@testitem "Standard DFT model construction" setup=[TestCases] tags=[:minimal] begin
    using DFTK
    using PseudoPotentialData
    using DFTK: _parse_functionals
    using .TestCases: silicon
    using DftFunctionals: identifier
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    system = DFTK.atomic_system(silicon.lattice, silicon.atoms, silicon.positions)

    @testset "reduced HF" begin
        (; dftterms, model_name) = _parse_functionals([])
        @test length(dftterms) == 1
        @test dftterms[1] isa Xc
        @test isempty(dftterms[1].functionals)
        @test model_name == "rHF"
    end

    @testset "lda from symbols" begin
        (; dftterms, model_name) = _parse_functionals([:lda_x, :lda_c_pw])
        @test length(dftterms) == 1
        @test dftterms[1] isa Xc
        @test identifier.(dftterms[1].functionals) == [:lda_x, :lda_c_pw]
        @test model_name == "lda_x+lda_c_pw"
    end

    @testset "lda from object" begin
        lda = LDA(; potential_threshold=1e-2)
        (; dftterms, model_name) = _parse_functionals([lda])
        @test length(dftterms) == 1
        @test dftterms[1] isa Xc
        @test identifier.(dftterms[1].functionals) == [:lda_x, :lda_c_pw]
        @test dftterms[1].potential_threshold == 1e-2
        @test model_name == "lda_x+lda_c_pw"
    end

    @testset "xc object plus symbol is error" begin
        xc = Xc([:lda_x, :lda_c_pw])
        @test_throws ArgumentError model_DFT(system; functionals=[xc, :lda_x], pseudopotentials)
    end

    @testset "HF is error" begin
        exx = ExactExchange()
        @test_throws ArgumentError model_DFT(system; functionals=[exx], pseudopotentials)
    end

    @testset "hybrid with implicit exx" begin
        (; dftterms, model_name) = _parse_functionals([:hyb_gga_xc_pbeh])
        @test length(dftterms) == 2
        @test dftterms[1] isa Xc
        @test identifier.(dftterms[1].functionals) == [:hyb_gga_xc_pbeh]
        @test dftterms[2] isa ExactExchange
        @test dftterms[2] == ExactExchange(; scaling_factor=0.25)
        @test model_name == "hyb_gga_xc_pbeh"
    end

    @testset "hybrid with explicit exx" begin
        exx = ExactExchange(; scaling_factor=0.2,
                            interaction_model=TruncatedCoulomb(Spherically()))
        (; dftterms, model_name) = _parse_functionals([:hyb_gga_xc_pbeh, exx])
        @test length(dftterms) == 2
        @test dftterms[1] isa Xc
        @test identifier.(dftterms[1].functionals) == [:hyb_gga_xc_pbeh]
        @test dftterms[2] isa ExactExchange
        @test dftterms[2] == exx
        @test model_name == "hyb_gga_xc_pbeh"
    end

    @testset "pbe0 with exx parameters" begin
        pbe0 = PBE0(; potential_threshold=1e-2, interaction_model=TruncatedCoulomb(Spherically()))
        (; dftterms, model_name) = _parse_functionals(pbe0)
        @test length(dftterms) == 2
        @test dftterms[1] isa Xc
        @test identifier.(dftterms[1].functionals) == [:hyb_gga_xc_pbeh]
        @test dftterms[1].potential_threshold == 1e-2
        @test dftterms[2] isa ExactExchange
        @test dftterms[2] == ExactExchange(; scaling_factor=0.25,
                                           interaction_model=TruncatedCoulomb(Spherically()))
    end
end
