@testitem "Test construction of dielectric terms in χ0Mixing" setup=[TestCases] begin
    function build_eps_terms(mixing, scfres)
        DFTK.build_dielectric_terms_(mixing, scfres.basis; ρin=scfres.ρ, scfres...)
    end

    @testset "insulator (no temperature)" begin
        silicon  = TestCases.silicon
        model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                          functionals=PBE())
        basis = PlaneWaveBasis(model; Ecut=6, kgrid=[3, 3, 3])
        scfres = self_consistent_field(basis; tol=10, callback=identity)

        for mixing in (LdosMixing(), LdosMixing(RPA=false), LdosXcDiagonalMixing())
            # These mixings should reduce to an identity operation
            terms = build_eps_terms(mixing, scfres)
            @test isempty(terms)
        end

        @testset "LdosDielectricMixing(RPA=true)" begin
            terms = build_eps_terms(LdosDielectricMixing(RPA=true), scfres)
            @test length(terms) == 1
            kernel_terms, chi0_terms = terms[1]
            @test length(chi0_terms) == 1
            @test only(kernel_terms) isa DFTK.TermHartree
        end
    end

    @testset "insulator (temperature)" begin
        # All mixings should reduce to an identity operation
        silicon  = TestCases.silicon
        model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                          functionals=PBE(), temperature=0.01)
        basis = PlaneWaveBasis(model; Ecut=6, kgrid=[3, 3, 3])
        scfres = self_consistent_field(basis; tol=10, callback=identity)

        @testset "LdosMixing(RPA)" begin
            terms = build_eps_terms(LdosMixing(RPA=true), scfres)
            @test length(terms) == 1
            kernel_terms, chi0_terms = terms[1]
            @test length(chi0_terms)   == 1
            @test only(kernel_terms) isa DFTK.TermHartree
        end

        @testset "LdosMixing(RPA=false)" begin
            terms = build_eps_terms(LdosMixing(RPA=false), scfres)
            @test length(terms) == 1
            kernel_terms, chi0_terms = terms[1]
            @test length(chi0_terms)   == 1
            @test length(kernel_terms) == 2
            @test kernel_terms[1] isa DFTK.TermHartree
            @test kernel_terms[2] isa DFTK.TermXc
        end

        @testset "LdosXcDiagonalMixing()" begin
            terms = build_eps_terms(LdosXcDiagonalMixing(), scfres)
            @test length(terms) == 2
            @test length(terms[1][2]) == 1  # Ldos term
            @test length(terms[2][2]) == 1  # Diagonal chi0 term

            kernel_1 = only(terms[1][1])
            kernel_2 = only(terms[2][1])
            @test kernel_1 isa DFTK.TermHartree
            @test kernel_2 isa DFTK.TermXc
        end

        @testset "LdosDielectricMixing(RPA=true)" begin
            terms = build_eps_terms(LdosDielectricMixing(RPA=true), scfres)
            @test length(terms) == 1
            kernel_terms, chi0_terms = terms[1]
            @test length(chi0_terms) == 2
            @test only(kernel_terms) isa DFTK.TermHartree
        end

        @testset "LdosDielectricMixing(RPA=false)" begin
            terms = build_eps_terms(LdosDielectricMixing(RPA=false), scfres)
            @test length(terms) == 1
            kernel_terms, chi0_terms = terms[1]
            @test length(chi0_terms)   == 2
            @test length(kernel_terms) == 2
            @test kernel_terms[1] isa DFTK.TermHartree
            @test kernel_terms[2] isa DFTK.TermXc
        end
    end
end
