@testitem "Test reduced / Cartesian conversion" setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    silicon = TestCases.silicon

    Ecut = 3
    fft_size = [13, 15, 14]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)

    rred = randn(3)  # reduced "position"
    fred = randn(3)  # reduced "force"
    pred = randn(3)  # reduced "reciprocal vector"
    @test(  dot(DFTK.covector_red_to_cart(model, fred), DFTK.vector_red_to_cart(model, rred))
          ≈ dot(fred, rred))
    @test(dot(DFTK.vector_red_to_cart(model, rred), DFTK.recip_vector_red_to_cart(model, pred))
          ≈ 2π * dot(rred, pred))

    @test DFTK.vector_cart_to_red(model, DFTK.vector_red_to_cart(model, rred)) ≈ rred
    @test DFTK.covector_cart_to_red(model, DFTK.covector_red_to_cart(model, fred)) ≈ fred
    @test DFTK.recip_vector_cart_to_red(model, DFTK.recip_vector_red_to_cart(model, pred)) ≈ pred

    rcart = randn(3)
    fcart = randn(3)
    @test dot(DFTK.covector_cart_to_red(model, fcart),
              DFTK.vector_cart_to_red(model, rcart))   ≈ dot(fcart, rcart)

    Ared = randn(3, 3)  # real-space       "symmetry"
    Bred = randn(3, 3)  # reciprocal-space "symmetry"
    @test(  dot(DFTK.comatrix_red_to_cart(model, Bred) * DFTK.covector_red_to_cart(model, fred),
                DFTK.matrix_red_to_cart(model, Ared) * DFTK.vector_red_to_cart(model, rred))
          ≈ dot(Bred * fred, Ared * rred))

    @test DFTK.comatrix_cart_to_red(model, DFTK.comatrix_red_to_cart(model, Bred)) ≈ Bred
    @test DFTK.matrix_cart_to_red(model,   DFTK.matrix_red_to_cart(model,   Ared)) ≈ Ared
end

@testitem "Violation of charge neutrality" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    # This is fine as no Coulomb electrostatics
    Model(silicon.lattice; εF=0.1)
    Model(silicon.lattice; n_electrons=1)

    # Violation of charge neutrality should throw for models with atoms.
    @test_throws ErrorException model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                                          εF=0.1, functionals=LDA())
    @test_throws ErrorException model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                                          n_electrons=1, functionals=LDA())
end

@testitem "Test symmetries in update constructor" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon
    pos_broken = [silicon.positions[1], silicon.positions[2] .+ [0, 0, 0.1]]
    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                      functionals=LDA())
    model_broken = model_DFT(silicon.lattice, silicon.atoms, pos_broken;
                             functionals=LDA())
    @test length(model_broken.symmetries) < length(model.symmetries)

    # Test symmetries=true triggers a re-determination of the symmetries
    model_update = Model(model; positions=pos_broken, symmetries=true)
    @test model_update.symmetries == model_broken.symmetries
end

@testitem "Test pseudopotential family and data of Model" setup=[TestCases] begin
    using DFTK
    using DFTK: pseudofamily
    using PseudoPotentialData
    silicon = TestCases.silicon

    Si_hgh = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    Si_lda = ElementPsp(:Si, TestCases.pd_lda_family)
    Ga_lda = ElementPsp(:Ga, TestCases.pd_lda_family)
    let model = model_DFT(silicon.lattice, [Si_lda, Si_lda], silicon.positions;
                          functionals=LDA())
        @test pseudofamily(model) == TestCases.pd_lda_family

        cutoffs = recommended_cutoff(model)
        @test 16.0 == cutoffs.Ecut
        @test  2.0 == cutoffs.supersampling
        @test 64.0 == cutoffs.Ecut_density
    end
    let model = model_DFT(silicon.lattice, [Si_lda, Ga_lda], silicon.positions;
                          functionals=LDA())
        @test pseudofamily(model) == TestCases.pd_lda_family

        cutoffs = recommended_cutoff(model)
        @test  40.0 == cutoffs.Ecut
        @test   2.0 == cutoffs.supersampling
        @test 160.0 == cutoffs.Ecut_density
    end

    let model = model_DFT(silicon.lattice, [Si_lda, Si_hgh], silicon.positions;
                          functionals=LDA())
        @test isnothing(pseudofamily(model))

        cutoffs = recommended_cutoff(model)
        @test ismissing(cutoffs.Ecut)
        @test 2.0 == cutoffs.supersampling
        @test ismissing(cutoffs.Ecut_density)
    end
    let model = model_DFT(silicon.lattice, [Si_hgh, Si_hgh], silicon.positions;
                          functionals=LDA())
        @test pseudofamily(model) == PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")

        cutoffs = recommended_cutoff(model)
        @test ismissing(cutoffs.Ecut)
        @test 2.0 == cutoffs.supersampling
        @test ismissing(cutoffs.Ecut_density)
    end
    let model = Model(silicon.lattice)
        @test isnothing(pseudofamily(model))
        cutoffs = recommended_cutoff(model)
        @test ismissing(cutoffs.Ecut)
        @test 2.0 == cutoffs.supersampling
        @test ismissing(cutoffs.Ecut_density)
    end
end
