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
    @test_throws ErrorException model_LDA(silicon.lattice, silicon.atoms, silicon.positions;
                                          εF=0.1)
    @test_throws ErrorException model_LDA(silicon.lattice, silicon.atoms, silicon.positions;
                                          n_electrons=1)
end
