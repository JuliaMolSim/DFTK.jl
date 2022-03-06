using Test
using DFTK
include("testcases.jl")


@testset "Test reduced / cartesian conversion" begin
    Ecut = 3
    fft_size = [13, 15, 14]
    model = Model(silicon.lattice; silicon.atoms, silicon.positions)

    rred = randn(3)  # reduced "position"
    fred = randn(3)  # reduced "force"
    qred = randn(3)  # reduced "reciprocal vector"
    @test(  dot(DFTK.covector_red_to_cart(model, fred), DFTK.vector_red_to_cart(model, rred))
          ≈ dot(fred, rred))
    @test(dot(DFTK.vector_red_to_cart(model, rred), DFTK.recip_vector_red_to_cart(model, qred))
          ≈ 2π * dot(rred, qred))

    @test DFTK.vector_cart_to_red(model, DFTK.vector_red_to_cart(model, rred)) ≈ rred
    @test DFTK.covector_cart_to_red(model, DFTK.covector_red_to_cart(model, fred)) ≈ fred
    @test DFTK.recip_vector_cart_to_red(model, DFTK.recip_vector_red_to_cart(model, qred)) ≈ qred

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
