@testitem "ortho_qr is first order accurate" begin
    using DFTK: ortho_lowdin, ortho_qr
    using Random

    X = ortho_lowdin(randn(ComplexF64, 10, 3))

    dX = randn(size(X))
    # Project onto tangent spaace
    dX -= X * X' * dX
    h = 1e-5

    @test ortho_qr(X + h * dX) ≈ X + h * dX atol=1e-8
end